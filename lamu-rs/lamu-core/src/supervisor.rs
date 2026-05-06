//! Backend supervisor — restart-with-backoff + sticky quarantine.
//!
//! Direct port of `lamu/core/supervisor.py`. Same backoff schedule
//! (1s/2s/4s × 3), same JSON event format on stderr, same state machine
//! transitions on `BackendHealth`.
//!
//! The supervisor never silently fails. Every state transition emits a
//! line, every restart attempt is countable, quarantine is terminal.
//!
//! Restart hooks are typed as `FnMut() -> Result<(), Error>` so the
//! caller can return a concrete spawn error. The supervisor handles
//! both success (`Ok`) and failure (`Err`) — it does NOT panic, ever.

use crate::error::{Error, Result};
use crate::health::{BackendHealth, HealthState};
use serde_json::json;
use std::time::Duration;

/// How aggressively the supervisor retries a dead backend.
#[derive(Debug, Clone)]
pub struct RestartPolicy {
    pub max_attempts: u32,
    pub backoff_secs: Vec<u64>,
}

impl Default for RestartPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            backoff_secs: vec![1, 2, 4],
        }
    }
}

/// Operator-readable JSON line on stderr. Matches Python `_emit_event`.
fn emit_event(event: &str, fields: serde_json::Value) {
    let mut obj = json!({"event": event});
    if let Some(map) = fields.as_object() {
        for (k, v) in map {
            obj[k] = v.clone();
        }
    }
    eprintln!("{}", obj);
}

/// Coordinates restart attempts for a single backend.
///
/// Owns a `&mut BackendHealth` (passed each call) and a restart closure.
/// Tokio-friendly: takes `&mut self` rather than holding a `Notify` so
/// the caller's runtime drives the sleeps.
pub struct Supervisor<F>
where
    F: FnMut() -> Result<()>,
{
    pub backend_id: String,
    restart_fn: F,
    policy: RestartPolicy,
}

impl<F> Supervisor<F>
where
    F: FnMut() -> Result<()>,
{
    pub fn new(backend_id: impl Into<String>, restart_fn: F) -> Self {
        Self {
            backend_id: backend_id.into(),
            restart_fn,
            policy: RestartPolicy::default(),
        }
    }

    pub fn with_policy(mut self, policy: RestartPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Record a failure. Walks the backoff schedule when DEAD; quarantines
    /// once attempts exhaust. Sleeps via `tokio::time::sleep`.
    pub async fn report_failure(&mut self, health: &mut BackendHealth, err: &Error) {
        health.record_error(format!("{err}"));
        emit_event(
            "backend_failure",
            json!({
                "backend_id": self.backend_id,
                "state": format!("{:?}", health.state).to_lowercase(),
                "consecutive_errors": health.consecutive_errors,
                "error": health.last_error,
            }),
        );

        if matches!(health.state, HealthState::Dead) {
            self.attempt_restart(health).await;
        } else if matches!(health.state, HealthState::Quarantined) {
            emit_event(
                "backend_quarantined",
                json!({
                    "backend_id": self.backend_id,
                    "reason": health.last_error,
                }),
            );
        }
    }

    async fn attempt_restart(&mut self, health: &mut BackendHealth) {
        for (idx, &delay) in self.policy.backoff_secs.iter().enumerate() {
            let attempt = (idx + 1) as u32;
            if attempt > self.policy.max_attempts {
                break;
            }
            health.restart_attempts = attempt;
            emit_event(
                "backend_restart_attempt",
                json!({
                    "backend_id": self.backend_id,
                    "attempt": attempt,
                    "delay_s": delay,
                }),
            );
            tokio::time::sleep(Duration::from_secs(delay)).await;
            match (self.restart_fn)() {
                Ok(()) => {
                    health.record_success();
                    emit_event(
                        "backend_restarted",
                        json!({
                            "backend_id": self.backend_id,
                            "attempt": attempt,
                        }),
                    );
                    return;
                }
                Err(e) => {
                    emit_event(
                        "supervisor_restart_failed",
                        json!({
                            "backend_id": self.backend_id,
                            "attempt": attempt,
                            "error": format!("{e}"),
                        }),
                    );
                    continue;
                }
            }
        }

        health.force_quarantine("max restart attempts exhausted");
        emit_event(
            "backend_quarantined",
            json!({
                "backend_id": self.backend_id,
                "reason": "max restart attempts exhausted",
            }),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::health::HealthState;

    #[tokio::test]
    async fn first_failure_does_not_restart() {
        let mut h = BackendHealth::new("b");
        let calls = std::cell::Cell::new(0);
        let mut sup = Supervisor::new("b", || {
            calls.set(calls.get() + 1);
            Ok(())
        });
        sup.report_failure(&mut h, &Error::Backend("e1".into())).await;
        assert_eq!(h.state, HealthState::Degraded);
        assert_eq!(calls.get(), 0);
    }

    #[tokio::test]
    async fn dead_threshold_triggers_restart() {
        let mut h = BackendHealth::new("b");
        let calls = std::cell::Cell::new(0);
        // backoff_secs = [0,0,0] for fast test
        let mut sup = Supervisor::new("b", || {
            calls.set(calls.get() + 1);
            Ok(())
        })
        .with_policy(RestartPolicy {
            max_attempts: 3,
            backoff_secs: vec![0, 0, 0],
        });
        for i in 0..3 {
            sup.report_failure(&mut h, &Error::Backend(format!("e{i}"))).await;
        }
        // After third failure, state was Dead → restart called once.
        assert!(calls.get() >= 1);
        assert_eq!(h.state, HealthState::Healthy);
    }

    #[tokio::test]
    async fn exhausted_restarts_quarantine() {
        let mut h = BackendHealth::new("b");
        let mut sup = Supervisor::new("b", || Err(Error::Backend("nope".into())))
            .with_policy(RestartPolicy {
                max_attempts: 3,
                backoff_secs: vec![0, 0, 0],
            });
        for i in 0..3 {
            sup.report_failure(&mut h, &Error::Backend(format!("e{i}"))).await;
        }
        assert_eq!(h.state, HealthState::Quarantined);
    }
}
