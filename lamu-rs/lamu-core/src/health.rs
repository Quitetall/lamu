//! Backend health state machine. Mirrors `lamu/core/health.py`.
//!
//! Constants `DEAD_THRESHOLD` and `QUARANTINE_THRESHOLD` MUST stay in sync
//! with the Python side. There's a cross-language test in
//! `tests/test_health_constants.rs` that asserts the agreement.

use serde::Serialize;
use std::time::SystemTime;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HealthState {
    Healthy,
    Degraded,
    Dead,
    Quarantined,
}

#[derive(Debug, Clone, Serialize)]
pub struct BackendHealth {
    pub backend_id: String,
    pub state: HealthState,
    pub consecutive_errors: u32,
    pub last_error: Option<String>,
    pub last_error_unix: f64,
    pub restart_attempts: u32,
}

impl BackendHealth {
    pub const DEAD_THRESHOLD: u32 = 3;
    pub const QUARANTINE_THRESHOLD: u32 = 5;

    pub fn new(backend_id: impl Into<String>) -> Self {
        Self {
            backend_id: backend_id.into(),
            state: HealthState::Healthy,
            consecutive_errors: 0,
            last_error: None,
            last_error_unix: 0.0,
            restart_attempts: 0,
        }
    }

    pub fn record_success(&mut self) {
        if matches!(self.state, HealthState::Quarantined) {
            return; // sticky
        }
        self.consecutive_errors = 0;
        self.state = HealthState::Healthy;
        self.last_error = None;
        self.restart_attempts = 0; // recovered → reset the restart counter (#20)
    }

    pub fn record_error(&mut self, msg: impl Into<String>) {
        if matches!(self.state, HealthState::Quarantined) {
            return;
        }
        self.consecutive_errors += 1;
        self.last_error = Some(msg.into());
        self.last_error_unix = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);

        self.state = if self.consecutive_errors >= Self::QUARANTINE_THRESHOLD {
            HealthState::Quarantined
        } else if self.consecutive_errors >= Self::DEAD_THRESHOLD {
            HealthState::Dead
        } else {
            HealthState::Degraded
        };
    }

    pub fn force_quarantine(&mut self, reason: impl Into<String>) {
        self.state = HealthState::Quarantined;
        self.last_error = Some(format!("force_quarantine: {}", reason.into()));
    }

    pub fn usable(&self) -> bool {
        matches!(self.state, HealthState::Healthy | HealthState::Degraded)
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

/// Process-wide map of `backend_id -> BackendHealth`. Mirrors
/// `lamu.core.health.HealthRegistry`.
#[derive(Debug, Default)]
pub struct HealthRegistry {
    by_id: std::collections::HashMap<String, BackendHealth>,
}

impl HealthRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_or_create(&mut self, backend_id: &str) -> &mut BackendHealth {
        self.by_id
            .entry(backend_id.to_string())
            .or_insert_with(|| BackendHealth::new(backend_id))
    }

    pub fn get(&self, backend_id: &str) -> Option<&BackendHealth> {
        self.by_id.get(backend_id)
    }

    pub fn all(&self) -> &std::collections::HashMap<String, BackendHealth> {
        &self.by_id
    }

    pub fn snapshot(&self) -> std::collections::HashMap<String, serde_json::Value> {
        self.by_id.iter().map(|(k, v)| (k.clone(), v.to_json())).collect()
    }

    /// Set of backend ids currently usable (HEALTHY/DEGRADED). Quarantined
    /// or Dead backends are filtered out.
    pub fn usable_ids(&self) -> std::collections::HashSet<String> {
        self.by_id
            .iter()
            .filter(|(_, h)| h.usable())
            .map(|(k, _)| k.clone())
            .collect()
    }

    /// Remove a backend from the registry (e.g. after explicit unload).
    pub fn drop(&mut self, backend_id: &str) -> Option<BackendHealth> {
        self.by_id.remove(backend_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_healthy() {
        let h = BackendHealth::new("x");
        assert_eq!(h.state, HealthState::Healthy);
        assert!(h.usable());
        assert_eq!(h.consecutive_errors, 0);
    }

    #[test]
    fn record_error_progresses_through_states() {
        let mut h = BackendHealth::new("x");
        h.record_error("e1");
        assert_eq!(h.state, HealthState::Degraded);
        assert!(h.usable());

        h.record_error("e2");
        assert_eq!(h.state, HealthState::Degraded);

        h.record_error("e3");
        assert_eq!(h.state, HealthState::Dead);
        assert!(!h.usable());

        h.record_error("e4");
        h.record_error("e5");
        assert_eq!(h.state, HealthState::Quarantined);
    }

    #[test]
    fn record_success_resets_to_healthy() {
        let mut h = BackendHealth::new("x");
        h.record_error("oops");
        h.record_success();
        assert_eq!(h.state, HealthState::Healthy);
        assert_eq!(h.consecutive_errors, 0);
        assert!(h.last_error.is_none());
    }

    #[test]
    fn quarantine_is_sticky() {
        let mut h = BackendHealth::new("x");
        h.force_quarantine("explicit");
        h.record_success();
        assert_eq!(h.state, HealthState::Quarantined);
        h.record_error("ignored");
        assert_eq!(h.state, HealthState::Quarantined);
    }

    #[test]
    fn thresholds_match_python_constants() {
        // Python side: DEAD_THRESHOLD=3, QUARANTINE_THRESHOLD=5
        assert_eq!(BackendHealth::DEAD_THRESHOLD, 3);
        assert_eq!(BackendHealth::QUARANTINE_THRESHOLD, 5);
    }

    #[test]
    fn registry_get_or_create_idempotent() {
        let mut r = HealthRegistry::new();
        let _ = r.get_or_create("a");
        r.get_or_create("a").record_error("e1");
        let h = r.get("a").unwrap();
        assert_eq!(h.consecutive_errors, 1);
    }

    #[test]
    fn registry_usable_filters_dead() {
        let mut r = HealthRegistry::new();
        r.get_or_create("ok");
        r.get_or_create("bad");
        for _ in 0..5 {
            r.get_or_create("bad").record_error("boom");
        }
        let usable = r.usable_ids();
        assert!(usable.contains("ok"));
        assert!(!usable.contains("bad"));
    }

    #[test]
    fn registry_drop_removes() {
        let mut r = HealthRegistry::new();
        r.get_or_create("x");
        assert!(r.drop("x").is_some());
        assert!(r.get("x").is_none());
    }
}
