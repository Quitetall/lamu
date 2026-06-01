//! Per-user, in-memory token-bucket quotas for the OpenAI-compat HTTP
//! surface (ADR 0018 P2, sections 4-5).
//!
//! The KeyStore auth arm (`auth.rs require_bearer`) inserts a
//! `crate::keys::Principal` into the request extensions on a successful
//! `verify()`. This module turns `Principal.daily_token_quota` into an
//! enforced budget: each user gets a token bucket sized to their daily
//! quota, refilled linearly over a 24h window. A request `check`s remaining
//! budget BEFORE forwarding (429 on exhaustion) and `charge`s the actual
//! completion-token count AFTER the backend responds.
//!
//! Calibrated to LAMU's threat model (a handful of machine clients, ADR
//! 0018): in-memory only — NO durable usage DB (the structured tracing
//! event is the durable audit trail; a `usage` table in keys.db is a later
//! add). Process restart resets buckets to full; acceptable for a daily
//! quota on a single-host API, and fails OPEN (toward availability) rather
//! than locking a user out across a restart.
//!
//! `daily_token_quota == None` (the default key, and EVERY StaticToken/Off
//! request which carries no Principal) means UNLIMITED: `check` always
//! admits and `charge` is a cheap no-op. This is what keeps the ADR-0012
//! single-token path byte-for-byte unaffected.

use crate::keys::Principal;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// 24h refill window. `daily_token_quota` tokens are restored linearly over
/// this period; a bucket never exceeds its capacity.
const REFILL_WINDOW: Duration = Duration::from_secs(24 * 60 * 60);

/// One user's bucket. `capacity` is the daily quota (tokens). `remaining` is
/// a float so sub-token-per-second refill accrues smoothly between requests.
struct Bucket {
    capacity: f64,
    remaining: f64,
    last_refill: Instant,
}

impl Bucket {
    fn new(capacity: u32) -> Self {
        Bucket {
            capacity: capacity as f64,
            remaining: capacity as f64,
            last_refill: Instant::now(),
        }
    }

    /// Accrue refill since `last_refill`, clamped to `capacity`. Refill rate
    /// is `capacity / REFILL_WINDOW` tokens per second.
    fn refill(&mut self, now: Instant) {
        let elapsed = now.saturating_duration_since(self.last_refill).as_secs_f64();
        if elapsed <= 0.0 {
            return;
        }
        let rate = self.capacity / REFILL_WINDOW.as_secs_f64();
        self.remaining = (self.remaining + elapsed * rate).min(self.capacity);
        self.last_refill = now;
    }
}

/// Verdict from [`QuotaManager::check`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuotaCheck {
    /// Admit the request — either unlimited (no quota / no principal) or the
    /// bucket has > 0 tokens left.
    Ok,
    /// Reject: the user's bucket is exhausted. Carries the daily limit so the
    /// handler can surface it in the 429 body / Retry-After.
    Exhausted { limit: u32 },
}

/// In-memory per-user token-bucket registry. Cloneable handle semantics are
/// provided by storing it behind `Arc` in `AppState`; internally a single
/// `Mutex<HashMap<user, Bucket>>` (few users, contention negligible — same
/// posture as `KeyStore`'s cache mutex).
#[derive(Default)]
pub struct QuotaManager {
    buckets: Mutex<HashMap<String, Bucket>>,
}

impl QuotaManager {
    pub fn new() -> Self {
        QuotaManager {
            buckets: Mutex::new(HashMap::new()),
        }
    }

    /// Pre-flight admission check for `principal`. `None` (StaticToken/Off —
    /// no Principal in extensions) or `Some(p)` with `daily_token_quota ==
    /// None` is UNLIMITED → always `Ok`. Otherwise refill the user's bucket
    /// and admit iff at least one whole token remains. We do NOT pre-charge
    /// the prompt here (its cost is unknown until the backend returns usage),
    /// so this is a soft-gate: any requests that pass `check` in the same
    /// window before their `charge`s land may collectively overshoot — the
    /// overshoot is bounded by concurrency, not by one request. Once all
    /// charges settle, the next `check` rejects. Matches the ADR's "in-memory
    /// token-bucket → 429 on exhaustion" without needing tokenizer access in
    /// the gate.
    pub fn check(&self, principal: Option<&Principal>) -> QuotaCheck {
        let Some(p) = principal else { return QuotaCheck::Ok };
        let Some(quota) = p.daily_token_quota else {
            return QuotaCheck::Ok;
        };
        if quota == 0 {
            // A zero quota is a hard stop (explicitly disabled key).
            return QuotaCheck::Exhausted { limit: 0 };
        }
        let now = Instant::now();
        let mut buckets = self.buckets.lock();
        let bucket = buckets
            .entry(p.user.clone())
            .or_insert_with(|| Bucket::new(quota));
        // Capacity may have changed (key re-issued with a new quota); track it.
        if (bucket.capacity - quota as f64).abs() > f64::EPSILON {
            // Re-cap, preserving the *fraction* remaining so a quota change
            // doesn't hand out a free full bucket or zero one out.
            let frac = if bucket.capacity > 0.0 {
                (bucket.remaining / bucket.capacity).clamp(0.0, 1.0)
            } else {
                1.0
            };
            bucket.capacity = quota as f64;
            bucket.remaining = frac * quota as f64;
        }
        bucket.refill(now);
        // Admit iff at least one WHOLE token is available. A `> 0.0` threshold
        // would re-admit a just-drained bucket on the very next request, since
        // `refill` accrues a sub-token fraction in the microseconds between a
        // `charge` and the following `check`. Requiring `>= 1.0` means a
        // drained user waits `REFILL_WINDOW / quota` (one token's worth of
        // time) before the next admit — correct token-bucket behavior.
        if bucket.remaining >= 1.0 {
            QuotaCheck::Ok
        } else {
            QuotaCheck::Exhausted { limit: quota }
        }
    }

    /// Debit `tokens` from the user's bucket AFTER a successful completion.
    /// No-op for unlimited (no principal / no quota) so the StaticToken/Off
    /// path never touches the map. `remaining` may go negative (the in-flight
    /// request overshot the gate); the next `check` will then reject — this
    /// is the intended soft-limit behavior.
    pub fn charge(&self, principal: Option<&Principal>, tokens: u64) {
        let Some(p) = principal else { return };
        let Some(quota) = p.daily_token_quota else { return };
        if tokens == 0 {
            return;
        }
        let now = Instant::now();
        let mut buckets = self.buckets.lock();
        let bucket = buckets
            .entry(p.user.clone())
            .or_insert_with(|| Bucket::new(quota));
        bucket.refill(now);
        bucket.remaining -= tokens as f64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn principal(user: &str, quota: Option<u32>) -> Principal {
        Principal {
            user: user.to_string(),
            key_id: 1,
            priority: 0,
            daily_token_quota: quota,
        }
    }

    #[test]
    fn no_principal_is_unlimited() {
        let q = QuotaManager::new();
        assert_eq!(q.check(None), QuotaCheck::Ok);
        // charge(None, ..) is a no-op and never panics / never inserts.
        q.charge(None, 1_000_000);
        assert_eq!(q.check(None), QuotaCheck::Ok);
    }

    #[test]
    fn none_quota_is_unlimited() {
        let q = QuotaManager::new();
        let p = principal("alice", None);
        assert_eq!(q.check(Some(&p)), QuotaCheck::Ok);
        q.charge(Some(&p), 10_000_000);
        assert_eq!(q.check(Some(&p)), QuotaCheck::Ok);
    }

    #[test]
    fn exhaustion_returns_429_with_limit() {
        let q = QuotaManager::new();
        let p = principal("bob", Some(100));
        assert_eq!(q.check(Some(&p)), QuotaCheck::Ok);
        q.charge(Some(&p), 100); // drains exactly to zero
        assert_eq!(q.check(Some(&p)), QuotaCheck::Exhausted { limit: 100 });
    }

    #[test]
    fn overshoot_then_reject() {
        // A single request whose completion exceeds the remaining budget is
        // admitted (soft gate), then the NEXT check rejects.
        let q = QuotaManager::new();
        let p = principal("carol", Some(50));
        assert_eq!(q.check(Some(&p)), QuotaCheck::Ok);
        q.charge(Some(&p), 500); // way over
        assert_eq!(q.check(Some(&p)), QuotaCheck::Exhausted { limit: 50 });
    }

    #[test]
    fn zero_quota_is_hard_stop() {
        let q = QuotaManager::new();
        let p = principal("dave", Some(0));
        assert_eq!(q.check(Some(&p)), QuotaCheck::Exhausted { limit: 0 });
    }

    #[test]
    fn per_user_isolation() {
        let q = QuotaManager::new();
        let a = principal("alice", Some(100));
        let b = principal("bob", Some(100));
        q.charge(Some(&a), 100);
        assert_eq!(q.check(Some(&a)), QuotaCheck::Exhausted { limit: 100 });
        // bob untouched by alice's spend.
        assert_eq!(q.check(Some(&b)), QuotaCheck::Ok);
    }

    #[test]
    fn refill_restores_budget_over_time() {
        // Drive Bucket::refill directly with a synthetic clock so the test is
        // deterministic + fast (no 24h wait). capacity 86_400 over a 24h
        // window = exactly 1 token/sec.
        let mut bucket = Bucket::new(86_400);
        bucket.remaining = 0.0;
        let later = bucket.last_refill + Duration::from_secs(10);
        bucket.refill(later);
        // ~10 tokens back (1 tok/sec * 10s), clamped well under capacity.
        assert!(
            bucket.remaining >= 9.0 && bucket.remaining <= 11.0,
            "expected ~10 refilled, got {}",
            bucket.remaining
        );
    }

    #[test]
    fn refill_never_exceeds_capacity() {
        let mut bucket = Bucket::new(100);
        let way_later = bucket.last_refill + Duration::from_secs(48 * 3600);
        bucket.refill(way_later);
        assert_eq!(bucket.remaining, 100.0);
    }
}
