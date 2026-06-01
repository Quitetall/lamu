//! Prometheus metrics for the OpenAI-compat layer.
//!
//! Mirrors `lamu/api/metrics.py`. Same series names, same labels, same
//! semantics — clients can scrape either Python or Rust `lamu serve`
//! and dashboards keep working.
//!
//! All collectors live on a per-instance `Registry` (no global state).
//! `LamuMetrics::refresh(scheduler, health, queue_depths)` is called
//! from the `/metrics` handler so gauges reflect the current snapshot.

use lamu_core::health::{HealthRegistry, HealthState};
use lamu_core::scheduler::VramScheduler;
use prometheus::{
    Encoder, Gauge, GaugeVec, HistogramOpts, HistogramVec, IntCounter, IntCounterVec,
    IntGaugeVec, Opts, Registry, TextEncoder,
};
use std::collections::HashMap;

/// Numeric encoding so a gauge can plot the health state.
fn health_to_numeric(state: HealthState) -> i64 {
    match state {
        HealthState::Healthy => 2,
        HealthState::Degraded => 1,
        HealthState::Dead => 0,
        HealthState::Quarantined => -1,
    }
}

pub struct LamuMetrics {
    pub registry: Registry,

    pub requests_total: IntCounterVec,
    pub request_duration_seconds: HistogramVec,
    pub tokens_generated_total: IntCounterVec,
    pub vram_used_mb: GaugeVec,
    pub vram_total_mb: Gauge,
    pub queue_depth: IntGaugeVec,
    pub backend_health_state: IntGaugeVec,
    pub backend_restarts_total: IntCounterVec,
    pub backend_quarantined_total: IntCounterVec,
    /// Standalone counter — increments on every /metrics scrape so we can
    /// detect that the endpoint is actually being polled.
    pub scrapes_total: IntCounter,

    // ── refresh-bookkeeping for event counters / gauge cleanup ──────────
    // refresh() is the only place with the health + scheduler snapshot, so the
    // event-style counters are derived there by diffing against the previous
    // snapshot (M16), and stale per-model gauge series are removed (m24).
    prev_restarts: parking_lot::Mutex<std::collections::HashMap<String, u32>>,
    prev_quarantined: parking_lot::Mutex<std::collections::HashSet<String>>,
    prev_gauge_models: parking_lot::Mutex<std::collections::HashSet<String>>,
}

impl LamuMetrics {
    pub fn new() -> prometheus::Result<Self> {
        let registry = Registry::new();

        let requests_total = IntCounterVec::new(
            Opts::new(
                "lamu_requests_total",
                "Number of /v1/chat/completions requests served, by model + status.",
            ),
            &["model", "status", "user"],
        )?;
        let request_duration_seconds = HistogramVec::new(
            HistogramOpts::new(
                "lamu_request_duration_seconds",
                "End-to-end request latency, by model + phase.",
            )
            .buckets(vec![
                0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0,
            ]),
            &["model", "phase"],
        )?;
        let tokens_generated_total = IntCounterVec::new(
            Opts::new(
                "lamu_tokens_generated_total",
                "Tokens generated, by model + kind (content | reasoning).",
            ),
            &["model", "kind", "user"],
        )?;
        let vram_used_mb = GaugeVec::new(
            Opts::new("lamu_vram_used_mb", "VRAM in use per loaded model (MB)."),
            &["model"],
        )?;
        let vram_total_mb = Gauge::new(
            "lamu_vram_total_mb",
            "Total VRAM (MB) reported by nvidia-smi.",
        )?;
        let queue_depth = IntGaugeVec::new(
            Opts::new("lamu_queue_depth", "Pending requests per model queue."),
            &["model"],
        )?;
        let backend_health_state = IntGaugeVec::new(
            Opts::new(
                "lamu_backend_health_state",
                "Backend health: 2=healthy, 1=degraded, 0=dead, -1=quarantined.",
            ),
            &["model"],
        )?;
        let backend_restarts_total = IntCounterVec::new(
            Opts::new(
                "lamu_backend_restarts_total",
                "Successful supervisor restarts, by model.",
            ),
            &["model"],
        )?;
        let backend_quarantined_total = IntCounterVec::new(
            Opts::new(
                "lamu_backend_quarantined_total",
                "Times a backend has been quarantined, by model.",
            ),
            &["model"],
        )?;
        let scrapes_total = IntCounter::new(
            "lamu_metrics_scrapes_total",
            "Total /metrics scrapes — sanity check for collection setup.",
        )?;

        registry.register(Box::new(requests_total.clone()))?;
        registry.register(Box::new(request_duration_seconds.clone()))?;
        registry.register(Box::new(tokens_generated_total.clone()))?;
        registry.register(Box::new(vram_used_mb.clone()))?;
        registry.register(Box::new(vram_total_mb.clone()))?;
        registry.register(Box::new(queue_depth.clone()))?;
        registry.register(Box::new(backend_health_state.clone()))?;
        registry.register(Box::new(backend_restarts_total.clone()))?;
        registry.register(Box::new(backend_quarantined_total.clone()))?;
        registry.register(Box::new(scrapes_total.clone()))?;

        Ok(Self {
            registry,
            requests_total,
            request_duration_seconds,
            tokens_generated_total,
            vram_used_mb,
            vram_total_mb,
            queue_depth,
            backend_health_state,
            backend_restarts_total,
            backend_quarantined_total,
            scrapes_total,
            prev_restarts: parking_lot::Mutex::new(std::collections::HashMap::new()),
            prev_quarantined: parking_lot::Mutex::new(std::collections::HashSet::new()),
            prev_gauge_models: parking_lot::Mutex::new(std::collections::HashSet::new()),
        })
    }

    /// Pull instantaneous values into the gauges. Call from /metrics.
    pub fn refresh(
        &self,
        scheduler: &VramScheduler,
        health: &HealthRegistry,
        queue_depths: Option<&HashMap<String, i64>>,
    ) {
        let budget = scheduler.budget();
        self.vram_total_mb.set(budget.total_mb as f64);
        for (name, vram) in &budget.loaded_models {
            self.vram_used_mb
                .with_label_values(&[name.as_str()])
                .set(*vram as f64);
        }

        for (name, h) in health.all() {
            self.backend_health_state
                .with_label_values(&[name.as_str()])
                .set(health_to_numeric(h.state));
        }

        if let Some(depths) = queue_depths {
            for (name, depth) in depths {
                self.queue_depth
                    .with_label_values(&[name.as_str()])
                    .set(*depth);
            }
        }

        // M16: derive the two event counters by diffing against the previous
        // snapshot (refresh is the only place with the health view). Restarts:
        // inc by the positive delta of the supervisor's restart_attempts (it
        // resets to 0 on recovery, so a negative delta contributes 0). Quarantines:
        // inc once per NEW entry into the Quarantined state.
        {
            let mut prev = self.prev_restarts.lock();
            for (name, h) in health.all() {
                let was = prev.get(name).copied().unwrap_or(0);
                if h.restart_attempts > was {
                    self.backend_restarts_total
                        .with_label_values(&[name.as_str()])
                        .inc_by((h.restart_attempts - was) as u64);
                }
                prev.insert(name.clone(), h.restart_attempts);
            }
        }
        {
            let cur: std::collections::HashSet<String> = health
                .all()
                .iter()
                .filter(|(_, h)| matches!(h.state, lamu_core::health::HealthState::Quarantined))
                .map(|(n, _)| n.clone())
                .collect();
            let mut prev = self.prev_quarantined.lock();
            for name in cur.difference(&prev) {
                self.backend_quarantined_total.with_label_values(&[name.as_str()]).inc();
            }
            *prev = cur;
        }

        // m24: drop per-model gauge series for models no longer loaded, so an
        // unloaded model's vram_used_mb / queue_depth don't render their stale
        // last value forever.
        {
            let current: std::collections::HashSet<String> =
                budget.loaded_models.iter().map(|(n, _)| n.clone()).collect();
            let mut prev = self.prev_gauge_models.lock();
            for stale in prev.difference(&current) {
                let _ = self.vram_used_mb.remove_label_values(&[stale.as_str()]);
                let _ = self.queue_depth.remove_label_values(&[stale.as_str()]);
            }
            *prev = current;
        }
    }

    /// Serialise to Prometheus text. Returns `(body, content_type)`.
    pub fn render(&self) -> (Vec<u8>, &'static str) {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buf = Vec::with_capacity(4096);
        encoder
            .encode(&metric_families, &mut buf)
            .expect("Prometheus text encoder cannot fail on a well-formed registry");
        (buf, "text/plain; version=0.0.4; charset=utf-8")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lamu_core::health::BackendHealth;

    #[test]
    fn render_contains_lamu_series() {
        // Prometheus omits counters/histograms with zero observations from
        // the text output. Touch each label-bearing series so they show up.
        let m = LamuMetrics::new().unwrap();
        let scheduler = VramScheduler::new();
        let mut health = HealthRegistry::new();

        m.requests_total.with_label_values(&["x", "ok", "anon"]).inc();
        m.request_duration_seconds.with_label_values(&["x", "total"]).observe(0.1);
        m.tokens_generated_total.with_label_values(&["x", "content", "anon"]).inc();
        m.vram_used_mb.with_label_values(&["x"]).set(100.0);
        m.queue_depth.with_label_values(&["x"]).set(0);
        m.backend_restarts_total.with_label_values(&["x"]).inc();
        m.backend_quarantined_total.with_label_values(&["x"]).inc();
        let _ = health.get_or_create("x");

        m.refresh(&scheduler, &health, None);
        let (body, ct) = m.render();
        let text = String::from_utf8(body).unwrap();
        assert!(ct.starts_with("text/plain"));
        for series in [
            "lamu_requests_total",
            "lamu_request_duration_seconds",
            "lamu_tokens_generated_total",
            "lamu_vram_used_mb",
            "lamu_vram_total_mb",
            "lamu_queue_depth",
            "lamu_backend_health_state",
            "lamu_backend_restarts_total",
            "lamu_backend_quarantined_total",
        ] {
            assert!(text.contains(series), "missing series: {series}");
        }
    }

    #[test]
    fn request_counter_increments() {
        let m = LamuMetrics::new().unwrap();
        m.requests_total.with_label_values(&["m1", "ok", "anon"]).inc();
        m.requests_total.with_label_values(&["m1", "ok", "anon"]).inc_by(2);
        let text = String::from_utf8(m.render().0).unwrap();
        // The prometheus crate renders labels in lexicographic order, which for
        // {model, status, user} happens to match declaration order.
        assert!(text.contains(r#"lamu_requests_total{model="m1",status="ok",user="anon"} 3"#));
    }

    #[test]
    fn health_state_gauge_uses_numeric_encoding() {
        let m = LamuMetrics::new().unwrap();
        let scheduler = VramScheduler::new();
        let mut health = HealthRegistry::new();
        let _ = health.get_or_create("m1");
        let bad = health.get_or_create("m2");
        bad.force_quarantine("test");
        m.refresh(&scheduler, &health, None);
        let text = String::from_utf8(m.render().0).unwrap();
        assert!(text.contains(r#"lamu_backend_health_state{model="m1"} 2"#));
        assert!(text.contains(r#"lamu_backend_health_state{model="m2"} -1"#));
    }

    #[test]
    fn queue_depth_gauge_per_model() {
        let m = LamuMetrics::new().unwrap();
        let scheduler = VramScheduler::new();
        let health = HealthRegistry::new();
        let mut depths = HashMap::new();
        depths.insert("m1".to_string(), 5i64);
        depths.insert("m2".to_string(), 0i64);
        m.refresh(&scheduler, &health, Some(&depths));
        let text = String::from_utf8(m.render().0).unwrap();
        assert!(text.contains(r#"lamu_queue_depth{model="m1"} 5"#));
        assert!(text.contains(r#"lamu_queue_depth{model="m2"} 0"#));
    }

    #[test]
    fn quarantine_force_updates_health_struct() {
        let mut h = BackendHealth::new("m");
        h.force_quarantine("x");
        assert_eq!(health_to_numeric(h.state), -1);
    }

    #[test]
    fn refresh_increments_restart_and_quarantine_counters() {
        // M16: the two _total counters are derived in refresh() by diffing the
        // health snapshot. A quarantined backend with restart_attempts must move
        // them off zero (they used to render 0 forever).
        let m = LamuMetrics::new().unwrap();
        let scheduler = VramScheduler::new();
        let mut health = HealthRegistry::new();
        {
            let h = health.get_or_create("b");
            h.restart_attempts = 2;
            h.force_quarantine("crash loop");
        }
        m.refresh(&scheduler, &health, None);
        assert_eq!(m.backend_restarts_total.with_label_values(&["b"]).get(), 2);
        assert_eq!(m.backend_quarantined_total.with_label_values(&["b"]).get(), 1);
        // Idempotent: a second refresh with no new events must not double-count.
        m.refresh(&scheduler, &health, None);
        assert_eq!(m.backend_restarts_total.with_label_values(&["b"]).get(), 2);
        assert_eq!(m.backend_quarantined_total.with_label_values(&["b"]).get(), 1);
    }

    #[test]
    fn refresh_removes_stale_gauge_series_on_unload() {
        // m24: a model's vram_used_mb series must not linger at its last value
        // after the model unloads.
        let m = LamuMetrics::new().unwrap();
        let mut scheduler = VramScheduler::new();
        scheduler.set_total_mb_for_tests(24000);
        let entry = lamu_core::types::ModelEntry {
            name: "ghost".into(),
            path: std::path::PathBuf::from("/tmp/ghost.gguf"),
            format: lamu_core::types::ModelFormat::Gguf,
            backend: lamu_core::types::BackendType::LlamaCpp,
            arch: "qwen35".into(), params_b: 1.0, quant: "Q4".into(),
            vram_mb: 8000, context_max: 4096,
            capabilities: vec![lamu_core::types::Capability::Chat],
            reasoning_marker: None, speculative: None, sampling: None,
            pinned: false, main: false, notes: String::new(),
            status: lamu_core::types::ModelStatus::default(),
            modality: lamu_core::types::Modality::Llm,
        };
        scheduler.register_loaded(entry, Some(1), 8020, 8000);
        m.refresh(&scheduler, &health_reg(), None);
        assert!(String::from_utf8(m.render().0).unwrap().contains(r#"lamu_vram_used_mb{model="ghost"}"#));
        // Unload → next refresh drops the stale series.
        scheduler.mark_unloaded("ghost");
        m.refresh(&scheduler, &health_reg(), None);
        assert!(!String::from_utf8(m.render().0).unwrap().contains(r#"lamu_vram_used_mb{model="ghost"}"#),
            "unloaded model's gauge series must be removed (m24)");
    }

    fn health_reg() -> HealthRegistry {
        HealthRegistry::new()
    }
}
