"""Prometheus metrics for the OpenAI-compat layer.

All metrics live on a process-local CollectorRegistry — no global state
sharing — so two daemons in one Python process (tests, multi-tenant
later) don't trample each other.

Metric naming follows Prometheus conventions:
  - counters end in `_total`
  - histograms expose `_seconds` for durations
  - gauges describe the instantaneous value
  - labels are low-cardinality (model name, phase, kind)
"""
from __future__ import annotations

from typing import Optional

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from lamu.core.health import HealthRegistry, HealthState
from lamu.core.scheduler import VramScheduler


# Health state → numeric so a Prometheus gauge can plot it.
_HEALTH_NUMERIC = {
    HealthState.HEALTHY: 2,
    HealthState.DEGRADED: 1,
    HealthState.DEAD: 0,
    HealthState.QUARANTINED: -1,
}


class LamuMetrics:
    """Bundle of collectors keyed to a single registry. The OpenAI compat
    app holds one of these and refreshes the gauges on `/metrics` scrape.
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None) -> None:
        self.registry = registry or CollectorRegistry()

        self.requests_total = Counter(
            "lamu_requests_total",
            "Number of /v1/chat/completions requests served, by model + status.",
            ["model", "status"],
            registry=self.registry,
        )
        self.request_duration_seconds = Histogram(
            "lamu_request_duration_seconds",
            "End-to-end request latency, by model + phase.",
            ["model", "phase"],
            buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
            registry=self.registry,
        )
        self.tokens_generated_total = Counter(
            "lamu_tokens_generated_total",
            "Tokens generated, by model + kind (content | reasoning).",
            ["model", "kind"],
            registry=self.registry,
        )

        # Gauges scraped each /metrics call — refresh from scheduler+health.
        self.vram_used_mb = Gauge(
            "lamu_vram_used_mb",
            "VRAM in use per loaded model (MB).",
            ["model"],
            registry=self.registry,
        )
        self.vram_total_mb = Gauge(
            "lamu_vram_total_mb",
            "Total VRAM (MB) reported by nvidia-smi.",
            registry=self.registry,
        )
        self.queue_depth = Gauge(
            "lamu_queue_depth",
            "Pending requests per model queue.",
            ["model"],
            registry=self.registry,
        )
        self.backend_health_state = Gauge(
            "lamu_backend_health_state",
            "Backend health: 2=healthy, 1=degraded, 0=dead, -1=quarantined.",
            ["model"],
            registry=self.registry,
        )
        self.backend_restarts_total = Counter(
            "lamu_backend_restarts_total",
            "Successful supervisor restarts, by model.",
            ["model"],
            registry=self.registry,
        )
        self.backend_quarantined_total = Counter(
            "lamu_backend_quarantined_total",
            "Times a backend has been quarantined, by model.",
            ["model"],
            registry=self.registry,
        )

    def refresh(
        self,
        scheduler: VramScheduler,
        health: HealthRegistry,
        queue_depths: Optional[dict[str, int]] = None,
    ) -> None:
        """Pull instantaneous values into the gauges. Call from /metrics."""
        budget = scheduler.budget()
        self.vram_total_mb.set(budget.total_mb)
        for name, vram in budget.loaded_models:
            self.vram_used_mb.labels(model=name).set(vram)

        for name, h in health.all().items():
            self.backend_health_state.labels(model=name).set(_HEALTH_NUMERIC[h.state])

        if queue_depths:
            for name, depth in queue_depths.items():
                self.queue_depth.labels(model=name).set(depth)

    def render(self) -> tuple[bytes, str]:
        """Serialise to Prometheus text. Returns `(body, content_type)`."""
        return generate_latest(self.registry), CONTENT_TYPE_LATEST
