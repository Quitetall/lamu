"""Tests for lamu.api.metrics — Prometheus collectors + refresh."""
from __future__ import annotations

import pytest

from lamu.api.metrics import LamuMetrics
from lamu.core.health import BackendHealth, HealthRegistry, HealthState
from lamu.core.scheduler import VramScheduler


@pytest.fixture
def metrics():
    return LamuMetrics()


@pytest.fixture
def sched(mock_nvidia_smi):
    return VramScheduler(reserved_mb=1500)


def test_render_contains_lamu_metrics(metrics, sched):
    h = HealthRegistry()
    metrics.refresh(sched, h)
    body, ctype = metrics.render()
    text = body.decode()
    assert "text/plain" in ctype
    for series in (
        "lamu_requests_total",
        "lamu_request_duration_seconds",
        "lamu_tokens_generated_total",
        "lamu_vram_used_mb",
        "lamu_vram_total_mb",
        "lamu_queue_depth",
        "lamu_backend_health_state",
        "lamu_backend_restarts_total",
        "lamu_backend_quarantined_total",
    ):
        assert series in text, f"missing series: {series}"


def test_request_counter_increments(metrics):
    metrics.requests_total.labels(model="m1", status="ok").inc()
    metrics.requests_total.labels(model="m1", status="ok").inc(2)
    text = metrics.render()[0].decode()
    # counter total = 3
    assert 'lamu_requests_total{model="m1",status="ok"} 3.0' in text


def test_health_state_gauge_uses_numeric_encoding(metrics, sched):
    h = HealthRegistry()
    healthy = h.get_or_create("m1")
    bad = h.get_or_create("m2")
    bad.force_quarantine("test")
    assert bad.state is HealthState.QUARANTINED

    metrics.refresh(sched, h)
    text = metrics.render()[0].decode()
    assert 'lamu_backend_health_state{model="m1"} 2.0' in text
    assert 'lamu_backend_health_state{model="m2"} -1.0' in text


def test_vram_total_gauge_reflects_scheduler(metrics, sched, mock_nvidia_smi):
    metrics.refresh(sched, HealthRegistry())
    text = metrics.render()[0].decode()
    expected = mock_nvidia_smi["vram_total_mb"]
    assert f"lamu_vram_total_mb {expected}.0" in text


def test_queue_depth_gauge_per_model(metrics, sched):
    metrics.refresh(sched, HealthRegistry(), queue_depths={"m1": 5, "m2": 0})
    text = metrics.render()[0].decode()
    assert 'lamu_queue_depth{model="m1"} 5.0' in text
    assert 'lamu_queue_depth{model="m2"} 0.0' in text
