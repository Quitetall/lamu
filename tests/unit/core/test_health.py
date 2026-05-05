"""Tests for lamu.core.health — backend state machine."""
from __future__ import annotations

import pytest

from lamu.core.health import BackendHealth, HealthRegistry, HealthState


def test_starts_healthy():
    h = BackendHealth(backend_id="x")
    assert h.state is HealthState.HEALTHY
    assert h.usable
    assert h.consecutive_errors == 0


def test_first_error_promotes_to_degraded():
    h = BackendHealth(backend_id="x")
    h.record_error(RuntimeError("boom"))
    assert h.state is HealthState.DEGRADED
    assert h.usable
    assert h.consecutive_errors == 1
    assert h.last_error and "RuntimeError" in h.last_error


def test_dead_threshold_at_three():
    h = BackendHealth(backend_id="x")
    for _ in range(3):
        h.record_error(RuntimeError("e"))
    assert h.state is HealthState.DEAD
    assert h.usable is False


def test_quarantine_threshold_at_five():
    h = BackendHealth(backend_id="x")
    for _ in range(5):
        h.record_error(RuntimeError("e"))
    assert h.state is HealthState.QUARANTINED
    assert h.usable is False


def test_record_success_clears_state():
    h = BackendHealth(backend_id="x")
    h.record_error(RuntimeError("e"))
    h.record_success()
    assert h.state is HealthState.HEALTHY
    assert h.consecutive_errors == 0
    assert h.last_error is None


def test_quarantine_is_sticky_to_record_success():
    h = BackendHealth(backend_id="x")
    h.force_quarantine("manual")
    h.record_success()
    assert h.state is HealthState.QUARANTINED


def test_quarantine_is_sticky_to_record_error():
    h = BackendHealth(backend_id="x")
    h.force_quarantine("manual")
    pre_errors = h.consecutive_errors
    h.record_error(RuntimeError("more"))
    assert h.state is HealthState.QUARANTINED
    assert h.consecutive_errors == pre_errors


def test_to_dict_serializable():
    h = BackendHealth(backend_id="x")
    h.record_error(RuntimeError("oops"))
    d = h.to_dict()
    assert d["backend_id"] == "x"
    assert d["state"] == "degraded"
    assert d["consecutive_errors"] == 1


def test_registry_get_or_create_idempotent():
    reg = HealthRegistry()
    a = reg.get_or_create("m1")
    b = reg.get_or_create("m1")
    assert a is b


def test_registry_usable_ids():
    reg = HealthRegistry()
    reg.get_or_create("a")
    bad = reg.get_or_create("b")
    bad.force_quarantine("test")
    assert reg.usable_ids() == {"a"}


def test_registry_snapshot_serializable():
    reg = HealthRegistry()
    reg.get_or_create("a")
    snap = reg.snapshot()
    assert "a" in snap
    assert snap["a"]["state"] == "healthy"
