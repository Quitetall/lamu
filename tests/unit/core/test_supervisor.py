"""Tests for lamu.core.supervisor — restart-with-backoff."""
from __future__ import annotations

import pytest

from lamu.core.health import BackendHealth, HealthState
from lamu.core.supervisor import RestartPolicy, Supervisor


def _no_sleep(_s: float) -> None: pass


def test_first_failure_only_degrades_no_restart():
    h = BackendHealth(backend_id="x")
    calls = []
    s = Supervisor(h, restart_fn=lambda: calls.append("r"), sleep_fn=_no_sleep)
    s.report_failure(RuntimeError("boom"))
    assert h.state is HealthState.DEGRADED
    assert calls == []  # no restart on first failure


def test_three_failures_trigger_restart():
    h = BackendHealth(backend_id="x")
    calls = []
    s = Supervisor(
        h,
        restart_fn=lambda: calls.append("r"),
        policy=RestartPolicy(max_attempts=3, backoff_seconds=(0, 0, 0)),
        sleep_fn=_no_sleep,
    )
    for _ in range(3):
        s.report_failure(RuntimeError("e"))
    assert calls  # restart attempted at least once


def test_successful_restart_recovers_health():
    h = BackendHealth(backend_id="x")
    s = Supervisor(
        h,
        restart_fn=lambda: None,  # succeeds silently
        policy=RestartPolicy(max_attempts=3, backoff_seconds=(0, 0, 0)),
        sleep_fn=_no_sleep,
    )
    for _ in range(3):
        s.report_failure(RuntimeError("e"))
    assert h.state is HealthState.HEALTHY


def test_failing_restart_quarantines_eventually():
    h = BackendHealth(backend_id="x")
    s = Supervisor(
        h,
        restart_fn=lambda: (_ for _ in ()).throw(OSError("nope")),
        policy=RestartPolicy(max_attempts=3, backoff_seconds=(0, 0, 0)),
        sleep_fn=_no_sleep,
    )
    # Start dead-state with 3 errors so the first report_failure triggers
    # restart attempts.
    for _ in range(3):
        h.record_error(RuntimeError("seed"))
    s.report_failure(RuntimeError("trigger"))
    assert h.state is HealthState.QUARANTINED


def test_emits_quarantine_event_on_5_errors(capsys):
    h = BackendHealth(backend_id="x")
    s = Supervisor(h, restart_fn=lambda: None, sleep_fn=_no_sleep,
                   policy=RestartPolicy(max_attempts=0, backoff_seconds=()))
    for _ in range(5):
        s.report_failure(RuntimeError("e"))
    err = capsys.readouterr().err
    assert "backend_quarantined" in err
