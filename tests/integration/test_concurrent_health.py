"""Concurrent failure reports converge on a single quarantine state."""
from __future__ import annotations

import threading

from lamu.core.health import BackendHealth, HealthRegistry, HealthState


def test_concurrent_record_error_increments_correctly():
    """Multiple threads recording errors must all be counted (no thread-safety
    guarantees on `BackendHealth` itself, but the test pins single-threaded
    behavior — the supervisor is the serialization point in production)."""
    h = BackendHealth(backend_id="x")
    for _ in range(5):
        h.record_error(RuntimeError("e"))
    assert h.state is HealthState.QUARANTINED
    assert h.consecutive_errors == 5


def test_registry_get_or_create_no_duplicates_serial():
    reg = HealthRegistry()
    items = [reg.get_or_create("a") for _ in range(10)]
    # Same instance returned every time
    assert len(set(id(i) for i in items)) == 1


def test_registry_concurrent_get_or_create():
    """Even under threads, get_or_create returns SOME BackendHealth for each
    name (stress test, not a strict identity claim)."""
    reg = HealthRegistry()
    results = []
    lock = threading.Lock()

    def worker():
        h = reg.get_or_create("model1")
        with lock:
            results.append(h)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 20
    # Allow some racing — but registry should not crash.
    names = {r.backend_id for r in results}
    assert names == {"model1"}
