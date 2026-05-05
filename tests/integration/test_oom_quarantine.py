"""5+ consecutive backend errors → backend quarantined → router refuses it."""
from __future__ import annotations

import pytest

from lamu.core.health import BackendHealth, HealthState
from lamu.core.router import Router
from lamu.core.scheduler import VramScheduler


def test_quarantine_after_five_failures(mock_nvidia_smi, sample_registry):
    sched = VramScheduler(reserved_mb=1500)
    qwen = sample_registry[0]
    sched.register_loaded(qwen, pid=1, port=8020, vram_actual_mb=18000)

    health = BackendHealth(backend_id=qwen.name)
    for _ in range(5):
        health.record_error(MemoryError("CUDA OOM"))
    assert health.state is HealthState.QUARANTINED

    router = Router(sched, sample_registry)
    decision = router.route(model="qwen35-27b", health_map={qwen.name: health})
    assert decision.loaded is False
    assert "unhealthy" in decision.reason


def test_quarantine_blocks_capability_route(mock_nvidia_smi, sample_registry):
    """Even capability-based routing must skip a quarantined model."""
    from lamu.core.types import Capability
    sched = VramScheduler(reserved_mb=1500)
    qwen = sample_registry[0]
    sched.register_loaded(qwen, pid=1, port=8020, vram_actual_mb=18000)

    health = BackendHealth(backend_id=qwen.name)
    for _ in range(5):
        health.record_error(MemoryError("OOM"))

    router = Router(sched, sample_registry)
    d = router.route(
        capabilities=[Capability.REASONING],
        health_map={qwen.name: health},
    )
    assert d.model_name != qwen.name
