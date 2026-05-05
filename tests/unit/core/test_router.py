"""Tests for lamu.core.router — capability-based routing."""
from __future__ import annotations

import pytest

from lamu.core.router import Router
from lamu.core.scheduler import VramScheduler
from lamu.core.types import Capability


@pytest.fixture
def scheduler(mock_nvidia_smi):
    return VramScheduler(reserved_mb=1500)


@pytest.fixture
def router(scheduler, sample_registry):
    return Router(scheduler, sample_registry)


def test_explicit_model_not_in_registry(router):
    d = router.route(model="nonexistent")
    assert d.loaded is False
    assert "not found" in d.reason


def test_explicit_model_partial_match(router):
    d = router.route(model="qwen35-27")
    assert d.model_name == "qwen35-27b"


def test_explicit_model_loaded(router, scheduler, sample_registry):
    qwen = sample_registry[0]
    scheduler.register_loaded(qwen, pid=1, port=8020, vram_actual_mb=18000)
    d = router.route(model="qwen35-27b")
    assert d.loaded is True
    assert d.model_name == "qwen35-27b"


def test_explicit_model_will_load(router, sample_registry):
    """Not loaded but fits in VRAM — decision says will-load with no eviction."""
    d = router.route(model="qwen35-0.8b")
    assert d.loaded is False
    assert "will load" in d.reason
    assert d.would_evict == ()


def test_capability_routing_chat(router):
    d = router.route(capabilities=[Capability.CHAT])
    # No model loaded yet — will pick the smallest matching
    assert d.model_name in {"qwen35-27b", "qwen35-0.8b", "gpt2-xl"}


def test_capability_routing_reasoning_excludes_gpt2(router):
    d = router.route(capabilities=[Capability.REASONING])
    assert d.model_name != "gpt2-xl"


def test_capability_routing_loaded_preferred(router, scheduler, sample_registry):
    qwen = sample_registry[0]
    scheduler.register_loaded(qwen, pid=1, port=8020, vram_actual_mb=18000)
    d = router.route(capabilities=[Capability.CHAT])
    assert d.loaded is True
    assert d.model_name == "qwen35-27b"


def test_capability_routing_no_match(router):
    d = router.route(capabilities=[Capability.VISION])
    assert d.model_name == ""
    assert "no model in registry" in d.reason


def test_default_capability_is_chat(router):
    d = router.route()
    assert d.model_name != ""


def test_update_registry_replaces(router, sample_registry, make_model_entry):
    new_only = [make_model_entry("zed", capabilities=(Capability.CHAT,))]
    router.update_registry(new_only)
    d = router.route(model="zed")
    assert d.model_name == "zed"
    # Old entries gone:
    d2 = router.route(model="qwen35-27b")
    assert d2.loaded is False
    assert "not found" in d2.reason


def test_ranked_loaded_prefers_largest_params(router, scheduler, sample_registry):
    """When multiple loaded models match, largest params_b wins."""
    qwen_big, qwen_small = sample_registry[0], sample_registry[1]
    scheduler.register_loaded(qwen_small, pid=1, port=8001, vram_actual_mb=900)
    scheduler.register_loaded(qwen_big, pid=2, port=8020, vram_actual_mb=18000)
    d = router.route(capabilities=[Capability.CHAT])
    assert d.model_name == "qwen35-27b"


def test_router_skips_unhealthy(router, scheduler, sample_registry):
    from lamu.core.health import BackendHealth, HealthState
    qwen = sample_registry[0]
    scheduler.register_loaded(qwen, pid=1, port=8020, vram_actual_mb=18000)
    health_map = {qwen.name: BackendHealth(backend_id=qwen.name, state=HealthState.DEAD)}
    d = router.route(
        capabilities=[Capability.CHAT], health_map=health_map,
    )
    assert d.model_name != qwen.name


def test_router_quarantined_backend_skipped(router, sample_registry):
    from lamu.core.health import BackendHealth, HealthState
    qwen = sample_registry[0]
    health_map = {
        qwen.name: BackendHealth(
            backend_id=qwen.name, state=HealthState.QUARANTINED,
        ),
    }
    d = router.route(model="qwen35-27b", health_map=health_map)
    assert d.loaded is False
    assert "unhealthy" in d.reason


def test_router_degraded_backend_still_routes(router, scheduler, sample_registry):
    """DEGRADED is still usable — gives the backend a chance to recover."""
    from lamu.core.health import BackendHealth, HealthState
    qwen = sample_registry[0]
    scheduler.register_loaded(qwen, pid=1, port=8020, vram_actual_mb=18000)
    h = BackendHealth(backend_id=qwen.name)
    h.state = HealthState.DEGRADED
    h.consecutive_errors = 1
    d = router.route(capabilities=[Capability.CHAT], health_map={qwen.name: h})
    assert d.loaded is True
    assert d.model_name == "qwen35-27b"
