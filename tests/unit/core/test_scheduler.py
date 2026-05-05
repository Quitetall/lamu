"""Tests for lamu.core.scheduler — VRAM accounting + LRU eviction."""
from __future__ import annotations

import time

import pytest

from lamu.core import scheduler as sched_mod
from lamu.core.scheduler import VramScheduler, _query_gpu_pids, _query_vram
from lamu.core.types import ModelState


@pytest.fixture
def sched(mock_nvidia_smi):
    """Scheduler initialized against a 24 GB / 1.5 GB used GPU stub."""
    return VramScheduler(reserved_mb=1500)


def test_init_queries_total(sched, mock_nvidia_smi):
    assert sched.total_mb == mock_nvidia_smi["vram_total_mb"]


def test_query_vram_failure_returns_zero(mock_nvidia_smi):
    mock_nvidia_smi["should_fail"] = True
    assert _query_vram() == (0, 0)


def test_query_vram_timeout_returns_zero(mock_nvidia_smi):
    mock_nvidia_smi["should_timeout"] = True
    assert _query_vram() == (0, 0)


def test_require_gpu_raises_after_smi_failure(mock_nvidia_smi):
    """Phase C: nvidia-smi failure flips the GPU-unavailable flag; subsequent
    `require_gpu()` calls raise GpuUnavailableError instead of letting the
    program silently keep operating with `total_mb=0`."""
    from lamu.core.errors import GpuUnavailableError
    from lamu.core.scheduler import require_gpu
    mock_nvidia_smi["should_fail"] = True
    _query_vram()  # marks GPU unavailable
    with pytest.raises(GpuUnavailableError):
        require_gpu()


def test_require_gpu_silent_when_healthy(mock_nvidia_smi):
    """A successful nvidia-smi query clears the unavailable flag."""
    from lamu.core.scheduler import require_gpu
    mock_nvidia_smi["should_fail"] = False
    _query_vram()  # success path
    require_gpu()  # should NOT raise


def test_query_gpu_pids_parses(mock_nvidia_smi):
    mock_nvidia_smi["pids"] = [(123, 4000), (456, 8000)]
    assert _query_gpu_pids() == [(123, 4000), (456, 8000)]


def test_query_gpu_pids_failure_returns_empty(mock_nvidia_smi):
    mock_nvidia_smi["should_fail"] = True
    assert _query_gpu_pids() == []


def test_register_loaded_records_state(sched, make_model_entry):
    e = make_model_entry()
    lm = sched.register_loaded(e, pid=1234, port=8020, vram_actual_mb=18000)
    assert lm.state is ModelState.LOADED
    assert sched.is_loaded(e.name)
    assert sched.get_loaded(e.name) is lm


def test_available_mb_subtracts_reserved_and_loaded(sched, make_model_entry):
    e = make_model_entry()
    sched.register_loaded(e, pid=1, port=8020, vram_actual_mb=18000)
    expected = sched.total_mb - 18000 - 1500
    assert sched.available_mb == expected


def test_can_fit_true_when_room(sched, make_model_entry):
    small = make_model_entry("tiny", vram_mb=500)
    assert sched.can_fit(small)


def test_can_fit_false_when_full(sched, make_model_entry):
    big = make_model_entry("huge", vram_mb=sched.total_mb)
    assert not sched.can_fit(big)


def test_plan_load_already_loaded(sched, make_model_entry):
    e = make_model_entry()
    sched.register_loaded(e, pid=1, port=8020, vram_actual_mb=18000)
    can, evict = sched.plan_load(e)
    assert can is True and evict == []


def test_plan_load_fits_directly(sched, make_model_entry):
    e = make_model_entry("small", vram_mb=2000)
    can, evict = sched.plan_load(e)
    assert can is True and evict == []


def test_plan_load_requires_eviction(sched, make_model_entry):
    big = make_model_entry("big1", vram_mb=18000)
    sched.register_loaded(big, pid=1, port=8020, vram_actual_mb=18000)
    other = make_model_entry("other", vram_mb=18000)
    can, evict = sched.plan_load(other)
    assert can is True
    assert evict == ["big1"]


def test_plan_load_skips_pinned(sched, make_model_entry):
    pinned = make_model_entry("pinned", vram_mb=20000, pinned=True)
    sched.register_loaded(pinned, pid=1, port=8020, vram_actual_mb=20000)
    other = make_model_entry("other", vram_mb=15000)
    can, evict = sched.plan_load(other)
    assert can is False
    assert evict == []


def test_plan_load_impossible(sched, make_model_entry):
    huge = make_model_entry("huge", vram_mb=sched.total_mb * 2)
    can, evict = sched.plan_load(huge)
    assert can is False and evict == []


def test_plan_eviction_lru_order(sched, make_model_entry):
    a = make_model_entry("a", vram_mb=4000)
    b = make_model_entry("b", vram_mb=4000)
    sched.register_loaded(a, pid=1, port=8020, vram_actual_mb=4000)
    time.sleep(0.001)
    sched.register_loaded(b, pid=2, port=8001, vram_actual_mb=4000)
    sched.mark_used("a")  # bump a, so b becomes oldest
    plan = sched.plan_eviction(needed_mb=3000)
    assert plan == ["b"]


def test_plan_eviction_zero_needed():
    s = VramScheduler(reserved_mb=0)
    assert s.plan_eviction(0) == []


def test_mark_unloaded_removes(sched, make_model_entry):
    e = make_model_entry()
    sched.register_loaded(e, pid=1, port=8020, vram_actual_mb=18000)
    sched.mark_unloaded(e.name)
    assert not sched.is_loaded(e.name)


def test_mark_loading_then_confirm(sched, make_model_entry):
    e = make_model_entry()
    sched.mark_loading(e)
    assert sched.get_loaded(e.name).state is ModelState.LOADING
    sched.confirm_loaded(e.name, pid=1234, port=8001, vram_actual_mb=17500)
    lm = sched.get_loaded(e.name)
    assert lm.state is ModelState.LOADED
    assert lm.pid == 1234
    assert lm.vram_actual_mb == 17500


def test_budget_snapshot_shape(sched, make_model_entry):
    e = make_model_entry()
    sched.register_loaded(e, pid=1, port=8020, vram_actual_mb=18000)
    b = sched.budget()
    assert b.total_mb > 0
    assert b.loaded_models == ((e.name, 18000),)


def test_loaded_models_listing(sched, make_model_entry):
    a = make_model_entry("a", vram_mb=2000)
    b = make_model_entry("b", vram_mb=3000)
    sched.register_loaded(a, pid=1, port=1, vram_actual_mb=2000)
    sched.register_loaded(b, pid=2, port=2, vram_actual_mb=3000)
    names = {m.entry.name for m in sched.loaded_models()}
    assert names == {"a", "b"}


def test_is_loaded_only_when_state_is_loaded(sched, make_model_entry):
    e = make_model_entry()
    sched.mark_loading(e)
    assert sched.is_loaded(e.name) is False
    sched.confirm_loaded(e.name, 1, 8020, 18000)
    assert sched.is_loaded(e.name) is True
