"""Tests for lamu.core.types — pure dataclass + enum contracts."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from lamu.core.types import (
    BackendType,
    Capability,
    LoadedModel,
    ModelEntry,
    ModelFormat,
    ModelState,
    QueryResult,
    QueryStats,
    ReasoningMarker,
    RouteDecision,
    SpeculativeConfig,
    StreamChunk,
    VramBudget,
)


def test_capability_values_stable():
    assert Capability.CHAT.value == "chat"
    assert Capability.CODE.value == "code"
    assert Capability.REASONING.value == "reasoning"
    assert Capability.ROUTING.value == "routing"
    assert Capability.VISION.value == "vision"
    assert Capability.LONG_CONTEXT.value == "long_context"


def test_model_format_values():
    assert {f.value for f in ModelFormat} == {"gguf", "safetensors", "onnx", "custom"}


def test_backend_type_values():
    assert BackendType.LLAMACPP.value == "llama_cpp"
    assert BackendType.MEGAKERNEL.value == "megakernel"
    assert BackendType.DFLASH.value == "dflash"
    assert BackendType.DFLASH_LUCEBOX.value == "dflash_lucebox"


def test_model_state_distinct():
    assert len({s for s in ModelState}) == 4


def test_reasoning_marker_frozen():
    m = ReasoningMarker(open_tag="<t>", close_tag="</t>", family="x")
    with pytest.raises(FrozenInstanceError):
        m.family = "y"  # type: ignore[misc]


def test_speculative_config_default_draft_max():
    cfg = SpeculativeConfig(draft_path=Path("/tmp/d.gguf"), method="dflash")
    assert cfg.draft_max == 8


def test_model_entry_immutable():
    entry = ModelEntry(
        name="x", path=Path("/tmp/x.gguf"),
        format=ModelFormat.GGUF, backend=BackendType.LLAMACPP,
        arch="qwen35", params_b=27.0, quant="Q5_K_M",
        vram_mb=18000, context_max=131072,
        capabilities=(Capability.CHAT,),
    )
    with pytest.raises(FrozenInstanceError):
        entry.name = "y"  # type: ignore[misc]
    assert entry.pinned is False
    assert entry.reasoning_marker is None
    assert entry.speculative is None


def test_loaded_model_mutable():
    """LoadedModel is intentionally mutable — runtime state changes."""
    entry = ModelEntry(
        name="x", path=Path("/tmp/x"), format=ModelFormat.GGUF,
        backend=BackendType.LLAMACPP, arch="qwen35", params_b=27.0,
        quant="Q5_K_M", vram_mb=18000, context_max=131072,
        capabilities=(Capability.CHAT,),
    )
    lm = LoadedModel(
        entry=entry, state=ModelState.LOADING, pid=None, port=8020,
        vram_actual_mb=17500, last_used_ts=0.0,
    )
    lm.state = ModelState.LOADED
    lm.last_used_ts = 100.0
    assert lm.state is ModelState.LOADED
    assert lm.last_used_ts == 100.0


def test_route_decision_defaults():
    d = RouteDecision(model_name="x", reason="r", loaded=True)
    assert d.would_evict == ()


def test_stream_chunk_types():
    chunk = StreamChunk(type="content", text="hi")
    assert chunk.type == "content"
    with pytest.raises(FrozenInstanceError):
        chunk.text = "y"  # type: ignore[misc]


def test_query_stats_default_retries():
    s = QueryStats(latency_ms=10.0, time_to_first_token_ms=1.0,
                   tokens_generated=5, tokens_per_second=500.0,
                   prompt_tokens=3)
    assert s.retries == 0
    assert s.stream_chunks == 0


def test_query_result_finish_reason_default():
    s = QueryStats(latency_ms=0, time_to_first_token_ms=0,
                   tokens_generated=0, tokens_per_second=0, prompt_tokens=0)
    r = QueryResult(content="hi", reasoning=None, model_used="x", stats=s)
    assert r.finish_reason == "stop"


def test_vram_budget_fields():
    b = VramBudget(
        total_mb=24576, used_mb=4000, free_mb=20576,
        loaded_models=(("a", 4000),), available_mb=19000,
    )
    assert b.loaded_models[0] == ("a", 4000)
    with pytest.raises(FrozenInstanceError):
        b.used_mb = 1  # type: ignore[misc]
