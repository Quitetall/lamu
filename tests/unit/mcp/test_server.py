"""Tests for lamu.mcp.server — MCP tool dispatch."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from lamu.core.scheduler import VramScheduler
from lamu.core.types import Capability
from lamu.mcp.server import LamuMcpServer


@pytest.fixture
def server(tmp_path, mock_nvidia_smi, sample_registry):
    """LamuMcpServer wired against a tmp registry, no models on disk."""
    from lamu.core.registry import write_registry
    reg_path = tmp_path / "registry.yaml"
    write_registry(sample_registry, reg_path)
    sched = VramScheduler(reserved_mb=1500)
    return LamuMcpServer(
        models_dir=tmp_path,
        registry_path=reg_path,
        scheduler=sched,
    )


def _text(result):
    return result[0].text


def test_init_loads_registry(server, sample_registry):
    assert set(server._entries.keys()) == {e.name for e in sample_registry}


def test_handle_list_models_lists_all(server):
    out = _text(server._handle_list_models())
    assert "qwen35-27b" in out
    assert "gpt2-xl" in out


def test_handle_vram_status_includes_totals(server):
    out = _text(server._handle_vram_status())
    assert "VRAM:" in out
    assert "Available for models" in out


def test_handle_plan_query_serializes_decision(server):
    out = _text(server._handle_plan_query({"model": "qwen35-27b", "prompt": "hi"}))
    parsed = json.loads(out)
    assert parsed["would_route_to"] == "qwen35-27b"
    assert "loaded" in parsed
    assert "would_evict" in parsed


def test_handle_plan_query_capability_invalid_raises(server):
    """Bad capability strings should not be silently dropped."""
    with pytest.raises(ValueError):
        server._handle_plan_query({"prompt": "x", "capabilities": ["totally_fake"]})


def test_handle_load_model_unknown(server):
    out = _text(server._handle_load_model({"name": "nonexistent-model-xyz"}))
    assert "not found" in out


def test_handle_load_model_already_loaded(server, sample_registry):
    qwen = sample_registry[0]
    server._scheduler.register_loaded(qwen, pid=1, port=8020, vram_actual_mb=18000)
    out = _text(server._handle_load_model({"name": "qwen35-27b"}))
    assert "already loaded" in out


def test_handle_unload_not_loaded(server):
    out = _text(server._handle_unload_model({"name": "qwen35-27b"}))
    assert "not loaded" in out


def test_handle_unload_kills_process(server, sample_registry):
    import signal
    qwen = sample_registry[0]
    server._scheduler.register_loaded(qwen, pid=4321, port=8020, vram_actual_mb=18000)
    with patch("os.kill") as kill:
        out = _text(server._handle_unload_model({"name": "qwen35-27b"}))
    kill.assert_called_with(4321, signal.SIGKILL)
    assert "Unloaded" in out


def test_handle_unload_handles_dead_proc(server, sample_registry):
    qwen = sample_registry[0]
    server._scheduler.register_loaded(qwen, pid=4321, port=8020, vram_actual_mb=18000)
    with patch("os.kill", side_effect=ProcessLookupError):
        out = _text(server._handle_unload_model({"name": "qwen35-27b"}))
    assert "Unloaded" in out


def test_handle_scan_writes_registry(tmp_path, mock_nvidia_smi, gguf_bytes_factory):
    """scan_models tool calls scan_directory + write_registry."""
    from lamu.core.registry import write_registry
    (tmp_path / "qwen35-test.gguf").write_bytes(gguf_bytes_factory("qwen35"))
    reg_path = tmp_path / "registry.yaml"
    write_registry([], reg_path)
    sched = VramScheduler(reserved_mb=1500)
    s = LamuMcpServer(models_dir=tmp_path, registry_path=reg_path, scheduler=sched)
    out = _text(s._handle_scan())
    assert "models found" in out


@pytest.mark.asyncio
async def test_handle_query_no_model(server):
    out = _text(await server._handle_query({
        "prompt": "hi", "capabilities": ["vision"],
    }))
    assert "No model available" in out


@pytest.mark.asyncio
async def test_handle_query_not_loaded(server):
    """Routing succeeds but model isn't loaded → returns load instructions."""
    out = _text(await server._handle_query({
        "prompt": "hi", "model": "qwen35-27b",
    }))
    assert "not loaded" in out


@pytest.mark.asyncio
async def test_handle_query_generates(server, sample_registry):
    qwen = sample_registry[0]
    server._scheduler.register_loaded(qwen, pid=1, port=8020, vram_actual_mb=18000)

    payload = {
        "choices": [{
            "message": {"content": "hello", "reasoning_content": ""},
            "finish_reason": "stop",
        }],
    }
    resp = MagicMock()
    resp.read.return_value = json.dumps(payload).encode()
    resp.__enter__ = lambda self: resp
    resp.__exit__ = lambda *_: None

    with patch("urllib.request.urlopen", return_value=resp):
        out = _text(await server._handle_query({
            "prompt": "hi", "model": "qwen35-27b",
        }))
    assert "hello" in out


@pytest.mark.asyncio
async def test_handle_query_uses_typed_error(server, sample_registry):
    """Phase C: query failures raise BackendError (not silent text envelope)."""
    from lamu.core.errors import BackendError
    qwen = sample_registry[0]
    server._scheduler.register_loaded(qwen, pid=1, port=8020, vram_actual_mb=18000)
    with patch("urllib.request.urlopen", side_effect=ConnectionRefusedError("nope")):
        with pytest.raises(BackendError):
            await server._handle_query({"prompt": "hi", "model": "qwen35-27b"})


@pytest.mark.asyncio
async def test_handle_query_records_health_on_failure(server, sample_registry):
    """Backend failure → health.record_error → state degrades."""
    from lamu.core.errors import BackendError
    from lamu.core.health import HealthState
    qwen = sample_registry[0]
    server._scheduler.register_loaded(qwen, pid=1, port=8020, vram_actual_mb=18000)
    with patch("urllib.request.urlopen", side_effect=ConnectionError("nope")):
        with pytest.raises(BackendError):
            await server._handle_query({"prompt": "hi", "model": "qwen35-27b"})
    h = server._health.get(qwen.name)
    assert h is not None
    assert h.state is HealthState.DEGRADED
    assert h.consecutive_errors == 1
