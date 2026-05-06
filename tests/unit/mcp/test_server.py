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


# ── v3 supervisor wiring ────────────────────────────────────────────────────


def test_load_model_registers_supervisor(server, sample_registry, monkeypatch):
    """Successful load_model installs a Supervisor keyed by model name."""
    qwen = sample_registry[0]
    monkeypatch.setattr(server, "_spawn_backend", lambda entry, port: 12345)
    out = _text(server._handle_load_model({"name": "qwen35-27b"}))
    assert "Loaded" in out
    assert qwen.name in server._supervisors


def test_unload_model_drops_supervisor(server, sample_registry, monkeypatch):
    """Unload tears the Supervisor down — its lifetime ends with the backend."""
    qwen = sample_registry[0]
    monkeypatch.setattr(server, "_spawn_backend", lambda entry, port: 12345)
    server._handle_load_model({"name": "qwen35-27b"})
    assert qwen.name in server._supervisors

    with patch("os.kill"):
        server._handle_unload_model({"name": "qwen35-27b"})
    assert qwen.name not in server._supervisors


@pytest.mark.asyncio
async def test_query_failure_routes_through_supervisor(
    server, sample_registry, monkeypatch
):
    """Once a Supervisor is registered, query failures advance health via it
    (not via a raw record_error). With max_attempts=0 the supervisor takes the
    DEAD path without trying to restart, so the test stays fast and silent."""
    from lamu.core.errors import BackendError
    from lamu.core.health import HealthState
    from lamu.core.supervisor import Supervisor, RestartPolicy

    qwen = sample_registry[0]
    monkeypatch.setattr(server, "_spawn_backend", lambda entry, port: 4321)
    server._handle_load_model({"name": "qwen35-27b"})
    h = server._health.get(qwen.name)

    # Replace the registered supervisor with one that won't actually restart.
    restart_calls: list[int] = []
    server._supervisors[qwen.name] = Supervisor(
        health=h,
        restart_fn=lambda: restart_calls.append(1),
        policy=RestartPolicy(max_attempts=0, backoff_seconds=()),
    )

    # Drive the failure threshold (DEGRADED → DEAD).
    with patch("urllib.request.urlopen", side_effect=ConnectionError("nope")):
        for _ in range(3):
            with pytest.raises(BackendError):
                await server._handle_query({"prompt": "hi", "model": qwen.name})

    assert h.state in (HealthState.DEAD, HealthState.QUARANTINED)


def test_restart_backend_re_spawns_and_updates_scheduler(
    server, sample_registry, monkeypatch
):
    """Supervisor.restart_fn → _restart_backend → _spawn_backend + confirm_loaded."""
    qwen = sample_registry[0]
    monkeypatch.setattr(server, "_spawn_backend", lambda entry, port: 12345)
    server._handle_load_model({"name": "qwen35-27b"})
    assert server._scheduler.get_loaded(qwen.name).pid == 12345

    # Pretend the backend died; restart should produce a new pid.
    monkeypatch.setattr(server, "_spawn_backend", lambda entry, port: 67890)
    server._restart_backend(qwen.name)
    assert server._scheduler.get_loaded(qwen.name).pid == 67890


def test_restart_backend_unknown_model_raises(server):
    """Restart on a model that's not in the scheduler surfaces typed error."""
    from lamu.core.errors import BackendUnavailable
    with pytest.raises(BackendUnavailable):
        server._restart_backend("ghost-model")


@pytest.mark.asyncio
async def test_query_refuses_quarantined_model(server, sample_registry):
    """Phase A2: router gets health_map. A QUARANTINED backend never routes."""
    from lamu.core.health import BackendHealth, HealthState
    qwen = sample_registry[0]
    server._scheduler.register_loaded(qwen, pid=1, port=8020, vram_actual_mb=18000)
    server._health._by_id[qwen.name] = BackendHealth(
        backend_id=qwen.name, state=HealthState.QUARANTINED,
    )
    out = _text(await server._handle_query({"prompt": "hi", "model": qwen.name}))
    assert "No model available" in out or "unhealthy" in out


# ── v3 phase C: typed errors surfaced ───────────────────────────────────────


def test_load_model_refuses_when_gpu_unavailable(server, mock_nvidia_smi):
    """Phase C3: GPU unavailability surfaces; load_model never falls back to CPU."""
    mock_nvidia_smi["should_fail"] = True
    server._scheduler.query_vram()  # poisons scheduler with the failure
    assert not server._scheduler.gpu_available
    out = _text(server._handle_load_model({"name": "qwen35-27b"}))
    assert "GPU unavailable" in out


def test_vram_status_reports_gpu_unavailable(server, mock_nvidia_smi):
    """Phase C3: vram_status reports the typed reason, not zeros silently."""
    mock_nvidia_smi["should_fail"] = True
    server._scheduler.query_vram()
    out = _text(server._handle_vram_status())
    assert "GPU unavailable" in out


# ── v3 phase D3: trace IDs ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_query_emits_trace_id_events(server, sample_registry, capsys):
    """Phase D3: every query emits start + done events with the same trace_id."""
    import json
    qwen = sample_registry[0]
    server._scheduler.register_loaded(qwen, pid=1, port=8020, vram_actual_mb=18000)

    payload = {
        "choices": [{
            "message": {"content": "ok", "reasoning_content": ""},
            "finish_reason": "stop",
        }],
    }
    resp = MagicMock()
    resp.read.return_value = json.dumps(payload).encode()
    resp.__enter__ = lambda self: resp
    resp.__exit__ = lambda *_: None

    with patch("urllib.request.urlopen", return_value=resp):
        await server._handle_query({"prompt": "hi", "model": qwen.name})

    err = capsys.readouterr().err
    start_lines = [l for l in err.splitlines() if '"event": "mcp_query_start"' in l]
    done_lines = [l for l in err.splitlines() if '"event": "mcp_query_done"' in l]
    assert start_lines and done_lines, f"missing events in: {err}"
    start = json.loads(start_lines[0])
    done = json.loads(done_lines[0])
    assert start["trace_id"] == done["trace_id"]
    assert len(start["trace_id"]) == 16


@pytest.mark.asyncio
async def test_query_accepts_traceparent_meta(server, sample_registry, capsys):
    """Phase D3: W3C traceparent in `_meta` propagates as the trace_id."""
    import json
    qwen = sample_registry[0]
    server._scheduler.register_loaded(qwen, pid=1, port=8020, vram_actual_mb=18000)

    payload = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
    resp = MagicMock()
    resp.read.return_value = json.dumps(payload).encode()
    resp.__enter__ = lambda self: resp
    resp.__exit__ = lambda *_: None

    # Format: 00-<32 hex traceid>-<16 hex spanid>-<2 hex flags>
    traceparent = "00-0123456789abcdef0123456789abcdef-0011223344556677-01"
    with patch("urllib.request.urlopen", return_value=resp):
        await server._handle_query({
            "prompt": "hi", "model": qwen.name,
            "_meta": {"traceparent": traceparent},
        })

    err = capsys.readouterr().err
    start = next(json.loads(l) for l in err.splitlines() if '"event": "mcp_query_start"' in l)
    # First 16 hex of the traceid become our internal id.
    assert start["trace_id"] == "0123456789abcdef"
