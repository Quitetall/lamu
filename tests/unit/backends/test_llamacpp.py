"""Tests for lamu.backends.llamacpp — llama-server subprocess wrapper."""
from __future__ import annotations

import json
import signal
import subprocess
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lamu.backends.llamacpp import LlamaCppBackend


@pytest.fixture
def fake_bin(tmp_path):
    p = tmp_path / "llama-server"
    p.write_text("#!/bin/sh\n")
    p.chmod(0o755)
    return p


@pytest.fixture
def backend(fake_bin):
    return LlamaCppBackend(bin_path=fake_bin)


def _fake_health_response(status: str = "ok"):
    body = json.dumps({"status": status}).encode()
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = lambda self: resp
    resp.__exit__ = lambda *_: None
    return resp


def test_load_raises_when_binary_missing(tmp_path, make_model_entry):
    from lamu.core.errors import BackendError
    b = LlamaCppBackend(bin_path=tmp_path / "missing")
    with pytest.raises(BackendError, match="not found"):
        b.load(make_model_entry(), port=8020)


def test_is_healthy_true_on_status_ok(backend):
    backend._port = 8020
    with patch("urllib.request.urlopen", return_value=_fake_health_response("ok")):
        assert backend.is_healthy() is True


def test_is_healthy_false_on_url_error(backend):
    backend._port = 8020
    import urllib.error
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("connection refused")):
        assert backend.is_healthy() is False


def test_is_healthy_false_on_bad_json(backend):
    backend._port = 8020
    bad_resp = MagicMock()
    bad_resp.read.return_value = b"not json"
    bad_resp.__enter__ = lambda self: bad_resp
    bad_resp.__exit__ = lambda *_: None
    with patch("urllib.request.urlopen", return_value=bad_resp):
        assert backend.is_healthy() is False


def test_generate_returns_content(backend):
    backend._port = 8020
    body = json.dumps({
        "choices": [{"message": {"content": "hello", "reasoning_content": ""}}]
    }).encode()
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = lambda self: resp
    resp.__exit__ = lambda *_: None
    with patch("urllib.request.urlopen", return_value=resp):
        out = backend.generate([{"role": "user", "content": "hi"}])
    assert out == "hello"


def test_generate_wraps_reasoning(backend):
    backend._port = 8020
    body = json.dumps({
        "choices": [{"message": {"content": "answer", "reasoning_content": "think"}}]
    }).encode()
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = lambda self: resp
    resp.__exit__ = lambda *_: None
    with patch("urllib.request.urlopen", return_value=resp):
        out = backend.generate([{"role": "user", "content": "hi"}])
    assert "<think>" in out and "</think>" in out
    assert "answer" in out


def test_stream_yields_content_only(backend):
    backend._port = 8020
    sse = (
        b'data: {"choices":[{"delta":{"content":"a"}}]}\n'
        b'data: {"choices":[{"delta":{"content":"b"}}]}\n'
        b'data: [DONE]\n'
    )
    resp = MagicMock()
    resp.__enter__ = lambda self: iter(sse.splitlines(keepends=True))
    resp.__exit__ = lambda *_: None
    with patch("urllib.request.urlopen", return_value=resp):
        chunks = list(backend.stream([{"role": "user", "content": "x"}]))
    assert chunks == ["a", "b"]


def test_stream_skips_malformed_chunks(backend):
    backend._port = 8020
    sse = (
        b'data: not-json\n'
        b'data: {"choices":[{"delta":{}}]}\n'
        b'data: {"choices":[{"delta":{"content":"ok"}}]}\n'
        b'data: [DONE]\n'
    )
    resp = MagicMock()
    resp.__enter__ = lambda self: iter(sse.splitlines(keepends=True))
    resp.__exit__ = lambda *_: None
    with patch("urllib.request.urlopen", return_value=resp):
        chunks = list(backend.stream([]))
    assert chunks == ["ok"]


def test_unload_kills_proc(backend):
    proc = MagicMock()
    proc.send_signal = MagicMock()
    proc.wait = MagicMock(return_value=0)
    backend._proc = proc
    backend.unload()
    proc.send_signal.assert_called_with(signal.SIGKILL)
    assert backend._proc is None
    assert backend.model_name == ""


def test_unload_handles_dead_proc(backend):
    proc = MagicMock()
    proc.send_signal.side_effect = ProcessLookupError()
    backend._proc = proc
    backend.unload()
    assert backend._proc is None


def test_get_vram_mb_zero_when_no_proc(backend):
    assert backend.get_vram_mb() == 0


def test_get_vram_mb_parses_smi(backend, fake_completed_process, monkeypatch):
    proc = MagicMock(); proc.pid = 4321
    backend._proc = proc

    def fake_run(cmd, *a, **k):
        return fake_completed_process(stdout="4321, 5678\n9999, 1234\n")
    monkeypatch.setattr("subprocess.run", fake_run)
    assert backend.get_vram_mb() == 5678


def test_load_command_includes_qwen_speculation(backend, make_model_entry):
    """When model arch is qwen35, ngram-mod flags must be added."""
    captured: dict = {}

    def fake_popen(cmd, *a, **k):
        captured["cmd"] = cmd
        proc = MagicMock(); proc.pid = 1
        return proc

    with patch("subprocess.Popen", fake_popen), \
         patch.object(backend, "is_healthy", return_value=True):
        backend.load(make_model_entry(arch="qwen35"), port=8020)
    cmd = captured["cmd"]
    assert "--spec-type" in cmd
    assert "ngram-mod" in cmd


def test_load_command_skips_speculation_for_other_arch(backend, make_model_entry):
    captured: dict = {}

    def fake_popen(cmd, *a, **k):
        captured["cmd"] = cmd
        proc = MagicMock(); proc.pid = 1
        return proc

    with patch("subprocess.Popen", fake_popen), \
         patch.object(backend, "is_healthy", return_value=True):
        backend.load(make_model_entry(arch="gpt2"), port=8020)
    assert "--spec-type" not in captured["cmd"]


def test_load_timeout_unloads_and_raises(backend, make_model_entry):
    from lamu.core.errors import BackendError
    with patch("subprocess.Popen") as popen, \
         patch.object(backend, "is_healthy", return_value=False), \
         patch("time.sleep"):
        proc = MagicMock(); proc.pid = 1; popen.return_value = proc
        with pytest.raises(BackendError, match="failed to start"):
            backend.load(make_model_entry(), port=8020)
        assert backend._proc is None  # unloaded after timeout


def test_load_uses_typed_error_class(backend, make_model_entry):
    from lamu.core.errors import BackendError
    with patch("subprocess.Popen") as popen, \
         patch.object(backend, "is_healthy", return_value=False), \
         patch("time.sleep"):
        proc = MagicMock(); proc.pid = 1; popen.return_value = proc
        with pytest.raises(BackendError):
            backend.load(make_model_entry(), port=8020)
