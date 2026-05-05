"""Tests for cli.chat_repl — endpoint discovery + streaming logic."""
from __future__ import annotations

import importlib.util
import json
import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _load_module():
    """Load cli/chat_repl.py as a regular module (no package init present)."""
    src = Path("/home/brianklam/local-llm/cli/chat_repl.py")
    spec = importlib.util.spec_from_file_location("chat_repl", src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["chat_repl"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def mod():
    return _load_module()


def _resp(payload: bytes | dict):
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = lambda self: resp
    resp.__exit__ = lambda *_: None
    return resp


def test_module_loads(mod):
    assert hasattr(mod, "main")
    assert hasattr(mod, "stream_response")


def test_probe_endpoint_returns_models(mod):
    payload = {"data": [{"id": "qwen35-27b"}, {"id": "gpt2-xl"}]}
    with patch("urllib.request.urlopen", return_value=_resp(payload)):
        out = mod.probe_endpoint("http://localhost:8020/v1/models")
    assert out == ["qwen35-27b", "gpt2-xl"]


def test_probe_endpoint_handles_unreachable(mod):
    """Currently swallows everything to []. Phase C will narrow to URLError."""
    import urllib.error
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
        assert mod.probe_endpoint("http://x/y") == []


def test_probe_endpoint_handles_bad_json(mod):
    with patch("urllib.request.urlopen", return_value=_resp(b"not json")):
        assert mod.probe_endpoint("http://x/y") == []


def test_discover_models_aggregates(mod):
    def fake_urlopen(req, *a, **k):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "8020" in url:
            return _resp({"data": [{"id": "qwen35"}]})
        return _resp({"data": []})
    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        out = mod.discover_models()
    assert out == {"qwen36": ["qwen35"]}


def test_get_available_models_flattens(mod):
    with patch.object(mod, "probe_endpoint", side_effect=[["a"], ["b", "c"], []]):
        assert mod.get_available_models() == ["a", "b", "c"]


def test_cmd_status_prints_running_summary(mod, capsys):
    """Pinned: function should print without crashing even if every endpoint fails."""
    with patch.object(mod, "probe_endpoint", return_value=[]):
        mod.cmd_status()


def test_probe_endpoint_propagates_unexpected_errors():
    """Phase C: probe_endpoint catches only expected I/O errors. RuntimeError
    is a programming bug and must propagate."""
    mod = _load_module()
    with patch("urllib.request.urlopen", side_effect=RuntimeError("bug")):
        with pytest.raises(RuntimeError):
            mod.probe_endpoint("http://x/y")
