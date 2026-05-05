"""Tests for server.client — LocalLLM HTTP client."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from server.client import LocalLLM, chat, get_default, models


def _resp(payload: dict | bytes):
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = lambda self: resp
    resp.__exit__ = lambda *_: None
    return resp


def test_init_defaults():
    c = LocalLLM()
    assert c.base_url.startswith("http")
    assert c.api_key
    assert c.default_model


def test_init_env_overrides(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://custom:9999/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    c = LocalLLM()
    assert c.base_url == "http://custom:9999/v1"
    assert c.api_key == "sk-test"
    assert c.default_model == "test-model"


def test_strip_think_no_marker():
    assert LocalLLM._strip_think("hi") == "hi"


def test_strip_think_with_marker():
    assert LocalLLM._strip_think("<think>x</think>answer") == "answer"


def test_chat_returns_content():
    c = LocalLLM()
    payload = {"choices": [{"message": {"content": "hi there"}}]}
    with patch("urllib.request.urlopen", return_value=_resp(payload)):
        assert c.chat("test") == "hi there"


def test_chat_strips_think_by_default():
    c = LocalLLM()
    payload = {"choices": [{"message": {"content": "<think>plan</think>final"}}]}
    with patch("urllib.request.urlopen", return_value=_resp(payload)):
        assert c.chat("x") == "final"


def test_chat_raw_keeps_think():
    c = LocalLLM()
    payload = {"choices": [{"message": {"content": "<think>plan</think>final"}}]}
    with patch("urllib.request.urlopen", return_value=_resp(payload)):
        assert "plan" in c.chat("x", raw=True)


def test_chat_multi_passes_messages():
    c = LocalLLM()
    payload = {"choices": [{"message": {"content": "out"}}]}
    captured = {}

    def fake_urlopen(req, *a, **k):
        captured["body"] = req.data
        return _resp(payload)

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        c.chat_multi([{"role": "user", "content": "hi"}])
    body = json.loads(captured["body"].decode())
    assert body["messages"] == [{"role": "user", "content": "hi"}]


def test_models_aggregates_endpoints():
    c = LocalLLM()
    def fake_urlopen(req, *a, **k):
        url = req.full_url
        if "8020" in url:
            return _resp({"data": [{"id": "qwen3.6"}]})
        if "8000" in url:
            return _resp({"data": [{"id": "luce-dflash"}]})
        # Phase C: only EXPECTED I/O errors get swallowed by models().
        raise ConnectionRefusedError("not running")
    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        out = c.models()
    assert "qwen/qwen3.6" in out
    assert "dflash/luce-dflash" in out


def test_models_handles_all_down():
    c = LocalLLM()
    with patch("urllib.request.urlopen", side_effect=ConnectionError):
        assert c.models() == []


def test_is_running_true():
    c = LocalLLM()
    with patch("urllib.request.urlopen", return_value=_resp({"data": []})):
        assert c.is_running() is True


def test_is_running_false():
    c = LocalLLM()
    with patch("urllib.request.urlopen", side_effect=ConnectionError):
        assert c.is_running() is False


def test_health_marks_each_endpoint():
    c = LocalLLM()
    with patch("urllib.request.urlopen", return_value=_resp({"ok": 1})):
        h = c.health()
    assert set(h.keys()) == {"bifrost", "qwen36", "dflash", "sglang", "gpt2proxy"}
    assert all(v == "up" for v in h.values())


def test_health_marks_down_when_unreachable():
    c = LocalLLM()
    with patch("urllib.request.urlopen", side_effect=ConnectionError):
        h = c.health()
    assert all(v == "down" for v in h.values())


def test_get_default_singleton():
    a = get_default()
    b = get_default()
    assert a is b


def test_module_level_chat_helper():
    payload = {"choices": [{"message": {"content": "ok"}}]}
    with patch("urllib.request.urlopen", return_value=_resp(payload)):
        assert chat("hi") == "ok"


def test_stream_yields_tokens():
    c = LocalLLM()
    sse = (
        b'data: {"choices":[{"delta":{"content":"a"}}]}\n'
        b'data: {"choices":[{"delta":{"content":"b"}}]}\n'
        b'data: [DONE]\n'
    )
    resp = MagicMock()
    resp.__enter__ = lambda self: iter(sse.splitlines(keepends=True))
    resp.__exit__ = lambda *_: None
    with patch("urllib.request.urlopen", return_value=resp):
        out = list(c.stream("x"))
    assert out == ["a", "b"]


def test_models_propagates_unexpected_exception():
    """Phase C: typed catches in models()/health(). RuntimeError surfaces."""
    c = LocalLLM()
    with patch("urllib.request.urlopen", side_effect=RuntimeError("bug")):
        with pytest.raises(RuntimeError):
            c.models()
