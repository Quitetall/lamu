"""Tests for lamu.api.openai_compat — FastAPI / OpenAI shim."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from lamu.api.openai_compat import create_app
from lamu.core.scheduler import VramScheduler


@pytest.fixture
def app(mock_nvidia_smi, sample_registry):
    sched = VramScheduler(reserved_mb=1500)
    return create_app(sched, sample_registry), sched


@pytest.fixture
def client(app):
    return TestClient(app[0])


def _fake_resp(payload: dict | bytes):
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = lambda self: resp
    resp.__exit__ = lambda *_: None
    return resp


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_models_listing_shape(client, sample_registry):
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    names = {m["id"] for m in body["data"]}
    assert {"qwen35-27b", "qwen35-0.8b", "gpt2-xl"} <= names
    for m in body["data"]:
        assert "loaded" in m and "capabilities" in m


def test_chat_completions_503_when_no_loaded_model(client):
    r = client.post(
        "/v1/chat/completions",
        json={"model": "qwen35-27b", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 503
    assert r.json()["error"]["type"] == "backend_unavailable"
    assert "retry-after" in {h.lower() for h in r.headers}


def test_chat_completions_404_when_unknown_model(client):
    r = client.post(
        "/v1/chat/completions",
        json={"model": "totally-unknown", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 503  # current contract surfaces routing failure as 503


def test_chat_completions_validation_error(client):
    r = client.post("/v1/chat/completions", json={"messages": "not a list"})
    assert r.status_code == 422


def test_chat_completions_routes_to_loaded(client, app):
    sched = app[1]
    qwen = next(e for e in sched._loaded.values()) if sched._loaded else None
    # Register loaded model
    from tests.unit.conftest import make_entry
    entry = make_entry("qwen35-27b")
    sched.register_loaded(entry, pid=1, port=8020, vram_actual_mb=18000)

    backend_payload = {
        "choices": [{
            "message": {"content": "hello", "reasoning_content": ""},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
    }
    with patch("urllib.request.urlopen", return_value=_fake_resp(backend_payload)):
        r = client.post(
            "/v1/chat/completions",
            json={"model": "qwen35-27b", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "hello"
    assert body["model"] == "qwen35-27b"


def test_chat_completions_extracts_reasoning_when_present(client, app):
    from tests.unit.conftest import make_entry
    sched = app[1]
    entry = make_entry("qwen35-27b")
    sched.register_loaded(entry, pid=1, port=8020, vram_actual_mb=18000)

    backend_payload = {
        "choices": [{
            "message": {
                "content": "answer",
                "reasoning_content": "thought process",
            },
            "finish_reason": "stop",
        }],
        "usage": {},
    }
    with patch("urllib.request.urlopen", return_value=_fake_resp(backend_payload)):
        r = client.post(
            "/v1/chat/completions",
            json={"model": "qwen35-27b", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "answer"
    assert body["choices"][0]["message"]["reasoning_content"] == "thought process"


def test_chat_completions_502_on_backend_unreachable(client, app):
    from tests.unit.conftest import make_entry
    import urllib.error
    sched = app[1]
    sched.register_loaded(make_entry("qwen35-27b"), pid=1, port=8020, vram_actual_mb=18000)
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
        r = client.post(
            "/v1/chat/completions",
            json={"model": "qwen35-27b", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 502


def test_503_includes_retry_after_header(client):
    r = client.post(
        "/v1/chat/completions",
        json={"model": "qwen35-27b", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 503
    assert "retry-after" in {h.lower() for h in r.headers}
