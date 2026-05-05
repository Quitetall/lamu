"""All API endpoints must return within 1s when no backend is loaded."""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from lamu.api.openai_compat import create_app
from lamu.core.scheduler import VramScheduler


@pytest.fixture
def client(mock_nvidia_smi, sample_registry):
    sched = VramScheduler(reserved_mb=1500)
    # No models registered as loaded — every chat request must 503 fast.
    return TestClient(create_app(sched, sample_registry))


def _under(client, method: str, path: str, **kwargs) -> float:
    t0 = time.monotonic()
    if method == "GET":
        client.get(path)
    else:
        client.post(path, **kwargs)
    return time.monotonic() - t0


def test_health_endpoint_under_1s(client):
    assert _under(client, "GET", "/health") < 1.0


def test_models_endpoint_under_1s(client):
    assert _under(client, "GET", "/v1/models") < 1.0


def test_chat_completions_503_under_1s(client):
    elapsed = _under(
        client, "POST", "/v1/chat/completions",
        json={"model": "qwen35-27b",
              "messages": [{"role": "user", "content": "hi"}]},
    )
    assert elapsed < 1.0


def test_chat_completions_validation_under_1s(client):
    elapsed = _under(
        client, "POST", "/v1/chat/completions",
        json={"messages": "garbage"},
    )
    assert elapsed < 1.0
