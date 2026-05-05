"""Backend dies mid-request → 503 within timeout, daemon stays up."""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest
import urllib.error
from fastapi.testclient import TestClient

from lamu.api.openai_compat import create_app
from lamu.core.scheduler import VramScheduler


@pytest.fixture
def client(mock_nvidia_smi, sample_registry):
    sched = VramScheduler(reserved_mb=1500)
    qwen = sample_registry[0]
    sched.register_loaded(qwen, pid=1, port=8020, vram_actual_mb=18000)
    app = create_app(sched, sample_registry)
    return TestClient(app)


def test_backend_unreachable_returns_502_fast(client):
    """When the backend refuses connection, API returns 502 in <1s
    (NOT a hang waiting for 300s timeout)."""
    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("connection refused"),
    ):
        t0 = time.monotonic()
        r = client.post(
            "/v1/chat/completions",
            json={"model": "qwen35-27b",
                  "messages": [{"role": "user", "content": "hi"}]},
        )
        elapsed = time.monotonic() - t0
    assert r.status_code == 502
    assert elapsed < 1.0  # no hang


def test_daemon_survives_after_backend_failure(client):
    """Sequential failures don't bring down the daemon process."""
    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("dead"),
    ):
        for _ in range(5):
            r = client.post(
                "/v1/chat/completions",
                json={"model": "qwen35-27b",
                      "messages": [{"role": "user", "content": "x"}]},
            )
            assert r.status_code in (502, 503)

    # Daemon still serves /health afterwards
    r = client.get("/health")
    assert r.status_code == 200
