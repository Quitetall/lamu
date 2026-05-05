"""Tests for server.mcp_qwen — legacy MCP server."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


def _resp(payload: dict | bytes):
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = lambda self: resp
    resp.__exit__ = lambda *_: None
    return resp


def test_module_imports():
    from server import mcp_qwen
    assert hasattr(mcp_qwen, "main")
    assert hasattr(mcp_qwen, "_chat")
    assert hasattr(mcp_qwen, "_discover_models")


def test_discover_models_aggregates():
    from server import mcp_qwen
    with patch(
        "urllib.request.urlopen",
        return_value=_resp({"data": [{"id": "qwen-test"}]}),
    ):
        out = mcp_qwen._discover_models()
    assert isinstance(out, dict)
    # Each endpoint either gets the model list or an empty fallback
    assert all(isinstance(v, list) for v in out.values())


def test_discover_models_handles_all_down():
    from server import mcp_qwen
    with patch("urllib.request.urlopen", side_effect=ConnectionError):
        out = mcp_qwen._discover_models()
    assert all(out[k] == [] for k in out)


def test_chat_returns_content():
    from server import mcp_qwen
    payload = {"choices": [{"message": {"content": "hi"}}]}
    with patch("urllib.request.urlopen", return_value=_resp(payload)):
        out = mcp_qwen._chat("hello")
    assert "hi" in out


def test_chat_strips_think():
    from server import mcp_qwen
    payload = {"choices": [{"message": {"content": "<think>x</think>final"}}]}
    with patch("urllib.request.urlopen", return_value=_resp(payload)):
        out = mcp_qwen._chat("hi")
    assert "<think>" not in out


def test_chat_typed_error_on_unreachable():
    """Phase C: connection failures raise BackendError, not silent strings."""
    from server import mcp_qwen
    from lamu.core.errors import BackendError
    with patch("urllib.request.urlopen", side_effect=ConnectionError("nope")):
        with pytest.raises(BackendError):
            mcp_qwen._chat("hi")
