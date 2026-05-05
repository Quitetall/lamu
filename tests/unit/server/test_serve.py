"""Tests for server.serve — pure helpers + ASGI middleware shape."""
from __future__ import annotations

import json

import pytest

from server import serve


def test_strip_think_no_marker():
    assert serve.strip_think("hello") == "hello"


def test_strip_think_simple():
    assert serve.strip_think("<think>x</think>answer") == "answer"


def test_strip_think_only_close_partition():
    assert serve.strip_think("noisy</think>core") == "core"


def test_kv_type_q5():
    assert serve.kv_type_for_quant("model-Q5_K_M.gguf") == 8


def test_kv_type_other():
    assert serve.kv_type_for_quant("model-Q4_K_M.gguf") == 2


def test_ctx_for_quant_q5_default():
    if serve.CTX == 262144:
        assert serve.ctx_for_quant("foo-Q5_K_M.gguf") == 108000
    else:
        assert serve.ctx_for_quant("foo-Q5_K_M.gguf") == serve.CTX


def test_find_gguf_raises_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(serve, "MODELS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        serve.find_gguf()


def test_find_gguf_returns_first_match(monkeypatch, tmp_path):
    p = tmp_path / "model-Q5_K_S.gguf"
    p.write_bytes(b"x")
    monkeypatch.setattr(serve, "MODELS_DIR", tmp_path)
    assert serve.find_gguf() == str(p)


def test_think_strip_asgi_class_init():
    inner = object()
    middleware = serve.ThinkStripASGI(inner)
    assert middleware.app is inner


def test_filter_sse_strips_think():
    middleware = serve.ThinkStripASGI(app=None)
    chunk = (
        'data: {"choices":[{"delta":{"content":"<think>foo"}}]}\n'
        'data: {"choices":[{"delta":{"content":"</think>visible"}}]}\n'
        'data: [DONE]\n'
    )
    out = middleware._filter_sse(chunk)
    assert "<think>" not in out
    assert "visible" in out


def test_filter_sse_works_without_priming():
    """Phase C: _think_done initialized in __init__ — middleware usable
    standalone without a prior __call__. Verifies no AttributeError."""
    middleware = serve.ThinkStripASGI(app=None)
    # First, the middleware suppresses content until it sees </think>:
    pre = middleware._filter_sse(
        'data: {"choices":[{"delta":{"content":"thinking..."}}]}\n'
    )
    assert "thinking..." not in pre
    # After </think> arrives, subsequent content passes through.
    middleware._filter_sse(
        'data: {"choices":[{"delta":{"content":"</think>"}}]}\n'
    )
    out = middleware._filter_sse(
        'data: {"choices":[{"delta":{"content":"hello"}}]}\n'
    )
    assert "hello" in out


def test_filter_sse_passthrough_on_bad_json():
    """Malformed SSE chunks pass through verbatim — filtering is best-effort
    and must never crash an in-flight stream."""
    middleware = serve.ThinkStripASGI(app=None)
    out = middleware._filter_sse("data: {totally bad json\n")
    # The line is preserved (not dropped) so downstream sees it.
    assert "totally bad json" in out
