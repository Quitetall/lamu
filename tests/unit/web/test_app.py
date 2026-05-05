"""Tests for web.app — Chainlit frontend pure helpers."""
from __future__ import annotations

import importlib
import sys

import pytest


@pytest.fixture
def app_mod(monkeypatch, tmp_path):
    """Import web/app.py — it does `from data_layer import ...` so the web/
    directory must be on sys.path."""
    web_dir = "/home/brianklam/local-llm/web"
    if web_dir not in sys.path:
        monkeypatch.syspath_prepend(web_dir)
    for name in ("web.app", "web.data_layer", "web", "data_layer", "app"):
        sys.modules.pop(name, None)

    import data_layer as dl  # noqa: WPS433 — sibling import per script layout
    monkeypatch.setattr(dl, "DB_PATH", str(tmp_path / "chats.db"))
    dl._init_db()

    import importlib.util
    spec = importlib.util.spec_from_file_location("app", f"{web_dir}/app.py")
    app = importlib.util.module_from_spec(spec)
    sys.modules["app"] = app
    spec.loader.exec_module(app)
    return app


def test_module_imports(app_mod):
    assert hasattr(app_mod, "extract_think")
    assert hasattr(app_mod, "build_llm")
    assert hasattr(app_mod, "python_repl")


def test_extract_think_no_marker(app_mod):
    assert app_mod.extract_think("just an answer") == ("", "just an answer")


def test_extract_think_with_marker(app_mod):
    out = app_mod.extract_think("<think>thoughts</think>answer")
    assert out[0] == "<think>thoughts"
    assert out[1] == "answer"


def _call_tool(tool, code: str) -> str:
    """python_repl is decorated with @tool → StructuredTool. Unwrap to .func."""
    raw = getattr(tool, "func", None) or getattr(tool, "_run", None) or tool
    return raw(code)


def test_python_repl_runs_simple(app_mod):
    out = _call_tool(app_mod.python_repl, "print(2+2)")
    assert "4" in out


def test_python_repl_no_output(app_mod):
    out = _call_tool(app_mod.python_repl, "x = 1")
    assert out == "(no output)"


def test_python_repl_returns_typed_error_string(app_mod):
    """Phase C: error message includes exception class name so the model
    can act on the failure instead of seeing a generic 'Error: ...' blob."""
    out = _call_tool(app_mod.python_repl, "raise ValueError('x')")
    assert "ValueError" in out


def test_python_repl_syntax_error_returned_as_string(app_mod):
    """python_repl is a model-facing tool — syntax errors come back as text
    so the model can self-correct."""
    out = _call_tool(app_mod.python_repl, "def bad(:")
    assert "SyntaxError" in out
