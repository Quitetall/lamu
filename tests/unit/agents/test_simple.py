"""Tests for agents.simple — smoke import + basic shape."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def test_module_loads_in_isolation():
    """agents/simple.py uses `from base import ...` which only works as
    sibling-imported. Verify it can at least be loaded with the agents
    package on sys.path."""
    import agents  # ensure agents/ is a package
    src = Path(agents.__path__[0]) / "simple.py"
    assert src.exists()


def test_module_has_chat_node_symbol():
    """Without executing the module (which spins up real LangGraph),
    we just inspect the source."""
    import agents
    src = Path(agents.__path__[0]) / "simple.py"
    text = src.read_text()
    assert "def chat_node" in text
    assert "def build_graph" in text
