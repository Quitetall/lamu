"""Tests for agents.base — env loading + langfuse handler."""
from __future__ import annotations

import importlib
import sys

import pytest


def test_module_imports():
    """Smoke: agents.base loads even when langfuse keys absent."""
    import agents.base as base
    assert hasattr(base, "llm")
    assert hasattr(base, "get_config")


def test_get_config_returns_dict():
    import agents.base as base
    cfg = base.get_config()
    assert isinstance(cfg, dict)
    assert "callbacks" in cfg


def test_tracing_disabled_flag_when_init_fails():
    """Phase C: agents.base exposes a public `tracing_enabled` bool."""
    import agents.base as base
    importlib.reload(base)
    assert hasattr(base, "tracing_enabled")
    assert isinstance(base.tracing_enabled, bool)
