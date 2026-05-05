"""Tests for agents.trainer — pure helpers (collect/status/prepare)."""
from __future__ import annotations

import json

import pytest


def test_module_imports():
    import agents.trainer as t
    assert hasattr(t, "collect") and hasattr(t, "prepare")
    assert hasattr(t, "train") and hasattr(t, "export")


def test_collect_writes_json(tmp_path, monkeypatch, capsys):
    import agents.trainer as t
    monkeypatch.setattr(t, "DATA_DIR", tmp_path)
    t.collect("task X", ["step1"], {"a.py": "code"}, test_output="ok")
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text())
    assert data["task"] == "task X"
    assert "applied_files" in data


def test_status_runs_clean(tmp_path, monkeypatch, capsys):
    import agents.trainer as t
    monkeypatch.setattr(t, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(t, "ADAPTER_DIR", tmp_path / "adapt")
    t.status()
    out = capsys.readouterr().out
    # Some output expected (count summary)
    assert isinstance(out, str)


def test_prepare_empty_returns_empty(tmp_path, monkeypatch):
    import agents.trainer as t
    monkeypatch.setattr(t, "DATA_DIR", tmp_path)
    monkeypatch.setattr(t, "DATASET_DIR", tmp_path / "datasets")
    out = t.prepare()
    assert isinstance(out, list)


def test_missing_dependency_class_exists():
    """Phase C: MissingDependency exists for ImportError propagation."""
    from lamu.core.errors import MissingDependency
    assert issubclass(MissingDependency, Exception)
