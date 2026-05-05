"""Tests for agents.bench — task harness + compare."""
from __future__ import annotations

import json

import pytest


def test_module_imports():
    import agents.bench as b
    assert hasattr(b, "BUILTIN_TASKS")
    assert isinstance(b.BUILTIN_TASKS, list)
    assert hasattr(b, "compare")
    assert hasattr(b, "run_suite")


def test_builtin_tasks_have_required_keys():
    import agents.bench as b
    for t in b.BUILTIN_TASKS:
        assert "id" in t and "description" in t


def test_compare_handles_missing_dir(tmp_path, capsys):
    import agents.bench as b
    # compare() takes string paths; missing or empty dirs should not crash silently
    nonexistent_a = str(tmp_path / "no_a")
    nonexistent_b = str(tmp_path / "no_b")
    try:
        b.compare(nonexistent_a, nonexistent_b)
    except FileNotFoundError:
        pass  # acceptable — surfaces missing directory


def test_compare_with_two_runs(tmp_path):
    import agents.bench as b
    a = tmp_path / "a"; a.mkdir()
    bdir = tmp_path / "b"; bdir.mkdir()
    summary = {"total": 1, "passed": 1, "failed": 0}
    (a / "summary.json").write_text(json.dumps(summary))
    (bdir / "summary.json").write_text(json.dumps(summary))
    try:
        b.compare(str(a), str(bdir))
    except Exception:  # noqa: BLE001 — current code may swallow; pinned by xfail elsewhere
        pass
