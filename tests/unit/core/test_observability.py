"""Tests for lamu.core.observability — structured event sink."""
from __future__ import annotations

import json
import os

import pytest

from lamu.core.observability import emit, new_trace_id


def test_emit_writes_jsonl_to_stderr(capsys):
    emit("test_event", foo="bar", n=3)
    err = capsys.readouterr().err.strip()
    obj = json.loads(err)
    assert obj["event"] == "test_event"
    assert obj["foo"] == "bar"
    assert obj["n"] == 3


def test_emit_includes_trace_id(capsys):
    emit("e", trace_id="abc123def")
    obj = json.loads(capsys.readouterr().err.strip())
    assert obj["trace_id"] == "abc123def"


def test_emit_appends_to_file_sink(tmp_path, capsys, monkeypatch):
    log = tmp_path / "events.jsonl"
    monkeypatch.setenv("LAMU_EVENT_LOG", str(log))

    emit("e1", k=1)
    emit("e2", k=2)
    capsys.readouterr()  # drain stderr

    lines = log.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["event"] == "e1"
    assert json.loads(lines[1])["event"] == "e2"


def test_emit_swallows_bad_file_sink(monkeypatch, capsys):
    """A broken file sink must not block the stderr emit."""
    monkeypatch.setenv("LAMU_EVENT_LOG", "/proc/1/cannot-write-here.jsonl")
    emit("survives", info="ok")
    err = capsys.readouterr().err.strip()
    assert "survives" in err  # stderr got the event regardless


def test_new_trace_id_is_16_hex_chars():
    tid = new_trace_id()
    assert len(tid) == 16
    int(tid, 16)  # parses as hex


def test_emit_default_no_file_sink(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("LAMU_EVENT_LOG", raising=False)
    emit("only_stderr")
    capsys.readouterr()  # drain
    # No file should have been created since LAMU_EVENT_LOG is unset.
    assert list(tmp_path.iterdir()) == []
