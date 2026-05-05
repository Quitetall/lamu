"""Tests for lamu.daemon — CLI dispatch."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from lamu import daemon as dmod


def test_main_no_args_exits(capsys):
    with patch.object(sys, "argv", ["lamu"]):
        with pytest.raises(SystemExit):
            dmod.main()
    assert "Usage" in capsys.readouterr().out


def test_main_unknown_command(capsys):
    with patch.object(sys, "argv", ["lamu", "blah"]):
        with pytest.raises(SystemExit):
            dmod.main()
    assert "Unknown" in capsys.readouterr().out


def test_main_dispatches_scan(monkeypatch):
    called = MagicMock()
    monkeypatch.setattr(dmod, "cmd_scan", called)
    with patch.object(sys, "argv", ["lamu", "scan"]):
        dmod.main()
    called.assert_called_once()


def test_main_dispatches_status(monkeypatch):
    called = MagicMock()
    monkeypatch.setattr(dmod, "cmd_status", called)
    with patch.object(sys, "argv", ["lamu", "status"]):
        dmod.main()
    called.assert_called_once()


def test_main_dispatches_start(monkeypatch):
    called = MagicMock()
    monkeypatch.setattr(dmod, "cmd_start", called)
    with patch.object(sys, "argv", ["lamu", "start"]):
        dmod.main()
    called.assert_called_once()


def test_cmd_scan_writes_registry(tmp_path, mock_nvidia_smi, gguf_bytes_factory, monkeypatch):
    (tmp_path / "qwen35-1.gguf").write_bytes(gguf_bytes_factory("qwen35"))
    reg = tmp_path / "registry.yaml"
    monkeypatch.setattr(dmod, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(dmod, "REGISTRY_PATH", reg)
    dmod.cmd_scan()
    assert reg.exists()


def test_cmd_status_no_running_servers(tmp_path, mock_nvidia_smi, monkeypatch, capsys):
    """All probe ports refused — current bare-except prints '⚪ :port'.
    Phase C will replace bare except with typed log + non-zero exit-info."""
    import urllib.error
    monkeypatch.setattr(dmod, "REGISTRY_PATH", tmp_path / "no.yaml")
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("nope")):
        dmod.cmd_status()
    out = capsys.readouterr().out
    assert "VRAM:" in out
    assert "not running" in out


def test_cmd_status_lists_running(tmp_path, mock_nvidia_smi, monkeypatch, capsys):
    monkeypatch.setattr(dmod, "REGISTRY_PATH", tmp_path / "no.yaml")

    def fake_open(req, timeout=1):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "/health" in url:
            body = b'{"status":"ok"}'
        else:
            body = b'{"data":[{"id":"qwen35-27b"}]}'
        resp = MagicMock()
        resp.read.return_value = body
        resp.__enter__ = lambda self: resp
        resp.__exit__ = lambda *_: None
        return resp

    with patch("urllib.request.urlopen", side_effect=fake_open):
        dmod.cmd_status()
    out = capsys.readouterr().out
    assert "qwen35-27b" in out


def test_cmd_status_typed_errors(tmp_path, mock_nvidia_smi, monkeypatch):
    """Phase C: cmd_status uses typed catches. RuntimeError is NOT in the
    expected error tuple, so it must surface."""
    monkeypatch.setattr(dmod, "REGISTRY_PATH", tmp_path / "no.yaml")
    with patch("urllib.request.urlopen", side_effect=RuntimeError("bug")):
        with pytest.raises(RuntimeError):
            dmod.cmd_status()
