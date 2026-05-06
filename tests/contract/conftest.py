"""Shared scaffolding for cross-language MCP contract tests.

Spawn either Python or Rust MCP server as a subprocess, pipe corpus
requests on stdin, collect JSON-RPC responses on stdout. Compare the
two with a normalisation pass that strips id/timestamp/version noise.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import pytest


CORPUS_PATH = Path(__file__).parent / "corpus.jsonl"
ROOT = Path(__file__).resolve().parents[2]


def _load_corpus() -> list[dict]:
    """Read corpus.jsonl into a list of JSON requests."""
    out: list[dict] = []
    for line in CORPUS_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


@pytest.fixture(scope="module")
def corpus() -> list[dict]:
    return _load_corpus()


def replay(cmd: list[str], requests: Iterable[dict], timeout: int = 30) -> list[dict]:
    """Spawn `cmd`, write each request, read responses interactively.

    Two-thread approach: writer feeds requests one-by-one with a 5ms gap
    so the server's stdin buffer doesn't see EOF before processing; reader
    pulls JSON lines off stdout. Returns once N expected responses arrive
    or `timeout` elapses.

    A single subprocess.run() with bulk input + EOF was unreliable for the
    Python MCP server: when the asyncio event loop sees stdin EOF before
    consuming the buffered tail, it shuts down mid-corpus. Real MCP
    clients keep stdin open — match that.
    """
    requests = list(requests)
    expected = sum(1 for r in requests if "id" in r)

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
        cwd=str(ROOT),
    )

    out: list[dict] = []
    import threading
    import time

    def reader():
        for line in proc.stdout:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(out) >= expected:
                return

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    try:
        for r in requests:
            proc.stdin.write(json.dumps(r) + "\n")
            proc.stdin.flush()
            time.sleep(0.02)  # let the server consume before next write
        # Wait for reader to collect everything (or hit timeout).
        t.join(timeout=timeout)
    finally:
        try:
            proc.stdin.close()
        except (BrokenPipeError, ValueError):
            pass
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
    return out


def normalise(resp: dict) -> dict:
    """Strip fields that legitimately differ between implementations.

    - `id`: drop (Python and Rust echo verbatim, but keep this stable to
      survive future framing changes).
    - `serverInfo.version`: tolerate version drift.
    - timestamps anywhere: drop.

    Keeps the structural shape, capability flags, tool names, and tool
    output text — those MUST match.
    """
    if not isinstance(resp, dict):
        return resp
    keep_id = resp.get("id")
    out = dict(resp)
    out.pop("id", None)
    if isinstance(out.get("result"), dict):
        out["result"] = _scrub(out["result"])
    return out


def _scrub(node):
    if isinstance(node, dict):
        # Drop known volatile fields wholesale.
        for k in ("created", "timestamp", "ts", "last_error_unix"):
            node.pop(k, None)
        # serverInfo.version drift.
        if "serverInfo" in node and isinstance(node["serverInfo"], dict):
            node["serverInfo"].pop("version", None)
        return {k: _scrub(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_scrub(v) for v in node]
    return node


@pytest.fixture(scope="module")
def python_mcp_cmd() -> list[str]:
    """How to launch the Python MCP server.

    Uses the project venv's interpreter; skips the contract suite if the
    venv isn't materialised.
    """
    py = ROOT / ".venv" / "bin" / "python"
    if not py.exists():
        pytest.skip(f"Python venv not present at {py}")
    return [str(py), "-m", "lamu", "start"]


@pytest.fixture(scope="module")
def rust_mcp_cmd() -> list[str]:
    """How to launch the Rust MCP server.

    Looks for the release binary; skips when it's not built (CI builds
    it in the rust-tests job).
    """
    rs = ROOT / "lamu-rs" / "target" / "release" / "lamu"
    if not rs.exists():
        pytest.skip(f"Rust binary not built at {rs} (run cargo build --release)")
    return [str(rs), "start"]
