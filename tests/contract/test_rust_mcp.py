"""Smoke contract: Rust MCP server replays the corpus and produces the
same shape of responses as Python."""
from __future__ import annotations

import pytest

from tests.contract.conftest import normalise, replay


@pytest.mark.contract
def test_rust_corpus_runs(rust_mcp_cmd, corpus):
    responses = replay(rust_mcp_cmd, corpus, timeout=20)
    expected = sum(1 for r in corpus if "id" in r)
    assert len(responses) == expected, (
        f"got {len(responses)} responses for {expected} id'd requests; "
        f"raw responses: {responses}"
    )
    for r in responses:
        assert r["jsonrpc"] == "2.0"
        assert "id" in r
        assert "result" in r or "error" in r


@pytest.mark.contract
def test_rust_initialize_advertises_tools(rust_mcp_cmd, corpus):
    responses = replay(rust_mcp_cmd, corpus, timeout=20)
    init_resp = next(r for r in responses if r["id"] == 1)
    norm = normalise(init_resp)
    assert norm["jsonrpc"] == "2.0"
    caps = norm["result"]["capabilities"]
    assert "tools" in caps


@pytest.mark.contract
def test_rust_tools_list_includes_query(rust_mcp_cmd, corpus):
    responses = replay(rust_mcp_cmd, corpus, timeout=20)
    list_resp = next(r for r in responses if r["id"] == 2)
    names = {t["name"] for t in list_resp["result"]["tools"]}
    assert {
        "query", "plan_query", "list_models", "load_model",
        "unload_model", "vram_status", "scan_models", "queue_status",
    }.issubset(names)
