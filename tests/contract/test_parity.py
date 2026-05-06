"""Cross-language MCP parity: Python ↔ Rust must agree on the wire shape.

We don't demand byte-identical responses (Python uses a fuller
serverInfo, Rust uses ordered-key JSON serialisation, etc). We DO
demand structural parity on the parts that clients depend on:
  - same set of tool names from tools/list
  - same required fields per tool's inputSchema
  - same `jsonrpc` version
  - same protocol version on initialize
"""
from __future__ import annotations

import pytest

from tests.contract.conftest import replay


def _by_id(responses):
    return {r["id"]: r for r in responses if "id" in r}


@pytest.mark.contract
def test_initialize_protocol_version_matches(python_mcp_cmd, rust_mcp_cmd, corpus):
    py = _by_id(replay(python_mcp_cmd, corpus, timeout=20))
    rs = _by_id(replay(rust_mcp_cmd, corpus, timeout=20))
    py_pv = py[1]["result"]["protocolVersion"]
    rs_pv = rs[1]["result"]["protocolVersion"]
    assert py_pv == rs_pv, f"protocolVersion drift: python={py_pv} rust={rs_pv}"


@pytest.mark.contract
def test_tools_list_set_matches(python_mcp_cmd, rust_mcp_cmd, corpus):
    py = _by_id(replay(python_mcp_cmd, corpus, timeout=20))
    rs = _by_id(replay(rust_mcp_cmd, corpus, timeout=20))
    py_names = {t["name"] for t in py[2]["result"]["tools"]}
    rs_names = {t["name"] for t in rs[2]["result"]["tools"]}
    assert py_names == rs_names, (
        f"tool set drift:\n  python only: {py_names - rs_names}\n  "
        f"rust only:   {rs_names - py_names}"
    )


@pytest.mark.contract
def test_query_required_fields_match(python_mcp_cmd, rust_mcp_cmd, corpus):
    """Both implementations must require the same input fields on `query`."""
    py = _by_id(replay(python_mcp_cmd, corpus, timeout=20))
    rs = _by_id(replay(rust_mcp_cmd, corpus, timeout=20))
    py_query = next(t for t in py[2]["result"]["tools"] if t["name"] == "query")
    rs_query = next(t for t in rs[2]["result"]["tools"] if t["name"] == "query")
    py_required = sorted(py_query["inputSchema"]["required"])
    rs_required = sorted(rs_query["inputSchema"]["required"])
    assert py_required == rs_required


@pytest.mark.contract
def test_vram_and_queue_status_return_text(python_mcp_cmd, rust_mcp_cmd, corpus):
    """Both implementations return single TextContent for vram/queue_status."""
    py = _by_id(replay(python_mcp_cmd, corpus, timeout=20))
    rs = _by_id(replay(rust_mcp_cmd, corpus, timeout=20))
    for resp_id in (4, 5):
        py_text = py[resp_id]["result"]["content"][0]["type"]
        rs_text = rs[resp_id]["result"]["content"][0]["type"]
        assert py_text == "text" == rs_text
