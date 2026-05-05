"""Tests for server.poincare — knowledge-graph viz (smoke only)."""
from __future__ import annotations


def test_module_imports():
    from server import poincare
    assert hasattr(poincare, "PoincareBallEmbedding")
    assert hasattr(poincare, "load_graphify_json")
    assert hasattr(poincare, "create_poincare_plot")
    assert hasattr(poincare, "main")


def test_load_graphify_json_missing(tmp_path):
    """Currently raises FileNotFoundError or similar — pin behavior."""
    from server import poincare
    import pytest
    with pytest.raises((FileNotFoundError, Exception)):
        poincare.load_graphify_json(str(tmp_path / "missing.json"))


def test_load_graphify_json_empty_graph(tmp_path):
    from server import poincare
    import json
    p = tmp_path / "g.json"
    p.write_text(json.dumps({"nodes": [], "edges": []}))
    g = poincare.load_graphify_json(str(p))
    # MagicMock NetworkX may not return real graph — just verify call doesn't raise.
    assert g is not None
