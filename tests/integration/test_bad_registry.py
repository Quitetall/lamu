"""Corrupt YAML registry → load_registry raises (no silent fallback)."""
from __future__ import annotations

import pytest
import yaml

from lamu.core.registry import load_registry


def test_corrupt_yaml_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("models: {qwen35: [unclosed list, no closing bracket\n")
    with pytest.raises(yaml.YAMLError):
        load_registry(p)


def test_missing_required_field_raises(tmp_path):
    p = tmp_path / "missing.yaml"
    p.write_text("models:\n  qwen35:\n    arch: qwen35\n")  # no path/format/etc
    with pytest.raises(KeyError):
        load_registry(p)


def test_unknown_capability_raises(tmp_path):
    p = tmp_path / "bad_cap.yaml"
    p.write_text(
        "models:\n"
        "  qwen35:\n"
        "    path: /tmp/x.gguf\n"
        "    format: gguf\n"
        "    backend: llama_cpp\n"
        "    arch: qwen35\n"
        "    params_b: 27.0\n"
        "    quant: Q5_K_M\n"
        "    vram_mb: 18000\n"
        "    context_max: 131072\n"
        "    capabilities: [chat, totally_fake_capability]\n"
    )
    with pytest.raises(ValueError):
        load_registry(p)
