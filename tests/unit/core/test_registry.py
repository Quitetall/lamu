"""Tests for lamu.core.registry — GGUF parsing, scan, YAML round-trip."""
from __future__ import annotations

from pathlib import Path

import pytest

from lamu.core.registry import (
    _detect_quant,
    _estimate_params_b,
    _estimate_vram_mb,
    _parse_gguf_metadata,
    load_registry,
    scan_directory,
    write_registry,
)
from lamu.core.types import (
    BackendType,
    Capability,
    ModelEntry,
    ModelFormat,
    ReasoningMarker,
    SpeculativeConfig,
)


def test_parse_valid_gguf(tmp_path, gguf_bytes_factory):
    p = tmp_path / "x.gguf"
    p.write_bytes(gguf_bytes_factory("qwen35", file_type=17))
    meta = _parse_gguf_metadata(p)
    assert meta["version"] == 3
    assert meta["general.architecture"] == "qwen35"
    assert meta["general.file_type"] == 17


def test_parse_bad_magic_returns_empty(tmp_path, gguf_bytes_factory):
    p = tmp_path / "bad.gguf"
    p.write_bytes(gguf_bytes_factory(bad_magic=True))
    meta = _parse_gguf_metadata(p)
    assert meta == {}


def test_parse_truncated_emits_warning(tmp_path, gguf_bytes_factory, recwarn):
    """Phase C: malformed GGUF must surface as RegistryParseWarning."""
    import warnings
    from lamu.core.errors import RegistryParseWarning
    p = tmp_path / "trunc.gguf"
    p.write_bytes(gguf_bytes_factory(truncate=10))  # header only
    warnings.simplefilter("always")
    meta = _parse_gguf_metadata(p)
    assert "general.architecture" not in meta
    assert any(issubclass(w.category, RegistryParseWarning) for w in recwarn)


def test_parse_missing_file_raises(tmp_path):
    """Phase C: explicit FileNotFoundError, not silent empty dict."""
    with pytest.raises(FileNotFoundError):
        _parse_gguf_metadata(tmp_path / "nope.gguf")


def test_estimate_vram_includes_overhead():
    assert _estimate_vram_mb(10000, "Q5_K_M") == 11000


def test_detect_quant_from_metadata():
    assert _detect_quant({"general.file_type": 17}, "anything.gguf") == "Q5_K_M"
    assert _detect_quant({"general.file_type": 7}, "anything.gguf") == "Q8_0"


def test_detect_quant_from_filename():
    assert _detect_quant({}, "Qwen-Q4_K_M.gguf") == "Q4_K_M"
    assert _detect_quant({}, "model-q5_k_s.gguf") == "Q5_K_S"


def test_detect_quant_unknown():
    assert _detect_quant({}, "weird.gguf") == "unknown"


def test_estimate_params_zero_when_meta_empty():
    assert _estimate_params_b({}) == 0.0


def test_estimate_params_nonzero_with_filesize():
    val = _estimate_params_b({"n_tensors": 100, "file_size_mb": 16384})
    assert val > 0


def test_scan_directory_finds_gguf(tmp_model_dir):
    entries = scan_directory(tmp_model_dir)
    names = [e.name for e in entries]
    assert any("qwen35" in n for n in names)
    assert all(e.format is ModelFormat.GGUF for e in entries)
    assert all(e.backend is BackendType.LLAMACPP for e in entries)


def test_scan_skips_dflash_drafts(tmp_path, gguf_bytes_factory):
    (tmp_path / "main-Q5_K_M.gguf").write_bytes(gguf_bytes_factory("qwen35"))
    (tmp_path / "main-dflash-draft.gguf").write_bytes(gguf_bytes_factory("dflash"))
    entries = scan_directory(tmp_path)
    names = [e.name for e in entries]
    assert not any("draft" in n for n in names)


def test_scan_assigns_capabilities(tmp_model_dir):
    entries = scan_directory(tmp_model_dir)
    qwen = next(e for e in entries if "qwen35" in e.name)
    assert Capability.CHAT in qwen.capabilities
    assert Capability.CODE in qwen.capabilities
    assert Capability.REASONING in qwen.capabilities


def test_scan_assigns_reasoning_marker_for_qwen(tmp_model_dir):
    entries = scan_directory(tmp_model_dir)
    qwen = next(e for e in entries if "qwen35" in e.name)
    assert qwen.reasoning_marker is not None
    assert qwen.reasoning_marker.family == "qwen35"


def test_scan_no_marker_for_gpt2(tmp_model_dir):
    entries = scan_directory(tmp_model_dir)
    gpt2 = next(e for e in entries if "gpt2" in e.name)
    assert gpt2.reasoning_marker is None


def test_scan_empty_dir(tmp_path):
    assert scan_directory(tmp_path) == []


def test_write_then_load_roundtrip(tmp_path):
    entries = [
        ModelEntry(
            name="m1", path=tmp_path / "m1.gguf",
            format=ModelFormat.GGUF, backend=BackendType.LLAMACPP,
            arch="qwen35", params_b=27.0, quant="Q5_K_M",
            vram_mb=18000, context_max=131072,
            capabilities=(Capability.CHAT, Capability.REASONING),
            reasoning_marker=ReasoningMarker(
                open_tag="<think>", close_tag="</think>", family="qwen35"),
            speculative=SpeculativeConfig(
                draft_path=tmp_path / "d.gguf", method="dflash", draft_max=4),
            pinned=True,
        ),
    ]
    out = tmp_path / "config" / "models.yaml"
    write_registry(entries, out)
    loaded = load_registry(out)
    assert len(loaded) == 1
    e = loaded[0]
    assert e.name == "m1"
    assert e.arch == "qwen35"
    assert e.pinned is True
    assert e.reasoning_marker is not None
    assert e.reasoning_marker.family == "qwen35"
    assert e.speculative is not None
    assert e.speculative.method == "dflash"
    assert e.speculative.draft_max == 4


def test_load_registry_missing_returns_empty(tmp_path):
    assert load_registry(tmp_path / "nope.yaml") == []


def test_load_registry_empty_file_returns_empty(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("")
    assert load_registry(p) == []


def test_load_registry_no_models_key(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text("other: 1\n")
    assert load_registry(p) == []
