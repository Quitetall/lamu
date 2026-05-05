"""Tests for lamu.core.config — path constants + ports."""
from __future__ import annotations

from pathlib import Path

from lamu.core import config


def test_paths_are_path_objects():
    assert isinstance(config.LAMU_ROOT, Path)
    assert isinstance(config.MODELS_DIR, Path)
    assert isinstance(config.REGISTRY_PATH, Path)
    assert isinstance(config.LLAMA_BIN, Path)


def test_registry_under_lamu_root():
    assert config.REGISTRY_PATH.is_relative_to(config.LAMU_ROOT) or \
        str(config.LAMU_ROOT) in str(config.REGISTRY_PATH)


def test_ports_distinct_and_in_range():
    ports = [config.PORT_MAIN, config.PORT_SIDECAR, config.PORT_DFLASH]
    assert len(set(ports)) == 3
    assert all(1024 <= p <= 65535 for p in ports)


def test_vram_reserved_positive():
    assert config.VRAM_RESERVED_MB > 0


def test_defaults_sane():
    assert config.DEFAULT_MAX_TOKENS > 0
    assert 0 < config.DEFAULT_TEMPERATURE <= 2.0
    assert config.DEFAULT_CTX_SIZE >= 4096
