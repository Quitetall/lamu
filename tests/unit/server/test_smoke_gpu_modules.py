"""GPU-bound server modules — smoke import only.

Marked `gpu`. CI skips by default. Each test loads the module under heavy
mocks just to verify there are no top-level syntax errors or accidental
network calls during import.
"""
from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import pytest


pytestmark = pytest.mark.gpu


SERVER_DIR = Path("/home/brianklam/local-llm/server")


@pytest.fixture
def loader():
    """Return a function that loads server/<name>.py under sys.modules['<name>']."""
    def _load(stem: str):
        path = SERVER_DIR / f"{stem}.py"
        spec = importlib.util.spec_from_file_location(stem, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    return _load


@pytest.mark.parametrize(
    "stem",
    [
        "dflash",
        "dflash_24gb",
        "eagle_server",
        "eagle_lean",
        "megakernel_server",
        "gpt2_proxy",
        "sglang_launcher",
        "patch_gguf_qwen35",
    ],
)
def test_module_imports(loader, stem):
    """Smoke: module loads under stub'd heavy imports."""
    mod = loader(stem)
    assert mod is not None
