"""scripts/ — smoke import test only (GPU-bound).

These scripts use torch/transformers/etc heavily and are not unit-testable
without a real GPU. Smoke = the script's source compiles and imports under
the heavy-deps stubs from conftest.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


pytestmark = pytest.mark.gpu


SCRIPTS_DIR = Path("/home/brianklam/local-llm/scripts")


@pytest.mark.parametrize(
    "fname",
    [
        "convert_eagle_to_bin.py",
        "convert_eagle_v3_to_bin.py",
        "gen_eagle_data.py",
        "quantize-awq.py",
        "quantize_inner.py",
        "train-eagle-v3.py",
        "train_eagle_v2_standalone.py",
    ],
)
def test_script_compiles(fname):
    """Verify the file at least parses — guards against syntax-error regressions."""
    src = (SCRIPTS_DIR / fname).read_text()
    compile(src, fname, "exec")
