"""Unit-test shared fixtures (build atop the root conftest)."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pytest

from lamu.core.types import (
    BackendType,
    Capability,
    ModelEntry,
    ModelFormat,
    ReasoningMarker,
)


def make_entry(
    name: str = "qwen35-27b",
    *,
    arch: str = "qwen35",
    params_b: float = 27.0,
    quant: str = "Q5_K_M",
    vram_mb: int = 18000,
    context_max: int = 131072,
    capabilities: Sequence[Capability] = (
        Capability.CHAT, Capability.CODE, Capability.REASONING,
    ),
    pinned: bool = False,
    path: Path | None = None,
) -> ModelEntry:
    return ModelEntry(
        name=name,
        path=path or Path(f"/tmp/{name}.gguf"),
        format=ModelFormat.GGUF,
        backend=BackendType.LLAMACPP,
        arch=arch,
        params_b=params_b,
        quant=quant,
        vram_mb=vram_mb,
        context_max=context_max,
        capabilities=tuple(capabilities),
        reasoning_marker=ReasoningMarker(
            open_tag="<think>", close_tag="</think>", family=arch,
        ) if arch.startswith("qwen") else None,
        pinned=pinned,
    )


@pytest.fixture
def make_model_entry():
    return make_entry


@pytest.fixture
def sample_registry():
    return [
        make_entry("qwen35-27b", vram_mb=18000),
        make_entry(
            "qwen35-0.8b",
            params_b=0.8, quant="Q8_0", vram_mb=900,
            capabilities=(Capability.CHAT, Capability.ROUTING),
        ),
        make_entry(
            "gpt2-xl",
            arch="gpt2", params_b=1.5, quant="F16", vram_mb=3000,
            context_max=1024,
            capabilities=(Capability.CHAT,),
        ),
    ]
