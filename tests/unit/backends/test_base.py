"""Tests for lamu.backends.base — abstract Backend protocol."""
from __future__ import annotations

from typing import Iterator

import pytest

from lamu.backends.base import Backend
from lamu.core.types import ModelEntry


def test_backend_cannot_instantiate_directly():
    with pytest.raises(TypeError):
        Backend()  # type: ignore[abstract]


def test_subclass_must_implement_all_methods(make_model_entry):
    """Partial implementation should fail to instantiate."""
    class Partial(Backend):
        def load(self, entry, port): return 1
        # missing the rest
    with pytest.raises(TypeError):
        Partial()  # type: ignore[abstract]


def test_concrete_subclass_works(make_model_entry):
    class Stub(Backend):
        def __init__(self):
            self._port = 8020
            self._name = "stub"
        def load(self, entry: ModelEntry, port: int) -> int:
            self._port = port
            self._name = entry.name
            return 42
        def unload(self) -> None: self._name = ""
        def is_healthy(self) -> bool: return True
        def generate(self, messages, max_tokens=16384, temperature=0.7, stream=False) -> str:
            return "ok"
        def stream(self, messages, max_tokens=16384, temperature=0.7) -> Iterator[str]:
            yield from ("a", "b")
        def get_vram_mb(self) -> int: return 0
        @property
        def port(self) -> int: return self._port
        @property
        def model_name(self) -> str: return self._name

    s = Stub()
    e = make_model_entry()
    assert s.load(e, 9000) == 42
    assert s.port == 9000
    assert s.is_healthy()
    assert s.generate([]) == "ok"
    assert list(s.stream([])) == ["a", "b"]
    s.unload()
    assert s.model_name == ""
