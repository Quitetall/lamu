"""Root pytest configuration.

Order matters: sys.modules patches MUST run before any project import. The
mock-injection happens at module import time (before fixtures resolve), so
heavy/optional deps never touch the runtime.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Iterable
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Heavy-import shims — installed at conftest import (before any fixture runs).
# ---------------------------------------------------------------------------

_HEAVY_MODULES: tuple[str, ...] = (
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.cuda",
    "torch.utils",
    "torch.utils.data",
    "transformers",
    "peft",
    "unsloth",
    "sglang",
    "llama_cpp",
    "chainlit",
    "chainlit.data",
    "chainlit.data.base",
    "chainlit.element",
    "chainlit.types",
    "chainlit.user",
    "chainlit.input_widget",
    "chainlit.message",
    "langfuse",
    "langfuse.callback",
    "langchain",
    "langchain.schema",
    "langchain_core",
    "langchain_core.messages",
    "langchain_openai",
    "langchain_community",
    "langchain_community.tools",
    "langchain_community.utilities",
    "langgraph",
    "langgraph.graph",
    "langgraph.graph.message",
    "langgraph.checkpoint",
    "langgraph.checkpoint.memory",
    "datasets",
    "auto_gptq",
    "bitsandbytes",
    "geoopt",
    "plotly",
    "plotly.graph_objects",
    "networkx",
    "dotenv",
    "rich",
    "rich.console",
    "rich.markdown",
    "rich.panel",
    "rich.rule",
    "rich.text",
    "rich.live",
    "rich.spinner",
    "rich.table",
    "typing_extensions",
)


class _StubModule(ModuleType):
    """Real ModuleType so Python's import machinery accepts __spec__/__path__."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.__path__ = []  # marks as package
        self.__file__ = f"<stub:{name}>"
        self._mock = MagicMock(name=name)

    def __getattr__(self, item: str):  # called for missing attributes only
        if item.startswith("__") and item.endswith("__"):
            raise AttributeError(item)
        return getattr(self._mock, item)


def _install_stub(name: str) -> ModuleType:
    """Install a permissive stub module so `from x.y import Z` works."""
    if name in sys.modules:
        return sys.modules[name]
    stub = _StubModule(name)
    sys.modules[name] = stub
    return stub


for _modname in _HEAVY_MODULES:
    try:
        __import__(_modname)
    except Exception:  # noqa: BLE001 — intentional, heavy deps optional
        _install_stub(_modname)


# Preset stub classes for libraries the project actually subclasses.
def _kwargs_init(self, *args, **kwargs):  # noqa: D401
    self.__dict__.update(kwargs)
    if args:
        self._args = args


def _ensure_real_class(modpath: str, attr: str) -> None:
    mod = sys.modules.get(modpath)
    if mod is None:
        mod = _install_stub(modpath)
    if not isinstance(getattr(mod, attr, None), type):
        cls = type(attr, (object,), {"__init__": _kwargs_init})
        setattr(mod, attr, cls)


_ensure_real_class("chainlit.data.base", "BaseDataLayer")
_ensure_real_class("chainlit.user", "PersistedUser")
_ensure_real_class("chainlit.user", "User")
_ensure_real_class("langchain_core.messages", "AIMessage")
_ensure_real_class("langchain_core.messages", "HumanMessage")
_ensure_real_class("langchain_core.messages", "SystemMessage")
_ensure_real_class("langchain_core.messages", "ToolMessage")


# Some libraries probe attributes on import — give them sane shapes.
if hasattr(sys.modules.get("torch", object()), "cuda"):
    try:
        sys.modules["torch"].cuda.is_available = MagicMock(return_value=False)
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# GGUF synthesis helper — used by registry tests and several fixtures.
# ---------------------------------------------------------------------------

_GGUF_MAGIC = b"GGUF"


def make_gguf_bytes(
    arch: str = "qwen35",
    file_type: int = 17,  # Q5_K_M
    *,
    truncate: int | None = None,
    bad_magic: bool = False,
) -> bytes:
    """Build a minimal but parseable GGUF blob.

    truncate: if set, return only first N bytes (for corruption tests).
    bad_magic: replace magic bytes (for invalid-file tests).
    """
    buf = bytearray()
    buf += b"XXXX" if bad_magic else _GGUF_MAGIC
    buf += struct.pack("<I", 3)              # version
    buf += struct.pack("<Q", 100)            # n_tensors
    buf += struct.pack("<Q", 2)              # n_kv

    def kv_string(key: str, val: str) -> bytes:
        out = bytearray()
        out += struct.pack("<Q", len(key))
        out += key.encode()
        out += struct.pack("<I", 8)          # type=string
        out += struct.pack("<Q", len(val))
        out += val.encode()
        return bytes(out)

    def kv_uint32(key: str, val: int) -> bytes:
        out = bytearray()
        out += struct.pack("<Q", len(key))
        out += key.encode()
        out += struct.pack("<I", 4)          # type=uint32
        out += struct.pack("<I", val)
        return bytes(out)

    buf += kv_string("general.architecture", arch)
    buf += kv_uint32("general.file_type", file_type)

    raw = bytes(buf)
    if truncate is not None:
        # Honour explicit truncation — no padding (corruption tests need
        # the file to actually be short).
        return raw[:truncate]
    # Pad with zeros so file size is realistic-ish (~1 KB)
    return raw + b"\x00" * max(0, 1024 - len(raw))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gguf_bytes_factory():
    """Return the make_gguf_bytes helper for tests that want custom GGUFs."""
    return make_gguf_bytes


@pytest.fixture
def tmp_model_dir(tmp_path: Path) -> Path:
    """Tmp dir with two synthetic GGUF files: qwen35 27B + gpt2 0.5B."""
    (tmp_path / "qwen35-27b-Q5_K_M.gguf").write_bytes(make_gguf_bytes("qwen35"))
    (tmp_path / "gpt2-small.gguf").write_bytes(make_gguf_bytes("gpt2", file_type=15))
    return tmp_path


@pytest.fixture
def tmp_registry_path(tmp_path: Path) -> Path:
    """Path that points to a not-yet-existing registry YAML."""
    return tmp_path / "config" / "models.yaml"


@pytest.fixture
def fake_completed_process():
    """Build a fake subprocess.CompletedProcess with given stdout."""
    def _make(stdout: str = "", returncode: int = 0, stderr: str = ""):
        return SimpleNamespace(
            stdout=stdout, stderr=stderr, returncode=returncode, args=[]
        )
    return _make


@pytest.fixture
def mock_nvidia_smi(monkeypatch, fake_completed_process):
    """Replace subprocess.run for nvidia-smi calls. Default: 4090 / 24 GB."""
    state: dict[str, object] = {
        "vram_used_mb": 1500,
        "vram_total_mb": 24576,
        "pids": [],          # list[(pid, mb)]
        "should_fail": False,
        "should_timeout": False,
    }

    def fake_run(cmd, *args, **kwargs):
        import subprocess
        if not (isinstance(cmd, (list, tuple)) and cmd and "nvidia-smi" in cmd[0]):
            # Not a nvidia-smi call — fall through to original or fail.
            return fake_completed_process(stdout="", returncode=0)
        if state["should_timeout"]:
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout", 5))
        if state["should_fail"]:
            return fake_completed_process(stdout="", returncode=1, stderr="error")
        joined = " ".join(cmd)
        if "memory.used,memory.total" in joined:
            return fake_completed_process(
                stdout=f"{state['vram_used_mb']}, {state['vram_total_mb']}\n",
            )
        if "compute-apps" in joined:
            lines = "\n".join(f"{p}, {m}" for p, m in state["pids"])  # type: ignore[misc]
            return fake_completed_process(stdout=lines + "\n")
        return fake_completed_process(stdout="")

    monkeypatch.setattr("subprocess.run", fake_run)
    return state


@pytest.fixture
def no_real_subprocess(monkeypatch):
    """Autouse-style guard: assert no un-mocked Popen escapes test scope.

    Tests that legitimately need Popen should mock it themselves explicitly.
    """
    real_popen = None
    calls: list[object] = []

    def fake_popen(*args, **kwargs):
        calls.append((args, kwargs))
        proc = MagicMock()
        proc.pid = 99999
        proc.poll.return_value = None
        proc.wait.return_value = 0
        proc.communicate.return_value = (b"", b"")
        return proc

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    return calls


@pytest.fixture
def tmp_sqlite():
    """In-memory sqlite3 connection."""
    import sqlite3
    conn = sqlite3.connect(":memory:")
    yield conn
    conn.close()


def _is_marked(item: pytest.Item, name: str) -> bool:
    return item.get_closest_marker(name) is not None


def pytest_collection_modifyitems(config, items: Iterable[pytest.Item]) -> None:
    """Auto-skip GPU tests when CUDA toolkit absent."""
    skip_gpu = pytest.mark.skip(reason="GPU/heavy ML deps not installed")
    cuda_ok = False
    try:
        import torch  # noqa: WPS433
        cuda_ok = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        cuda_ok = False
    for item in items:
        if _is_marked(item, "gpu") and not cuda_ok:
            item.add_marker(skip_gpu)
