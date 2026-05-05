"""VRAM Budget Scheduler — bin-packing for GPU model management."""
from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

from lamu.core.errors import GpuUnavailableError
from lamu.core.types import LoadedModel, ModelEntry, ModelState, VramBudget


_log = logging.getLogger(__name__)


# Reserve 1.5 GB for CUDA overhead, display, compute buffers
_VRAM_RESERVED_MB: int = 1500

# Module-level flag: once set, every GPU operation refuses to silently fall back.
_gpu_unavailable_reason: Optional[str] = None


def _mark_gpu_unavailable(reason: str) -> None:
    global _gpu_unavailable_reason
    _gpu_unavailable_reason = reason
    _log.warning("gpu_unavailable: %s", reason)


def gpu_available() -> bool:
    """True if nvidia-smi is reachable. Refreshed by every _query_vram call."""
    return _gpu_unavailable_reason is None


def _query_vram() -> tuple[int, int]:
    """Query GPU VRAM via nvidia-smi. Returns (used_mb, total_mb).

    Marks the GPU unavailable on subprocess/parse failure but does NOT raise
    here — `(0, 0)` is the long-standing contract for callers like `budget()`
    that need to keep returning a snapshot. To force a hard error use
    `require_gpu()`.
    """
    global _gpu_unavailable_reason
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            _mark_gpu_unavailable(
                f"nvidia-smi exit={result.returncode}: {result.stderr.strip()}"
            )
            return (0, 0)
        parts = result.stdout.strip().split(",")
        used = int(parts[0].strip())
        total = int(parts[1].strip())
        # Successful query — clear the unavailable flag.
        _gpu_unavailable_reason = None
        return (used, total)
    except (subprocess.TimeoutExpired, ValueError, IndexError, FileNotFoundError) as exc:
        _mark_gpu_unavailable(f"{type(exc).__name__}: {exc}")
        return (0, 0)


def require_gpu() -> None:
    """Raise GpuUnavailableError if the GPU is in unavailable state.

    Call this at the gate of any GPU-touching operation (model load,
    eviction, etc.) to make silent CPU fallback impossible.
    """
    if _gpu_unavailable_reason is not None:
        raise GpuUnavailableError(_gpu_unavailable_reason)


def _query_gpu_pids() -> list[tuple[int, int]]:
    """Query GPU processes. Returns [(pid, used_mb), ...]."""
    global _gpu_unavailable_reason
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            _mark_gpu_unavailable(
                f"nvidia-smi(pids) exit={result.returncode}"
            )
            return []
        pids: list[tuple[int, int]] = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split(",")
            pids.append((int(parts[0].strip()), int(parts[1].strip())))
        return pids
    except (subprocess.TimeoutExpired, ValueError, IndexError, FileNotFoundError) as exc:
        _mark_gpu_unavailable(f"{type(exc).__name__}: {exc}")
        return []


class VramScheduler:
    """Budget-aware VRAM scheduler with bin-packing and LRU eviction."""

    def __init__(self, reserved_mb: int = _VRAM_RESERVED_MB) -> None:
        self._reserved_mb = reserved_mb
        self._loaded: dict[str, LoadedModel] = {}
        self._total_mb: int = 0
        self._refresh_total()

    def _refresh_total(self) -> None:
        """Query total VRAM once (doesn't change)."""
        _, total = _query_vram()
        self._total_mb = total

    @property
    def total_mb(self) -> int:
        return self._total_mb

    @property
    def available_mb(self) -> int:
        """VRAM available for new models (total - loaded - reserved)."""
        used_by_models = sum(m.vram_actual_mb for m in self._loaded.values())
        return self._total_mb - used_by_models - self._reserved_mb

    def budget(self) -> VramBudget:
        """Snapshot current VRAM budget."""
        used_mb, total_mb = _query_vram()
        loaded_pairs = tuple(
            (name, m.vram_actual_mb)
            for name, m in self._loaded.items()
        )
        return VramBudget(
            total_mb=total_mb,
            used_mb=used_mb,
            free_mb=total_mb - used_mb,
            loaded_models=loaded_pairs,
            available_mb=self.available_mb,
        )

    def register_loaded(
        self,
        entry: ModelEntry,
        pid: Optional[int],
        port: int,
        vram_actual_mb: int,
    ) -> LoadedModel:
        """Register a model that's already loaded (e.g. from startup)."""
        model = LoadedModel(
            entry=entry,
            state=ModelState.LOADED,
            pid=pid,
            port=port,
            vram_actual_mb=vram_actual_mb,
            last_used_ts=time.monotonic(),
        )
        self._loaded[entry.name] = model
        return model

    def mark_used(self, name: str) -> None:
        """Update last-used timestamp (for LRU eviction)."""
        if name in self._loaded:
            self._loaded[name].last_used_ts = time.monotonic()

    def is_loaded(self, name: str) -> bool:
        return name in self._loaded and self._loaded[name].state == ModelState.LOADED

    def get_loaded(self, name: str) -> Optional[LoadedModel]:
        return self._loaded.get(name)

    def loaded_models(self) -> list[LoadedModel]:
        return list(self._loaded.values())

    def can_fit(self, entry: ModelEntry) -> bool:
        """Can this model fit in available VRAM without evicting anything?"""
        return entry.vram_mb <= self.available_mb

    def plan_eviction(self, needed_mb: int) -> list[str]:
        """
        Determine which models to evict to free `needed_mb` VRAM.
        Returns model names to evict (LRU order, skips pinned).
        Returns empty list if impossible.
        """
        if needed_mb <= 0:
            return []

        # Sort by last_used (oldest first), skip pinned
        evictable = [
            (name, m)
            for name, m in self._loaded.items()
            if not m.entry.pinned and m.state == ModelState.LOADED
        ]
        evictable.sort(key=lambda x: x[1].last_used_ts)

        to_evict: list[str] = []
        freed: int = 0
        for name, m in evictable:
            to_evict.append(name)
            freed += m.vram_actual_mb
            if freed >= needed_mb:
                return to_evict

        # Can't free enough even evicting everything
        return []

    def plan_load(self, entry: ModelEntry) -> tuple[bool, list[str]]:
        """
        Plan loading a model. Returns (can_load, models_to_evict).
        If can_load is False, it's impossible even with full eviction.
        """
        if self.is_loaded(entry.name):
            return (True, [])

        if self.can_fit(entry):
            return (True, [])

        # Need to evict
        deficit = entry.vram_mb - self.available_mb
        to_evict = self.plan_eviction(deficit)
        if not to_evict:
            return (False, [])

        return (True, to_evict)

    def mark_unloaded(self, name: str) -> None:
        """Remove a model from the loaded set."""
        if name in self._loaded:
            del self._loaded[name]

    def mark_loading(self, entry: ModelEntry) -> None:
        """Mark model as loading (VRAM reserved but not confirmed)."""
        model = LoadedModel(
            entry=entry,
            state=ModelState.LOADING,
            pid=None,
            port=0,
            vram_actual_mb=entry.vram_mb,  # use estimate until confirmed
            last_used_ts=time.monotonic(),
        )
        self._loaded[entry.name] = model

    def confirm_loaded(
        self, name: str, pid: int, port: int, vram_actual_mb: int
    ) -> None:
        """Confirm model finished loading with actual VRAM usage."""
        if name in self._loaded:
            m = self._loaded[name]
            m.state = ModelState.LOADED
            m.pid = pid
            m.port = port
            m.vram_actual_mb = vram_actual_mb
            m.last_used_ts = time.monotonic()
