"""Typed exception hierarchy for LAMU.

Every place the codebase used to write `except Exception: pass` should now
either log + raise one of these, or `record_error()` on the relevant
BackendHealth. Each class is narrow on purpose — generic `LamuError` exists
only as a hierarchy root; never raise it directly.

Hierarchy:
    LamuError
        ├── ConfigError              — invalid config / registry
        │   └── RegistryParseWarning — malformed GGUF (warning, not fatal)
        ├── BackendError             — backend (model server) failed
        │   ├── BackendUnavailable   — health check failed / no healthy backend
        │   └── BackendTimeout       — request exceeded deadline
        ├── GpuUnavailableError      — nvidia-smi unreachable / no CUDA
        ├── ReasoningOverflow        — think-block buffer exceeded cap
        ├── DataLayerError           — sqlite / storage failure
        ├── MissingDependency        — heavy optional dep absent
        ├── SwarmStepError           — worker / planner / critic step crashed
        └── ToolExecutionError       — agent tool (python_repl etc.) failed
"""
from __future__ import annotations


class LamuError(Exception):
    """Root of every LAMU exception. Never raise directly — use a subclass."""


# ── Config / registry ───────────────────────────────────────────────────────

class ConfigError(LamuError):
    """Invalid or missing configuration."""


class RegistryParseWarning(Warning):
    """Malformed GGUF — emitted via warnings.warn during scan."""


# ── Backends ────────────────────────────────────────────────────────────────

class BackendError(LamuError):
    """Backend (model server) failed in a recoverable-but-loud way."""


class BackendUnavailable(BackendError):
    """No healthy backend can serve the request."""


class BackendTimeout(BackendError):
    """Backend did not respond within the configured timeout."""


# ── Hardware / runtime ──────────────────────────────────────────────────────

class GpuUnavailableError(LamuError):
    """nvidia-smi missing, returned non-zero, or timed out.

    Raise this any time GPU operations are attempted while the scheduler
    is in `gpu_available=False` state. Makes silent CPU-fallback impossible.
    """


class ReasoningOverflow(LamuError):
    """Think-block streaming buffer exceeded the configured cap (64 KB)."""


# ── Storage ─────────────────────────────────────────────────────────────────

class DataLayerError(LamuError):
    """SQLite / persistence layer error.

    All raw `sqlite3.Error` instances should be wrapped into this so callers
    only need a single `except` to catch storage failures.
    """


# ── Misc ────────────────────────────────────────────────────────────────────

class MissingDependency(LamuError):
    """A heavy optional dependency (torch, transformers, peft) is absent."""


class SwarmStepError(LamuError):
    """Swarm worker/planner/critic step failed in a non-recoverable way."""


class ToolExecutionError(LamuError):
    """An agent tool (python_repl, search, etc.) failed to execute."""
