"""Backend protocol — interface that all model backends implement."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from lamu.core.types import ModelEntry


class Backend(ABC):
    """Abstract base for model backends.

    Each backend manages one model process (or in-process inference).
    Lifecycle: load → generate/stream → unload.
    """

    @abstractmethod
    def load(self, entry: ModelEntry, port: int) -> int:
        """Load model onto GPU. Returns PID of the model process.

        Args:
            entry: Model configuration from registry
            port: HTTP port to serve on

        Returns:
            PID of the subprocess (or 0 for in-process backends)

        Raises:
            RuntimeError: If loading fails (OOM, binary not found, etc.)
        """
        ...

    @abstractmethod
    def unload(self) -> None:
        """Stop model process and free VRAM."""
        ...

    @abstractmethod
    def is_healthy(self) -> bool:
        """Check if model is responding (health endpoint)."""
        ...

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 16384,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> str:
        """Generate a completion (non-streaming).

        Returns raw response text (including think blocks if present).
        Caller is responsible for reasoning extraction.
        """
        ...

    @abstractmethod
    def stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 16384,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        """Generate a completion, yielding tokens as they arrive.

        Yields raw tokens (including think blocks).
        Caller wraps with ReasoningExtractor.stream_filter().
        """
        ...

    @abstractmethod
    def get_vram_mb(self) -> int:
        """Query actual VRAM usage of this backend's process."""
        ...

    @property
    @abstractmethod
    def port(self) -> int:
        """HTTP port this backend serves on."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the loaded model."""
        ...
