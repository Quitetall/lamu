"""Core types for LAMU — strict dataclasses, enums, no **kwargs."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Sequence


class Capability(Enum):
    """Stable capability vocabulary. Resist expanding without clear use case."""
    CHAT = "chat"
    CODE = "code"
    REASONING = "reasoning"
    ROUTING = "routing"
    VISION = "vision"
    LONG_CONTEXT = "long_context"


class ModelFormat(Enum):
    GGUF = "gguf"
    SAFETENSORS = "safetensors"
    ONNX = "onnx"
    CUSTOM = "custom"


class BackendType(Enum):
    LLAMACPP = "llama_cpp"
    MEGAKERNEL = "megakernel"
    DFLASH = "dflash"
    DFLASH_LUCEBOX = "dflash_lucebox"


class ModelState(Enum):
    UNLOADED = auto()
    LOADING = auto()
    LOADED = auto()
    ERROR = auto()


@dataclass(frozen=True)
class ReasoningMarker:
    """Describes how a model family marks reasoning/thinking content."""
    open_tag: str       # "<think>"
    close_tag: str      # "</think>"
    family: str         # "qwen35", "deepseek", "o1"


@dataclass(frozen=True)
class SpeculativeConfig:
    """Speculative decoding configuration for a model."""
    draft_path: Path
    method: str         # "dflash", "eagle", "ngram-mod"
    draft_max: int = 8


@dataclass(frozen=True)
class ModelEntry:
    """A discovered model on disk. Immutable after scan."""
    name: str
    path: Path
    format: ModelFormat
    backend: BackendType
    arch: str                                   # "qwen35", "gpt2", "phi4"
    params_b: float                             # 27.0, 0.8, 1.5
    quant: str                                  # "Q4_K_M", "BF16", "F16"
    vram_mb: int                                # estimated VRAM when loaded
    context_max: int                            # max supported context length
    capabilities: tuple[Capability, ...]        # immutable sequence
    reasoning_marker: Optional[ReasoningMarker] = None
    speculative: Optional[SpeculativeConfig] = None
    pinned: bool = False                        # never auto-evict


@dataclass
class LoadedModel:
    """Runtime state for a currently loaded model."""
    entry: ModelEntry
    state: ModelState
    pid: Optional[int]          # subprocess PID if applicable
    port: int                   # HTTP port this model serves on
    vram_actual_mb: int         # actual VRAM usage (from nvidia-smi)
    last_used_ts: float         # monotonic timestamp of last request


@dataclass(frozen=True)
class RouteDecision:
    """Result of the router's model selection. Used by plan_query dry-run."""
    model_name: str
    reason: str
    loaded: bool
    would_evict: tuple[str, ...] = ()


@dataclass(frozen=True)
class StreamChunk:
    """A chunk of streaming output with type annotation."""
    type: str   # "reasoning" | "content"
    text: str


@dataclass(frozen=True)
class QueryStats:
    """Performance stats for a query (inspired by abdimoallim/llm)."""
    latency_ms: float               # total wall time
    time_to_first_token_ms: float   # streaming: ms until first token
    tokens_generated: int
    tokens_per_second: float
    prompt_tokens: int
    retries: int = 0
    stream_chunks: int = 0


@dataclass(frozen=True)
class QueryResult:
    """Result from a model query."""
    content: str
    reasoning: Optional[str]
    model_used: str
    stats: QueryStats
    finish_reason: str = "stop"     # "stop" | "length" | "error"


@dataclass(frozen=True)
class VramBudget:
    """Snapshot of VRAM allocation."""
    total_mb: int
    used_mb: int
    free_mb: int
    loaded_models: tuple[tuple[str, int], ...]  # (name, vram_mb) pairs
    available_mb: int                           # free - reserved overhead
