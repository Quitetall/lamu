"""Configuration constants and paths."""
from __future__ import annotations

from pathlib import Path


# Paths
LAMU_ROOT: Path = Path.home() / "local-llm"
MODELS_DIR: Path = Path.home() / "models"
REGISTRY_PATH: Path = LAMU_ROOT / "config" / "models.yaml"
LLAMA_BIN: Path = Path.home() / "llama.cpp" / "build" / "bin" / "llama-server"

# Ports
PORT_MAIN: int = 8020       # primary 27B model
PORT_SIDECAR: int = 8001    # fast tier (4B or megakernel)
PORT_DFLASH: int = 8000     # DFlash lucebox

# VRAM
VRAM_RESERVED_MB: int = 1500  # reserved for CUDA overhead + display

# Defaults
DEFAULT_MAX_TOKENS: int = 16384
DEFAULT_TEMPERATURE: float = 0.7
DEFAULT_CTX_SIZE: int = 131072
