"""Model registry — auto-discovers models on disk, writes/reads YAML config."""
from __future__ import annotations

import struct
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import yaml

from lamu.core.types import (
    BackendType,
    Capability,
    ModelEntry,
    ModelFormat,
    ReasoningMarker,
    SpeculativeConfig,
)

# VRAM estimation heuristics (bytes per parameter at various quant levels)
_QUANT_BPW: dict[str, float] = {
    "F32": 32.0, "F16": 16.0, "BF16": 16.0,
    "Q8_0": 8.5, "Q6_K": 6.6, "Q5_K_M": 5.7, "Q5_K_S": 5.5,
    "Q4_K_M": 4.9, "Q4_K_S": 4.6, "Q4_0": 4.5,
    "Q3_K_M": 3.9, "Q3_K_S": 3.7, "Q2_K": 3.0,
    "IQ4_NL": 4.5, "IQ4_XS": 4.3, "IQ3_XXS": 3.1,
}

_GGUF_MAGIC = b"GGUF"

# Known architecture → capabilities mapping
_ARCH_CAPABILITIES: dict[str, tuple[Capability, ...]] = {
    "qwen35": (Capability.CHAT, Capability.CODE, Capability.REASONING),
    "qwen3": (Capability.CHAT, Capability.CODE),
    "gpt2": (Capability.CHAT,),
    "phi3": (Capability.CHAT, Capability.CODE, Capability.REASONING),
    "llama": (Capability.CHAT, Capability.CODE),
    "gemma": (Capability.CHAT,),
    "dflash": (Capability.CHAT, Capability.CODE, Capability.REASONING),
}

# Known architecture → reasoning marker
_ARCH_REASONING: dict[str, ReasoningMarker] = {
    "qwen35": ReasoningMarker(open_tag="<think>", close_tag="</think>", family="qwen35"),
    "qwen3": ReasoningMarker(open_tag="<think>", close_tag="</think>", family="qwen3"),
}


def _parse_gguf_metadata(path: Path) -> dict[str, object]:
    """Read key GGUF metadata without loading full model."""
    meta: dict[str, object] = {}
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != _GGUF_MAGIC:
                return meta

            version = struct.unpack("<I", f.read(4))[0]
            n_tensors = struct.unpack("<Q", f.read(8))[0]
            n_kv = struct.unpack("<Q", f.read(8))[0]

            meta["version"] = version
            meta["n_tensors"] = n_tensors
            meta["n_kv"] = n_kv
            meta["file_size_mb"] = int(path.stat().st_size / (1024 * 1024))

            # Parse KV pairs to extract architecture and params
            for _ in range(min(n_kv, 100)):  # cap to avoid huge reads
                # Read key
                key_len = struct.unpack("<Q", f.read(8))[0]
                key = f.read(key_len).decode("utf-8", errors="replace")

                # Read value type
                val_type = struct.unpack("<I", f.read(4))[0]

                # Parse value based on type
                if val_type == 8:  # string
                    str_len = struct.unpack("<Q", f.read(8))[0]
                    val = f.read(str_len).decode("utf-8", errors="replace").strip("\x00")
                    meta[key] = val
                elif val_type == 4:  # uint32
                    meta[key] = struct.unpack("<I", f.read(4))[0]
                elif val_type == 5:  # int32
                    meta[key] = struct.unpack("<i", f.read(4))[0]
                elif val_type == 6:  # float32
                    meta[key] = struct.unpack("<f", f.read(4))[0]
                elif val_type == 10:  # uint64
                    meta[key] = struct.unpack("<Q", f.read(8))[0]
                elif val_type == 7:  # bool
                    meta[key] = struct.unpack("<?", f.read(1))[0]
                elif val_type == 9:  # array
                    arr_type = struct.unpack("<I", f.read(4))[0]
                    arr_len = struct.unpack("<Q", f.read(8))[0]
                    # Skip array data (complex to parse generically)
                    if arr_type == 8:  # string array
                        for _ in range(min(arr_len, 5)):
                            sl = struct.unpack("<Q", f.read(8))[0]
                            f.read(sl)
                    elif arr_type in (4, 5):  # uint32/int32 array
                        f.read(arr_len * 4)
                    elif arr_type == 6:  # float32 array
                        f.read(arr_len * 4)
                    else:
                        break  # unknown array type, stop parsing
                else:
                    break  # unknown type, stop parsing

    except (OSError, struct.error):
        pass

    return meta


def _estimate_vram_mb(file_size_mb: int, quant: str) -> int:
    """Estimate VRAM usage. Rough: file_size + 10% overhead for KV/compute."""
    return int(file_size_mb * 1.1)


def _detect_quant(meta: dict[str, object], filename: str) -> str:
    """Detect quantization type from metadata or filename."""
    # Try GGUF metadata
    file_type = meta.get("general.file_type")
    if isinstance(file_type, int):
        _FILE_TYPE_NAMES = {
            0: "F32", 1: "F16", 2: "Q4_0", 7: "Q8_0",
            15: "Q4_K_M", 14: "Q4_K_S", 16: "Q5_K_S", 17: "Q5_K_M",
            18: "Q6_K", 19: "Q2_K", 20: "Q3_K_S", 21: "Q3_K_M",
        }
        if file_type in _FILE_TYPE_NAMES:
            return _FILE_TYPE_NAMES[file_type]

    # Fallback: parse from filename
    fn_upper = filename.upper()
    for quant_name in sorted(_QUANT_BPW.keys(), key=len, reverse=True):
        if quant_name.replace("_", "-") in fn_upper or quant_name in fn_upper:
            return quant_name

    return "unknown"


def _estimate_params_b(meta: dict[str, object]) -> float:
    """Estimate parameter count in billions from tensor count + dims."""
    n_tensors = meta.get("n_tensors", 0)
    if isinstance(n_tensors, int) and n_tensors > 0:
        # Very rough: typical transformer has ~12 tensors per layer
        # Each layer ~= hidden^2 * 12 params. For 5120 hidden, 64 layers = 27B
        file_size = meta.get("file_size_mb", 0)
        if isinstance(file_size, int) and file_size > 0:
            # Rough: Q4_K_M ≈ 0.6 bytes per param → params ≈ filesize_bytes / 0.6
            return round(file_size * 1024 * 1024 / 0.6 / 1e9, 1)
    return 0.0


def scan_directory(
    models_dir: Path,
    existing: Optional[dict[str, ModelEntry]] = None,
) -> list[ModelEntry]:
    """Scan a directory recursively for model files. Returns discovered models."""
    discovered: list[ModelEntry] = []

    # Find all GGUF files
    for gguf_path in sorted(models_dir.rglob("*.gguf")):
        # Skip draft models (they're referenced via speculative config)
        if "dflash" in gguf_path.name.lower() and "draft" in str(gguf_path).lower():
            continue

        meta = _parse_gguf_metadata(gguf_path)
        arch_raw = meta.get("general.architecture", "")
        arch = str(arch_raw).strip("\x00").lower() if arch_raw else "unknown"
        quant = _detect_quant(meta, gguf_path.name)
        file_size_mb = int(gguf_path.stat().st_size / (1024 * 1024))
        vram_mb = _estimate_vram_mb(file_size_mb, quant)
        params_b = _estimate_params_b(meta)

        # Determine context max from metadata
        ctx_key = f"{arch}.block_count"
        block_count = meta.get(ctx_key, 0)
        # Default context by architecture
        context_max = 131072  # safe default for modern models

        # Name from filename
        name = gguf_path.stem.lower().replace(" ", "-")

        # Capabilities from architecture
        capabilities = _ARCH_CAPABILITIES.get(arch, (Capability.CHAT,))

        # Add long_context capability if ctx > 64K
        if context_max > 65536:
            capabilities = (*capabilities, Capability.LONG_CONTEXT)

        # Reasoning marker
        reasoning_marker = _ARCH_REASONING.get(arch)

        # Backend selection
        backend = BackendType.LLAMACPP

        entry = ModelEntry(
            name=name,
            path=gguf_path,
            format=ModelFormat.GGUF,
            backend=backend,
            arch=arch,
            params_b=params_b,
            quant=quant,
            vram_mb=vram_mb,
            context_max=context_max,
            capabilities=capabilities,
            reasoning_marker=reasoning_marker,
        )
        discovered.append(entry)

    return discovered


def write_registry(models: Sequence[ModelEntry], output: Path) -> None:
    """Write model registry to YAML."""
    data: dict[str, object] = {"models": {}}
    models_dict = data["models"]
    assert isinstance(models_dict, dict)

    for m in models:
        entry_dict: dict[str, object] = {
            "path": str(m.path),
            "format": m.format.value,
            "backend": m.backend.value,
            "arch": m.arch,
            "params_b": m.params_b,
            "quant": m.quant,
            "vram_mb": m.vram_mb,
            "context_max": m.context_max,
            "capabilities": [c.value for c in m.capabilities],
        }
        if m.reasoning_marker:
            entry_dict["reasoning_marker"] = {
                "open_tag": m.reasoning_marker.open_tag,
                "close_tag": m.reasoning_marker.close_tag,
                "family": m.reasoning_marker.family,
            }
        if m.speculative:
            entry_dict["speculative"] = {
                "draft_path": str(m.speculative.draft_path),
                "method": m.speculative.method,
                "draft_max": m.speculative.draft_max,
            }
        if m.pinned:
            entry_dict["pinned"] = True

        models_dict[m.name] = entry_dict

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_registry(path: Path) -> list[ModelEntry]:
    """Load model registry from YAML."""
    if not path.exists():
        return []

    with open(path) as f:
        data = yaml.safe_load(f)

    if not data or "models" not in data:
        return []

    entries: list[ModelEntry] = []
    for name, cfg in data["models"].items():
        reasoning_marker = None
        if "reasoning_marker" in cfg:
            rm = cfg["reasoning_marker"]
            reasoning_marker = ReasoningMarker(
                open_tag=rm["open_tag"],
                close_tag=rm["close_tag"],
                family=rm["family"],
            )

        speculative = None
        if "speculative" in cfg:
            sp = cfg["speculative"]
            speculative = SpeculativeConfig(
                draft_path=Path(sp["draft_path"]),
                method=sp["method"],
                draft_max=sp.get("draft_max", 8),
            )

        entries.append(ModelEntry(
            name=name,
            path=Path(cfg["path"]),
            format=ModelFormat(cfg["format"]),
            backend=BackendType(cfg["backend"]),
            arch=cfg["arch"],
            params_b=cfg["params_b"],
            quant=cfg["quant"],
            vram_mb=cfg["vram_mb"],
            context_max=cfg["context_max"],
            capabilities=tuple(Capability(c) for c in cfg["capabilities"]),
            reasoning_marker=reasoning_marker,
            speculative=speculative,
            pinned=cfg.get("pinned", False),
        ))

    return entries
