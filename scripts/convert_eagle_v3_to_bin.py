#!/usr/bin/env python3
"""Convert EAGLE v3 PyTorch weights to binary format for C++ integration.

Binary format:
  Header: magic(4) version(4) hidden_size(4) n_layers(4) ffn_mult(4) pad(4)
  Weights (all float16 for memory efficiency):
    fuse.weight [H*2, H]
    fuse.bias [H]
    fuse_norm.weight [H]
    fuse_norm.bias [H]
    For each layer:
      norm.weight [H]
      norm.bias [H]
      up.weight [H*ffn_mult, H]
      up.bias [H*ffn_mult]
      down.weight [H, H*ffn_mult]
      down.bias [H]
    out_norm.weight [H]
    out_norm.bias [H]
"""

import struct, json, sys, torch
import numpy as np
from pathlib import Path

EAGLE_DIR = Path.home() / "models" / "qwen3.6-27b-heretic-eagle"
V3_DIR = EAGLE_DIR / "eagle_head_v3"
OUTPUT = V3_DIR / "eagle_v3.bin"

MAGIC = 0x45334743  # "E3GC" - Eagle v3 ggml compatible

def main():
    config = json.loads((V3_DIR / "config.json").read_text())
    H = config["hidden_size"]
    N_LAYERS = config["n_layers"]
    FFN_MULT = config["ffn_mult"]

    print(f"Loading v3 weights (H={H}, layers={N_LAYERS}, ffn_mult={FFN_MULT})")

    # Load best checkpoint
    best_path = V3_DIR / "eagle_v3_best.pt"
    if not best_path.exists():
        best_path = V3_DIR / "eagle_v3.pt"
    state = torch.load(best_path, map_location="cpu", weights_only=True)

    print(f"Keys: {list(state.keys())}")

    with open(OUTPUT, "wb") as f:
        # Header
        f.write(struct.pack("<I", MAGIC))
        f.write(struct.pack("<I", 3))        # version 3
        f.write(struct.pack("<I", H))
        f.write(struct.pack("<I", N_LAYERS))
        f.write(struct.pack("<I", FFN_MULT))
        f.write(struct.pack("<I", 0))        # padding

        total_bytes = 24  # header

        def write_tensor(name, expected_shape=None):
            nonlocal total_bytes
            t = state[name].float().numpy()
            if expected_shape:
                assert t.shape == expected_shape, f"{name}: got {t.shape}, expected {expected_shape}"
            # Write as float32 to avoid CUDA mixed-type issues
            t32 = t.astype(np.float32)
            f.write(t32.tobytes())
            total_bytes += t32.nbytes
            print(f"  {name}: {t.shape} → {t32.nbytes/1024:.1f} KB")

        # Fuse layer
        write_tensor("fuse.weight", (H, H * 2))
        write_tensor("fuse.bias", (H,))
        write_tensor("fuse_norm.weight", (H,))
        write_tensor("fuse_norm.bias", (H,))

        # Residual MLP layers
        for l in range(N_LAYERS):
            write_tensor(f"layers.{l}.norm.weight", (H,))
            write_tensor(f"layers.{l}.norm.bias", (H,))
            write_tensor(f"layers.{l}.up.weight", (H * FFN_MULT, H))
            write_tensor(f"layers.{l}.up.bias", (H * FFN_MULT,))
            write_tensor(f"layers.{l}.down.weight", (H, H * FFN_MULT))
            write_tensor(f"layers.{l}.down.bias", (H,))

        # Output norm
        write_tensor("out_norm.weight", (H,))
        write_tensor("out_norm.bias", (H,))

    print(f"\nTotal: {total_bytes / 1024 / 1024:.1f} MB → {OUTPUT}")


if __name__ == "__main__":
    main()
