"""Convert PyTorch EAGLE head to binary format for llama.cpp."""
import struct
import torch
import numpy as np
from pathlib import Path

EAGLE_DIR = Path.home() / "models" / "qwen3.6-27b-heretic-eagle" / "eagle_head"
OUTPUT = EAGLE_DIR / "eagle_head.bin"
MAGIC = 0x454147CE

state = torch.load(EAGLE_DIR / "eagle_head_best.pt", map_location="cpu", weights_only=True)

# Map PyTorch names to our binary format order
H, D, V = 5120, 1024, 248320
N_LAYERS, N_HEADS, FFN = 2, 8, 4096

with open(OUTPUT, "wb") as f:
    # Header
    f.write(struct.pack("<IIIIIIII", MAGIC, 1, H, D, V, N_LAYERS, N_HEADS, FFN))

    def write_tensor(name):
        t = state[name].float().numpy()
        f.write(t.tobytes())
        print(f"  {name}: {t.shape}")

    # Down projection
    write_tensor("down.weight")
    write_tensor("down.bias")

    # Input norm
    write_tensor("norm.weight")
    write_tensor("norm.bias")

    # Transformer layers
    for l in range(N_LAYERS):
        write_tensor(f"enc.layers.{l}.self_attn.in_proj_weight")
        write_tensor(f"enc.layers.{l}.self_attn.in_proj_bias")
        write_tensor(f"enc.layers.{l}.self_attn.out_proj.weight")
        write_tensor(f"enc.layers.{l}.self_attn.out_proj.bias")
        write_tensor(f"enc.layers.{l}.linear1.weight")
        write_tensor(f"enc.layers.{l}.linear1.bias")
        write_tensor(f"enc.layers.{l}.linear2.weight")
        write_tensor(f"enc.layers.{l}.linear2.bias")
        write_tensor(f"enc.layers.{l}.norm1.weight")
        write_tensor(f"enc.layers.{l}.norm1.bias")
        write_tensor(f"enc.layers.{l}.norm2.weight")
        write_tensor(f"enc.layers.{l}.norm2.bias")

    # LM head
    write_tensor("lm.weight")

print(f"\nSaved to {OUTPUT} ({OUTPUT.stat().st_size / 1e6:.0f} MB)")
