#!/usr/bin/env python3
"""
Quantize the heretic uncensored model to AWQ INT4 for vLLM serving.

Downloads BF16 safetensors (~54GB), quantizes on GPU (~30-45 min),
saves AWQ model (~14GB), optionally deletes BF16 source.

Usage:
  python scripts/quantize-awq.py
  python scripts/quantize-awq.py --keep-source  # don't delete BF16 after
"""

import argparse
import shutil
from pathlib import Path

SOURCE_MODEL = "llmfan46/Qwen3.6-27B-uncensored-heretic-v2"
OUTPUT_DIR = Path.home() / "models" / "qwen3.6-27b-heretic-awq"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep-source", action="store_true", help="Don't delete BF16 after quantization")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    args = parser.parse_args()

    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    print(f"Loading tokenizer from {SOURCE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(SOURCE_MODEL, trust_remote_code=True)

    print(f"Loading model from {SOURCE_MODEL} (downloads ~54GB on first run)...")
    model = AutoAWQForCausalLM.from_pretrained(
        SOURCE_MODEL,
        trust_remote_code=True,
        safetensors=True,
    )

    quant_config = {
        "zero_point": True,
        "q_group_size": args.group_size,
        "w_bit": args.bits,
        "version": "GEMM",
    }

    print(f"Quantizing to AWQ INT{args.bits} (group_size={args.group_size})...")
    print("This takes ~30-45 minutes on a 4090.")
    model.quantize(tokenizer, quant_config=quant_config)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {OUTPUT_DIR}...")
    model.save_quantized(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    size_gb = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file()) / 1e9
    print(f"\nDone! AWQ model saved: {OUTPUT_DIR} ({size_gb:.1f} GB)")
    print(f"Serve with: just serve-vllm")

    if not args.keep_source:
        # Clean up HF cache of the BF16 model
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        for d in cache_dir.glob("models--llmfan46--Qwen3.6-27B-uncensored-heretic-v2*"):
            print(f"Cleaning cache: {d}")
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()
