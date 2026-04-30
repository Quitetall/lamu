"""Runs inside the container — quantizes the heretic model to W4A16 compressed-tensors."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SOURCE = "llmfan46/Qwen3.6-27B-uncensored-heretic-v2"
OUTPUT = "/output"

print(f"Loading tokenizer from {SOURCE}...")
tokenizer = AutoTokenizer.from_pretrained(SOURCE, trust_remote_code=True)

print(f"Loading {SOURCE} in 4-bit (bitsandbytes NF4)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    SOURCE,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)

print(f"Saving quantized model to {OUTPUT}...")
model.save_pretrained(OUTPUT)
tokenizer.save_pretrained(OUTPUT)

import os, pathlib
size = sum(f.stat().st_size for f in pathlib.Path(OUTPUT).rglob("*") if f.is_file()) / 1e9
print(f"Done! Model saved: {OUTPUT} ({size:.1f} GB)")
