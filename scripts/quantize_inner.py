"""Runs inside the container — quantizes the heretic model to GPTQ 4-bit."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

SOURCE = "llmfan46/Qwen3.6-27B-uncensored-heretic-v2"
OUTPUT = "/output"

print(f"Loading tokenizer from {SOURCE}...")
tokenizer = AutoTokenizer.from_pretrained(SOURCE, trust_remote_code=True)

print("Configuring GPTQ 4-bit quantization...")
gptq_config = GPTQConfig(
    bits=4,
    dataset="wikitext2",
    tokenizer=tokenizer,
    group_size=128,
    desc_act=False,
)

print(f"Loading and quantizing {SOURCE}...")
print("This loads the BF16 model, calibrates, and quantizes. ~30-45 min on 4090.")
model = AutoModelForCausalLM.from_pretrained(
    SOURCE,
    quantization_config=gptq_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)

print(f"Saving quantized model to {OUTPUT}...")
model.save_pretrained(OUTPUT)
tokenizer.save_pretrained(OUTPUT)

import os
size = sum(f.stat().st_size for f in __import__('pathlib').Path(OUTPUT).rglob("*") if f.is_file()) / 1e9
print(f"Done! Quantized model: {OUTPUT} ({size:.1f} GB)")
