# 262K Context on Single RTX 4090

## The Problem
Qwen3.6-27B with Q4_K_M GGUF is 16GB. KV cache for 262K tokens in FP16 would need 16GB more — doesn't fit in 24GB.

## The Solution
Three flags that make it work:

1. **`flash_attn=True`** — Required for quantized V cache. Without this, llama.cpp refuses Q4 V cache.

2. **`logits_all=False`** — llama-cpp-python server defaults `logits_all=True`, which allocates `n_ctx × vocab_size × 4 bytes` for the scores buffer. With 262K context × 248K vocab = **242 GiB of RAM**. Setting False uses `n_batch × vocab_size` instead (~500KB).

3. **`type_k=2, type_v=2`** (Q4_0 KV cache) — Compresses KV from 64KB to 16KB per token. Qwen3.6's DeltaNet hybrid only needs KV for 16/64 layers (`full_attention_interval: 4`), so the actual KV is even smaller.

## VRAM Breakdown
```
Model (GGUF Q4_K_M):      ~16 GiB
KV cache (Q4_0, 16 layers): ~4 GiB
Compute buffers:            ~1.4 GiB
Total:                     ~22 GiB on 24 GiB 4090
```

## Quality Tradeoff
Q4_0 KV has lower precision than Q8_0. For better quality at reduced context:
- Q5_K_S + Q8_0 KV → 108K context (best quality)
- Q4_K_M + Q8_0 KV → 173K context
- Q4_K_M + Q4_0 KV → 262K context (max context)

## Key Insight
The DeltaNet layers (48/64) use O(1) recurrent state — no KV cache needed. Only the 16 standard attention layers need KV. This is encoded in the GGUF metadata as `qwen35.full_attention_interval: 4`.
