# Why vLLM Can't Fit Qwen3.6-27B on 24GB

## Root Cause: Unquantized Embeddings

vLLM's AWQ/GPTQ loaders keep embedding tables and LM head in FP16:
```
Token embedding:  248,320 × 5,120 × 2 bytes = 2.42 GiB
LM head:          248,320 × 5,120 × 2 bytes = 2.42 GiB  (or tied = 0)
```

AWQ 4-bit layers for 27B params: ~14 GiB
Total with embeddings: 19-21 GiB
Remaining for KV cache: 0-2 GiB → not enough for any useful context

## Tested Configurations (all failed)
| Config | Model VRAM | Result |
|--------|-----------|--------|
| AWQ INT4 | 21.55 GiB | 0 bytes for KV |
| AWQ INT4 + `--language-model-only` | 21.28 GiB | Still no room |
| bitsandbytes 4-bit | 19.75 GiB | 1.53 GiB short for KV at 32K |
| bitsandbytes + `--enforce-eager` | ~20 GiB | Still OOM |

## vLLM's Pre-allocation Problem
vLLM checks `available_vram >= utilization × total_vram` at startup.
With display server using 1.5-3 GiB, the check fails even at 0.85 utilization.
llama-cpp-python doesn't have this check — it allocates dynamically.

## The 248K Vocab Is the Bottleneck
Qwen3.6's vocabulary is 248,320 tokens (multilingual, large). Smaller-vocab models (LLaMA's 128K vocab) would leave more room. But this specific model's vocab is fixed.

## Quantization Tools Also Broken
- autoawq: deprecated, doesn't support qwen3_5
- auto-gptq: deprecated, won't build with torch 2.11
- llmcompressor: dep conflicts with every container image
- Containerized quantization: torchvision ABI mismatch in vLLM image
