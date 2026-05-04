# DFlash Speculative Decoding

## Results (v1.0, RTX 4090)

| draft-max | Speed | Acceptance | Notes |
|-----------|-------|-----------|-------|
| 4 | 61.5 t/s | 92.0% | Safe, high acceptance |
| 6 | 58.0 t/s | 71.8% | |
| **8** | **82.0 t/s** | **77.9%** | **Sweet spot** |
| 10 | 71.1 t/s | 60.6% | Diminishing returns |

Baseline without DFlash: 49.5 t/s (ngram-mod warm), 9.8 t/s (cold).

## How it works

DFlash is block-diffusion drafting — a 5-layer non-causal denoising draft model conditioned on target hidden states. Accepts ~4-8 tokens per step vs ~1-3 for chain EAGLE. The draft model is tiny (974 MB Q4_K_M) and runs alongside the target on same GPU.

Paper: [DFlash: Block Diffusion for Flash Speculative Decoding (arXiv:2602.06036)](https://arxiv.org/abs/2602.06036)

## Setup

### Prerequisites
- llama.cpp built from PR #22105 (DFlash branch)
- Target: Qwen3.6-27B-uncensored-heretic-v2-Q4_K_M.gguf
- Draft: z-lab/Qwen3.6-27B-DFlash converted to GGUF

### Draft model conversion
PR branch's `convert_hf_to_gguf.py` doesn't recognize qwen3.5/3.6 tokenizer. Patched by adding fallback in `get_vocab_base_pre()`:
```python
# In convert_hf_to_gguf.py, before the NotImplementedError raise:
if "qwen" in str(getattr(tokenizer, 'name_or_path', '')).lower():
    res = "qwen35"
```

Then:
```bash
python convert_hf_to_gguf.py ~/models/qwen3.6-27b-dflash-draft/ \
  --outtype f16 --target-model-dir ~/models/qwen3.5-tokenizer/ \
  --outfile dflash-3.6-f16.gguf
llama-quantize dflash-3.6-f16.gguf dflash-3.6-q4km.gguf Q4_K_M
```

### Running
```bash
llama-speculative-simple \
  -m Qwen3.6-27B-Q4_K_M.gguf \
  -md dflash-3.6-q4km.gguf \
  --dflash --draft-max 8 \
  -c 4096 -cd 512 \
  --temp 0 --top-k 1 \
  -ngl 999 -ngld 99 -fa on \
  -ctk q4_0 -ctv q4_0 \
  -ctkd q8_0 -ctvd q8_0 -t 8
```

## Limitations
- Only `llama-speculative-simple` works, not `llama-server` (PR pending)
- Context limited by combined VRAM (target + draft + KV caches)
- Draft model SWA layers (Qwen3.6 matched draft) not supported in lucebox binary
- gcc-14 required for CUDA compilation

## vs Other Approaches

| Method | Speed | Overhead | Notes |
|--------|-------|----------|-------|
| DFlash (draft-max=8) | 82 t/s | 974 MB draft | Best overall |
| ngram-mod (warm) | 49.5 t/s | 0 MB | Free, no draft needed |
| ngram-mod (cold) | 9.8 t/s | 0 MB | First request penalty |
| EAGLE v3 | 11.9 t/s | 1 GB head | 25% acceptance, not viable |
| 0.8B megakernel | 494 t/s | 2.7 GB | Different model, simple tasks only |

## VRAM Budget
- Target Q4_K_M: ~15 GB
- Draft Q4_K_M: ~1 GB
- KV cache (q4_0, 4K ctx): ~0.5 GB
- Total: ~17 GB of 24 GB
- Headroom for larger context or F16 draft
