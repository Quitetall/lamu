# DFlash Speculative Decoding

## Summary

Two DFlash implementations available, both working:

| Implementation | Draft | Speed | Acceptance | Server Mode |
|---------------|-------|-------|-----------|-------------|
| **llama.cpp PR #22105** | 3.6 matched GGUF (Q4_K_M) | **82 t/s** | 77.9% | Crashes 2nd req |
| **Lucebox binary** (our fix PR #89) | 3.5 mismatch safetensors | **71.2 t/s** | 20% per step, 3.2 tok/step | One-shot + daemon |

Both run on RTX 4090, 24 GB, without unified memory.

## llama.cpp DFlash (PR #22105)

Best overall speed. Uses matched Qwen3.6-27B-DFlash GGUF draft.

### Results (v1.0+, RTX 4090)

| draft-max | Speed | Acceptance |
|-----------|-------|-----------|
| 4 | 61.5 t/s | 92.0% |
| **8** | **82.0 t/s** | **77.9%** |
| 12 | 80.5 t/s | 62.8% |
| 16 | 66.9 t/s | 43.3% |

### F16 vs Q4_K_M draft

| Draft quant | Speed | Acceptance | VRAM |
|------------|-------|-----------|------|
| Q4_K_M | **77.6 t/s** | **77.9%** | 1.0 GB |
| F16 | 72.7 t/s | 76.4% | 3.5 GB |

**Q4_K_M wins.** Bandwidth-bound: smaller reads > marginal accuracy gain.

### Setup

```bash
# On dflash-pr branch of ~/llama.cpp
llama-speculative-simple \
  -m Qwen3.6-27B-Q4_K_M.gguf \
  -md dflash-3.6-q4km.gguf \
  --dflash --draft-max 8 \
  -c 4096 -cd 512 --temp 0 --top-k 1 \
  -ngl 999 -ngld 99 -fa on \
  -ctk q4_0 -ctv q4_0 -ctkd q8_0 -ctvd q8_0 -t 8
```

### Limitations
- **Server mode crashes on 2nd request** — PR bug, draft context state not reset
- Only `llama-speculative-simple` works (one-shot)
- Draft GGUF conversion required patching tokenizer hash in `convert_hf_to_gguf.py`

### DFlash + ngram-mod stacking
Code supports both simultaneously (speculative.cpp lines 1262 + 1271 both push to configs). Flags: `--dflash --spec-type ngram-mod`. Server mode crashes before we can benchmark this combo.

## Lucebox DFlash + DDTree (our fix)

Uses lucebox's custom C++ binary with DDTree verify. Our PR #89 fixed the `ggml_cpy` conv_input_cache crash.

### Results (RTX 4090, Qwen3.6 target + 3.5 draft mismatch)

| Budget | Speed | Tok/step | Acceptance |
|--------|-------|----------|-----------|
| 10 | 64.4 t/s | 3.20 | 20.0% |
| 14 | 65.0 t/s | 3.20 | 20.0% |
| 18 | 62.9 t/s | 3.46 | 21.6% |
| **22** | **67.4 t/s** | **3.76** | **23.5%** |

Lower than llama.cpp path because using mismatched 3.5 draft (3.6 draft has SWA layers lucebox can't handle yet).

### PFlash (Speculative Prefill)

PFlash reduces TTFT at long context (128K: 257s → 24.8s). Requires:
1. Qwen3-0.6B drafter GGUF (for token importance scoring)
2. BSA (Block-Sparse-Attention) compiled with `DDFLASH27B_ENABLE_BSA=ON`
3. Park/unpark VRAM dance (`dflash_24gb.py` wrapper partially implements this)

**Status:** Not tested yet. The conv_input_cache fix unblocks the decode path but PFlash needs separate testing. The 24GB VRAM sequencing (park target → compress → free drafter → unpark → generate) is documented but our wrapper needs end-to-end validation.

### DDTree

DDTree = Dynamic Decoding Tree. Verifies multiple draft branches in parallel. At budget=22 on 3090: ~8 tokens/step, 129.5 t/s. On our 4090 with mismatched draft: 3.76 tokens/step, 67.4 t/s. Matched 3.6 draft would improve this significantly.

## Path to 100+ t/s

1. **llama.cpp PR merges** → DFlash in llama-server → stable persistent API
2. **PR + ngram-mod stacking** → DFlash handles novel text, ngram handles repeats → combined 100+ t/s on code
3. **Lucebox matched 3.6 draft** → when they add SWA layer support → 100+ t/s with DDTree budget=22
4. **PFlash** → 10x TTFT at 128K → instant long-context responses

## Key Files

- `~/llama.cpp/` branch `dflash-pr` — llama.cpp DFlash PR
- `~/models/qwen3.6-dflash-gguf/dflash-3.6-q4km.gguf` — matched draft GGUF
- `~/models/qwen3.6-dflash-gguf/dflash-3.6-f16.gguf` — F16 draft (slower, don't use)
- `~/local-llm/lucebox-hub/` branch `fix/dflash-conv-cache-prefill-mismatch` — our fix
- `~/local-llm/server/dflash_24gb.py` — 24GB VRAM wrapper for lucebox daemon
- `~/local-llm/legacy/scripts/serve-qwen36-fast.sh` — one-shot DFlash script (v1; v3 invokes via `just serve-fast`)

## Context Preset Benchmarks

`just swap 3.6 <preset>` — Qwen3.6-27B Q4_K_M, Q4_0 KV cache, ngram-mod, RTX 4090:

| Preset | Context | Cold | Warm (ngram) | VRAM | Megakernel alongside? |
|--------|---------|------|-------|------|---------|
| `lightning` | 32K | 42 t/s | 95+ t/s | 19.5 GB | Yes |
| `fast` | 64K | 43 t/s | 95+ t/s | 20 GB | Yes |
| `med` (default) | 131K | 43 t/s | 49 t/s | 20.5 GB | Yes (tight) |
| `big` | 262K | 41 t/s | **99 t/s** | 23 GB | No |

Cold = first request (no patterns cached). Warm = repeat/similar request (ngram matches).

DFlash (106 t/s) is independent of context preset — it uses its own max-ctx.
