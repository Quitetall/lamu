# Model Selection for RTX 4090

## Current: Qwen3.6-27B Dense Uncensored (Heretic v2)

Best option for single 4090. Here's why:

| Model | Params | VRAM (Q4) | SWE-bench | Refusals | Fits 4090? |
|-------|--------|-----------|-----------|----------|------------|
| **Qwen3.6-27B Dense** | 27B | 16 GB | 77.2 | 6/100 | **Yes, 262K ctx** |
| Qwen3.6-35B-A3B MoE | 35B (3B active) | 20 GB | 73.4 | 10/100 | No (no KV room) |
| Qwen3.5-27B (DFlash) | 27B | 16 GB | 75.0 | 83/100 | Yes, 8K via DFlash |

## Why Dense > MoE on Single GPU
MoE loads ALL 35B params into VRAM (all experts must be resident) even though only 3B activate per token. 20 GB model = no room for KV cache.

Dense 27B: smaller VRAM footprint AND higher benchmarks.

MoE wins when: you have 2+ GPUs (model splits, 3B-active inference is fast).

## Uncensored (Heretic) vs Censored
- 94% fewer refusals (6/100 vs 92/100)
- 0.0021 KL divergence (quality preserved)
- MMLU: 85.61% vs 86.65% (marginal drop)

For swarm workers, uncensored = zero wasted retries on refusals.

## Available Quants
| Quant | Size | Best For |
|-------|------|----------|
| Q5_K_S | ~18 GB | Best quality, 108K context (Q8 KV) |
| Q4_K_M | ~16 GB | Max context, 262K (Q4 KV) |

Both GGUFs downloaded. serve-qwen36.sh auto-picks best available.
