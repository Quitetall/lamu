# LAMU Cookbook Redesign + Double-Down Plan

## Part 1 — Cookbook: port hwfit's portable core into Rust

LAMU's `lamu cookbook` today (lamu-cli/src/main.rs:261, `fit_bucket` :237, `CURATED` :253) is a 3-bucket heuristic comparing a STATIC per-model `vram_mb` against current free/total VRAM. Odysseus's hwfit is a pure-arithmetic roofline + 4-factor composite scorer that LAMU can adopt almost line-for-line because **LAMU's registry already carries every input** (`ModelEntry { params_b, quant, context_max, vram_mb, arch }`, types.rs:250) and the NVML plumbing exists (`VramScheduler`, scheduler.rs).

### Port cleanly
1. Roofline tok/s (fit.py:62): `(bw/(active_pb*bpp))*0.55*mode_factor`; FALLBACK_K path for unknown GPUs.
2. GPU_BANDWIDTH ~95-row table, substring-matched longest-key-first against `device.name()`.
3. `estimate_memory_gb = pb*bpp + 8e-6*active_pb*ctx + 0.5` — context-aware budget.
4. Context-halving search (fit.py:162) → "fits after dropping ctx to N".
5. 4-factor weighted composite (quality/speed/fit/context × USE_CASE_WEIGHTS).
6. Dense quant-tier walk → "fits at Q4 but not Q8".

### Skip (matches LAMU's lean single-GPU philosophy)
- hardware.py remote-SSH/Windows/AMD-APU probing (~250 LOC). Local NVML + one /proc/meminfo read suffice.
- The dual QUANT_BPP/QUANT_BYTES_PER_PARAM tables — port ONE (QUANT_BPP), reuse for speed; the discrepancy is accreted drift, below the 0.55-fudge noise floor.
- cookbook_routes.py download/serve apparatus (1729 LOC; LAMU has `lamu pull`), image_models.py, the web what-if plumbing.

### Defer (multi-GPU only)
- GGUF-single-vs-sharded VRAM split + `_group_gpus`. Stub `effective_vram = single_gpu_vram`; wire the real split when a 2nd card lands.

### Algorithm shape (`lamu-core/src/cookbook.rs`, pure, no I/O)
`Hardware`, `FitResult`, `estimate_mem_mb`, `estimate_tps`, `try_fit` (halving), `score_model` (composite), `rank`. Keep hwfit's load-bearing distinction: **TOTAL params for VRAM, ACTIVE params for roofline+KV**. Keep too-tight models in the output (fit_level=TooBig, score 0) so `--simulate-vram` shows red rows.

### Tables as tunable embedded TOML
Ship QUANT_BPP/SPEED_MULT/QUALITY_PENALTY/HIERARCHY, USE_CASE_WEIGHTS, SPEED/CONTEXT_TARGET, FALLBACK_K, GPU_BANDWIDTH as `cookbook_tables.toml` via `include_str!` + once_cell, on-disk override allowed. The 0.55 efficiency fudge and bandwidth specs are per-rig tuning knobs — no recompile to retune.

### Model-DB strategy
- Do NOT import the 898-row HF snapshot. Score lamu-core::registry entries directly.
- Add `Option<u32> n_experts, active_params_m` (serde(default)) to ModelEntry; populate by extending parse_gguf_meta (registry.rs:56) to read `<arch>.expert_count`/`expert_used_count` (GGUF type-4, ~15 LOC at the existing type-4 arm). None → dense.
- Replace 5-row CURATED with a tiny enriched `curated.toml` (~15-20 rows with params_b/quant/ctx/moe/repo) so suggestions get the same scored treatment.

### CLI
`lamu cookbook [--use-case] [--quant] [--ctx] [--simulate-vram MB] [--top N] [--json]`. Retire `fit_bucket`+`Fit`. Map fit_level→traffic-light glyph (keep the familiar UX, real numbers behind it). `--json` feeds the TUI-dashboard TODO and a new `mcp__local-llm__cookbook` tool so the outer agent picks models by predicted throughput. Add one `/proc/meminfo` MemAvailable read for the cpu_offload branch.

### Phasing
1. Core scorer + tables + gpu_name(), all-dense, parity-tested. 2. CLI wiring, retire fit_bucket. 3. MoE fidelity. 4. Suggestions + simulate + MCP. 5. (deferred) multi-GPU split. One commit per phase → `review_commit`.

## Part 2 — Double-down: make genuine edges stronger

1. **Cross-modal VRAM scheduler** → feed cookbook's ctx-aware estimate + roofline INTO plan_eviction/plan_load: reload-cost-aware eviction sort, ctx-parameterized fit (stop over-reserving), and a "I can load this if I drop you to 16k ctx" fallback. Reuses ported code, stays single-GPU.
2. **PDEATHSIG lifecycle** → close the reconciliation loop: diff `query_gpu_pids()` (already implemented, unused in budget path) vs registered PIDs, surface orphan holders in `budget()`, add guarded `lamu reclaim`. Turns a defensive `max()` guard into actionable diagnosis.
3. **Bubblewrap sandbox** → extend to the review_commit cargo preflight + git reads (hostile `build.rs` runs during `cargo test` today, un-sandboxed). Structural complement to the #154 prompt envelope, reusing the `lamu agent` harness.
4. **Judged council** → hardware-size the council (sequential load/unload on one GPU via plan_eviction), envelope the blind-answer + judge concatenation (#154's highest-value site — the verdict is acted upon), log cross-vendor disagreement to temporal memory as calibration data.
5. **Temporal memory** → the id-guard is right; the fact-TEXT path is the gap: envelope recalled fact bodies into the contradiction judge + recall return, add a provenance field (user-stated > tool-ingested on conflict), surface superseded-fact history in recall.

All five extend existing structures; none add a subsystem; all hold to the single-user, loopback-default, single-GPU calibration.
