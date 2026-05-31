# ADR 0015: Cookbook roofline + composite scoring engine (ported from hwfit)

## Status

Accepted 2026-05-31

## Context

`lamu cookbook` was a 3-bucket heuristic (`fit_bucket` in `lamu-cli`): compare
a model's static `vram_mb` against current free/total VRAM → FitsNow /
AfterUnload / TooBig, over a 5-row hand-typed `CURATED` list. It could not
answer "fits, but at 4 tok/s" vs "80 tok/s", nor "fits at Q4 but not Q8", nor
"fits if you drop context to 16k". The Odysseus comparison
(`docs/comparison-odysseus.md`) scored its `hwfit` engine — a roofline tok/s
estimate + a 4-factor weighted composite over a vendored model DB — as the
clearest capability LAMU lacked.

`hwfit` is pure arithmetic over `(model_record, hardware)`, and LAMU's registry
already carries every input it needs (`ModelEntry { params_b, quant,
context_max, arch }`) plus NVML via `VramScheduler`.

## Decision

Port the portable core of `hwfit` (`fit.py` + `models.py`) into a new pure
module `lamu-core/src/cookbook.rs`: `estimate_mem_gb`, `estimate_tps`
(roofline: `(bandwidth / (active_pb · bytes_per_param)) · 0.55 · mode_factor`,
with a `FALLBACK_K/pb · speed_mult` path when bandwidth is unknown),
`try_fit` (context-halving search down to 1024), the four sub-scores
(quality / speed / fit / context), `score_model`, and `rank`. Tuning tables
(QUANT_*, GPU_BANDWIDTH ~75 rows, USE_CASE_WEIGHTS, SPEED/CONTEXT_TARGET,
FALLBACK_K) are transliterated as Rust `const` slices. `VramScheduler` gains
`gpu_name()` for the bandwidth lookup. The scorer is unit-tested for **numeric
parity** against hand-computed `fit.py` values.

## Rationale

- The arithmetic transliterates almost line-for-line and is the single biggest
  capability gap; throughput-awareness also feeds the scheduler double-down
  (ADR-to-come) so eviction can reason about reload cost.
- Keep `hwfit`'s load-bearing distinction: **TOTAL params drive VRAM (all MoE
  experts resident); ACTIVE params drive roofline + KV.** LAMU's old
  `fit_bucket` got both wrong.
- Keep too-tight models in the output (`fit_level = TooTight`, score 0) so a
  `--simulate-vram` view can show red rows for hardware the user is
  considering — silently dropping them hides exactly what the user wants to
  see.
- `fit_level` is ratio-based (`required/vram`) rather than `hwfit`'s
  `recommended_ram_gb` headroom, because LAMU registry entries carry no
  `recommended_ram_gb`. The composite SCORE still uses the faithful
  ratio-based `fit_score`, so parity holds; only the cosmetic glyph tier
  diverges, and that is documented.

## Alternatives Considered

- **Import Odysseus's 898-row `hf_models.json`.** Rejected: it is someone
  else's HF crawl; LAMU's registry is its own scanned/pulled models. The
  scorer points at registry entries — no new model DB.
- **Reproduce the dual `QUANT_BPP` (VRAM) vs `QUANT_BYTES_PER_PARAM` (speed)
  tables faithfully, including their ~10% k-quant drift.** Both were ported as
  given (they are genuinely different numbers in `models.py`), but the drift is
  accreted, not designed — flagged so a future cleanup can collapse them; it is
  below the 0.55 efficiency-fudge noise floor.
- **Tables as a tunable embedded TOML/YAML with on-disk override.** Deferred:
  Phase-1 ships `const` tables (no new dep, fastest path to a parity-tested
  scorer). The retunable-without-recompile override is a follow-up; the
  constants are isolated at the top of the module for an easy swap.
- **Port `hardware.py`'s remote-SSH / Windows / AMD-APU probing.** Rejected:
  ~250 LOC for a single-user local box; local NVML suffices (+ one
  `/proc/meminfo` read later for the cpu-offload branch).
- **Multi-GPU sharded-VRAM split + `_group_gpus`.** Deferred to Phase 4, gated
  on a real second GPU (ADR 0014).

## Consequences

- New `lamu-core::cookbook` module + `VramScheduler::gpu_name()`. The CLI
  (`lamu cookbook`) and the old `fit_bucket`/`Fit` enum are retired in the
  follow-up wiring commit.
- Phase 1 treats every model as DENSE (active == total). MoE models
  over-estimate speed until Phase 2 adds `n_experts`/`active_params` to
  `ModelEntry` (GGUF `expert_count`/`expert_used_count` parse).
- The `0.55` efficiency fudge + bandwidth specs are baked into the binary until
  the tunable-override follow-up lands.
- Parity tests pin the math to `fit.py`; a future `hwfit` change won't silently
  diverge LAMU without a test failure.

## Related Decisions

ADR 0003 (single-GPU scheduler — supplies `gpu_name`/VRAM), ADR 0014
(single-GPU deferral — why the sharded split is Phase 4), and the
throughput-aware-eviction double-down that consumes this engine.

## Validation

`cookbook::tests` assert numeric parity against hand-computed `fit.py` results
(mem estimate, roofline tps, full composite for a 7B/Q4_K_M/4090 case),
plus too-tight retention, lower-quant fallback, and rank ordering. Revisit when
MoE metadata lands (Phase 2) or when the tables become a tunable override.
