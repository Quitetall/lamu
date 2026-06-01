# ADR 0014: Single-GPU and ~/local-llm path assumptions are intentional

## Status

Accepted 2026-05-31 · single-GPU half **Superseded by [ADR 0017](./0017-multi-gpu-device-pool.md)** 2026-05-31 (the second-GPU trigger it named has arrived). The `~/local-llm` / `LAMU_ROOT` path half remains in force.

## Context

The Odysseus comparison flagged two LAMU "liabilities": the scheduler only
ever reads `device_by_index(0)` (`scheduler.rs:25/76/92`) and the backends pin
`CUDA_VISIBLE_DEVICES="0"`, so LAMU is single-GPU by construction; and the
project tree is assumed at `~/local-llm` (`config.rs::lamu_root`, plus
`megakernel.rs` / `dflash.rs` / `llamacpp.rs` build-path builders). Both are
real constraints — the question is whether to generalize now.

## Decision

Keep both assumptions. LAMU targets one operator's single-RTX-4090 rig; the
VRAM scheduler, eviction, and fit logic are correct for one GPU. The project
tree is home-relative (`~/local-llm`), not an absolute literal, so it already
works for any user who keeps the conventional layout. Record the upgrade seams
without building them: a `LAMU_GPU_INDEX` env (default 0) read once in the
scheduler + backend spawn, and a `LAMU_ROOT` env override in
`config.rs::lamu_root` consumed by the backend path builders. Build each only
when the triggering reality lands (a second GPU; a relocated repo).

## Rationale

- `device_by_index(0)` is not a bug on a single-GPU machine — it is the
  correct, simplest behavior. Generalizing to multi-GPU pulls in device
  selection, tensor-parallel grouping, and per-device budgets (the deferred
  Phase-4 cookbook work) for hardware that does not exist here yet.
- Home-relative paths already avoid the worst failure (a hardcoded
  `/home/<dev>` absolute — that one *was* a real bug and is fixed separately,
  `tui/mod.rs`). `~/local-llm` breaks only if the user relocates the repo,
  which the `LAMU_ROOT` seam will cover when it happens.
- Building env seams preemptively adds config surface + test matrix for paths
  no one exercises. The seam *design* being recorded means adding it later is a
  small, known change, not a redesign.

## Alternatives Considered

- **Add `LAMU_GPU_INDEX` + multi-GPU scheduling now.** Rejected: no second GPU
  to test against; the bin-packer's single-device math would gain untested
  branches. Deferred to the Phase-4 cookbook multi-GPU work, gated on real
  hardware.
- **Add `LAMU_ROOT` now.** Rejected as premature: home-relative already works
  for the conventional layout; the override is a ~10-line change when a
  relocation actually occurs.
- **Hardcode absolute paths.** Rejected outright — that is the
  `tui/mod.rs:246` bug being fixed, not a pattern to spread.

## Consequences

- LAMU runs correctly only on a single-GPU (device 0), CUDA/Linux host with
  the repo at `~/local-llm` (or `$LAMU_ROOT` once that seam exists). This is
  the documented support envelope, not an accident.
- A contributor adding a second GPU or relocating the tree has a named, small
  task (`LAMU_GPU_INDEX` / `LAMU_ROOT`) rather than discovering the assumption
  by failure.

## Related Decisions

ADR 0003 (single-GPU NVML scheduler — this records *why* the device-0 hardcode
is acceptable), ADR 0004 (backend spawn paths).

## Validation

Revisit `LAMU_GPU_INDEX` when a second GPU is installed; revisit `LAMU_ROOT`
when the repo is cloned somewhere other than `~/local-llm`. Until a concrete
trigger, neither seam is built.
