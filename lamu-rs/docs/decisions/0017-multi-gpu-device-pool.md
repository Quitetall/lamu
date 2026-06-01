# ADR 0017: Multi-GPU device pool + best-fit placement + opt-in tensor-parallel sharding

## Status

Accepted 2026-05-31

**Supersedes ADR 0014** (the single-GPU half; the `~/local-llm` / `LAMU_ROOT`
path half of 0014 is unaffected and remains in force).

## Context

ADR 0014 recorded, honestly, that LAMU was single-GPU *by construction* and that
generalizing was premature **because there was no second card to test against**.
That rationale was correct for its moment. The moment has changed: the operator
now requires genuine multi-GPU support — distinct models on distinct cards
simultaneously, and models too large for one card spread across several. This
ADR does not claim 0014 was wrong; it records that the triggering reality 0014
explicitly named ("revisit when a second GPU is installed") has arrived, and the
deferred work is now in scope.

The state ADR 0014 froze is concrete and shallow-but-wide:

- **Scheduler** (`lamu-core/src/scheduler.rs`) is a single scalar pool.
  `VramScheduler { reserved_mb, loaded: HashMap<String,LoadedModel>, total_mb,
  nvml }` reads only `device_by_index(0)` (ctor, `gpu_name()` L42-47,
  `query_vram()` L81-94, `query_gpu_pids()` L97-114). `available_mb()` (L49-60)
  is one global number; `plan_eviction`/`plan_load` (L192-237) operate over one
  flat map; `reserved_mb` (1500) is charged once.
- **Backends** each hardcode `CUDA_VISIBLE_DEVICES="0"` — `llamacpp.rs:85,160`
  (and the literal-`"0"` assertion at `:610`), `dflash.rs:86`,
  `megakernel.rs:72`, `fish_speech.rs:127`, `comfyui.rs:88`. No backend has a
  device parameter; `Backend::load(&mut self, entry, port)` (`mod.rs:56`) has no
  device arg.
- **Loader** (`loader.rs`) places nothing: `ensure_loaded_with` calls the global
  `plan_load`, and the post-spawn VRAM read-back (L162-169) queries device 0
  only — it would miss a PID on card 1.
- **Cookbook** (`cookbook.rs`) already has a pluralizable shape but is wired to
  one card; ADR 0015 explicitly **deferred** "Multi-GPU sharded-VRAM split +
  `_group_gpus`" to a Phase 4 gated on ADR 0014.
- The `LAMU_GPU_INDEX` seam that 0014 *named* was never built; grep confirms
  only docs mention it.

The constraint that produced 0014 has not fully lifted: the dev rig still has
one card. So this ADR must ship single-GPU-identical behavior as the
zero-config default and isolate every genuinely-multi-GPU code path behind
synthetic-pool unit tests, with real CUDA masking / `--tensor-split` marked
hardware-validation-pending — the same honesty 0014 used, now applied to the
*other* side of the decision.

## Decision

Replace the scalar VRAM pool with a **per-device pool** and add a **placement
step** to the load path, in four shippable phases, governed by one invariant:
**when `device_count() <= 1`, or when `LAMU_GPU_INDEX` pins a single index, every
observable number and every spawn env is byte-identical to the ADR-0014 code.**

1. **Per-device scheduler.** Introduce `DeviceBudget { index, name, total_mb,
   reserved_mb, loaded: HashMap<String,LoadedModel> }` with its own
   `available_mb()` = `total - max(sum_registered_here, actual_used_here) -
   reserved`. `VramScheduler` becomes `{ devices: Vec<DeviceBudget>, nvml }`,
   enumerated via `nvml.device_count()` at ctor; a single-card rig yields a
   1-element Vec. `reserved_mb` is charged **per device** (driver/context
   overhead is per-context), which is intentionally slightly more conservative
   than naive summing. The existing scalar methods (`total_mb`,
   `available_mb`, `gpu_name`, `budget`) survive as **aggregate facades** so no
   caller in lamu-mcp/lamu-api/lamu-cli churns; `budget()` gains an additive
   `per_device: Vec<DeviceVram>`.

2. **Placement types + policy.** New `enum DevicePlacement { Single(u32),
   Sharded(Vec<u32>) }` (`types.rs`); `LoadedModel` gains
   `#[serde(default)] device` (default `Single(0)`); `VramBudget` gains
   `#[serde(default)] per_device`. A `place(entry) -> DevicePlacement`:
   **best-fit** — among devices that fit, pick the one with the **most** free
   VRAM (spreads load, leaves headroom), tie-break lowest index; if no single
   card fits but the sum does and the entry is shardable, return `Sharded`;
   else fall through to per-device `plan_eviction` on the best candidate.
   `LAMU_PLACEMENT=best-fit|tight-fit` exposes the one genuinely tunable knob.

3. **Device-threaded backends.** Prefer an **additive `set_device(&mut self,
   DevicePlacement)` default-noop on the `Backend` trait** over changing
   `load`'s signature, so only `LlamaCppBackend` must honor sharding and the
   four Python backends only need their single-index env set by
   `make_backend(entry, device)`. Single → `CUDA_VISIBLE_DEVICES=<i>` +
   `--main-gpu 0` (device is index 0 post-mask). Sharded (llama.cpp only) →
   comma-join mask + `--split-mode layer` (default; `row` via
   `LAMU_SPLIT_MODE`) + optional `--tensor-split` proportional to per-card
   total. Python backends **reject `Sharded` with a clear error** — whole-model
   servers can't tensor-parallel.

4. **Wired loader + cookbook.** `ensure_loaded_with` calls `plan_load_placed`,
   `mark_loading(entry, device)`, threads the placement to spawn, reads VRAM
   back from the **placed** device(s), and `confirm_loaded(..., device)`.
   `Hardware` gains `devices: Vec<DeviceSpec>` (keeping `gpu_vram_gb` =
   device[0] for parity); `score_model` fits un-sharded entries against the
   largest single device and shardable entries against summed VRAM, **closing
   ADR 0015's deferred Phase 4**.

Config seams (all via `config.rs::parse_env_or`): `LAMU_GPU_INDEX` (pin one
card — the zero-config escape hatch and ADR-0014 fulfillment), `LAMU_GPU_INDICES`
(comma list of usable cards, default all), `LAMU_SPLIT_MODE`, `LAMU_PLACEMENT`.

## Rationale

- **The facade is the back-compat lever, not a nicety.** `VramBudget` is
  serialized into MCP `vram_status` and `lamu-api/metrics.rs:145`; `gpu_name`
  feeds cookbook bandwidth lookup; dozens of callers read the scalars. Keeping
  them as aggregates over the Vec means the multi-device change is *internal*
  until a caller opts into `per_device` — the difference between a localized
  change and a workspace-wide churn.
- **Best-fit (most-free) is the right default for this workload.** LAMU loads a
  few large, long-lived models, not many small ephemeral ones. Spreading by
  most-free keeps headroom on each card for the no-auto-evict HTTP path (ADR
  0006) and realizes the exact scenario ADR 0003 L41 flagged: heavy LLM on card
  0, S2-Pro/ComfyUI on card 1, concurrently. Tight-fit packs better but risks
  thrashing; it stays available behind `LAMU_PLACEMENT` rather than as default.
- **`set_device` default-noop beats a signature change.** The signature change
  touches all five backend impls plus the `FakeBackend` in `loader.rs` tests,
  `handle_load_model`, and the CLI swap path. The additive trait method confines
  the real work to llama.cpp (the only backend that can shard) and lets Python
  backends inherit the no-op while still getting their single-index env from the
  constructor.
- **Per-device `reserved_mb` matches physical reality.** CUDA context/driver
  overhead is incurred once per device, so charging the 1500 MB reserve per card
  is *more* correct than charging it once globally — it just looks like slightly
  less usable VRAM than a naive sum, which we document rather than hide.
- **Shard only when forced.** Tensor-parallel across consumer 4090s (no NVLink)
  pays per-layer PCIe transfer; `--split-mode layer` (pipeline) is cheaper than
  `row`, hence the default. Sharding a model that *did* fit one card is a perf
  regression, so `Sharded` is gated strictly on `vram_mb > max single-device
  total`.
- **Phasing keeps every step green and shippable.** Phase 1 already delivers
  correct per-device reporting on multi-GPU rigs while still loading single-card;
  Phase 2 delivers cross-card placement; Phase 3 the 70B-on-2x24 case; Phase 4
  the cookbook + the supersede. Each compiles with existing tests byte-identical
  because a 1-element Vec reproduces today's math.

## Alternatives Considered

- **Change `Backend::load` to take a device arg.** Rejected as wider churn for
  no extra capability: only llama.cpp uses anything but a single-index env, and
  `make_backend(entry, device)` + a `set_device` no-op covers the rest without
  touching five impls and their fakes.
- **Replace the scalar scheduler API outright (no facades).** Rejected: it
  fans churn across lamu-mcp, lamu-api, and lamu-cli and breaks serialized
  `VramBudget` consumers. Facades + additive `#[serde(default)]` fields keep the
  JSON purely additive.
- **Tight-fit (most-packed) as default.** Rejected as default: it maximizes the
  chance one large model is blocked by two mediums fragmenting two cards, and it
  thrashes the no-auto-evict path. Kept as an opt-in (`LAMU_PLACEMENT`).
- **Round-robin / lowest-index-first placement.** Rejected: ignores actual free
  VRAM, so it can place onto a card that then can't hold a later model best-fit
  would have packed elsewhere.
- **Build sharding before single-card placement (skip Phase 2).** Rejected:
  cross-card placement of distinct whole models is the common, high-value case
  and is testable with synthetic pools; sharding is the rarer case and is
  hardware-blocked. Ordering placement first delivers value sooner.
- **Generalize speculatively, as 0014 warned against.** Not applicable anymore —
  the requirement is now explicit. But the *discipline* survives: the genuinely
  untestable parts (real masking, `--tensor-split`) are isolated and marked
  pending, not asserted as validated.

## Consequences

- **Single-GPU stays the zero-config default forever.** A 1-element device Vec
  reproduces ADR-0014 numbers exactly; `LAMU_GPU_INDEX` pins a non-zero card on
  the unchanged single-pool fast path. This is the support floor, not a
  transitional state.
- **`LoadedModel`/`VramBudget` grow additive serde-default fields.** Existing
  snapshots, MCP `vram_status`, and `lamu-api` metrics consumers keep parsing;
  `device` defaults to `Single(0)`, `per_device` is new JSON.
- **Per-device accounting becomes load-bearing for correctness.** The post-spawn
  VRAM read-back MUST query the placed device (else `available_mb` drifts toward
  the `entry.vram_mb` estimate), and `orphan_pids`/eviction become per-device —
  an orphan on card 1 must not shrink card 0's budget. These are the two
  correctness hotspots Phase 2 must get right.
- **Eviction is now per-device.** `plan_eviction` evicts only from the target
  device's `loaded` map; the modality-tiered LRU (ADR 0003) runs per card.
- **The cookbook deferral closes.** ADR 0015's Phase-4 stub
  (`effective_vram = single_gpu_vram`) is replaced by largest-single-vs-summed
  fit; `lamu cookbook` ranks correctly on multi-GPU rigs.
- **Real multi-GPU behavior is not yet hardware-validated.** Synthetic
  `set_devices_for_tests` pools unit-test the bin-packer/placement/eviction math;
  actual `CUDA_VISIBLE_DEVICES` masking and `--tensor-split` cannot be validated
  until a second card lands. Phase 3 ships "merged, hardware-validation pending."
- **Two new tunable knobs** (`LAMU_PLACEMENT`, `LAMU_SPLIT_MODE`) and two
  enumeration controls (`LAMU_GPU_INDEX`, `LAMU_GPU_INDICES`) enter the config
  surface — a deliberate, documented expansion.

## Related Decisions

ADR 0014 — **superseded by this ADR** (single-GPU half only; update its Status
line to "Superseded by 0017"). ADR 0003 — the modality-tiered eviction this ADR
makes per-device; its L41 "revisit on a second GPU" trigger is now realized
(add a Related pointer to 0017). ADR 0015 — its deferred multi-GPU Phase 4 is
resolved here (add a Related pointer to 0017). ADR 0004 — backend spawn paths
that now carry the device env. ADR 0006 — the no-auto-evict HTTP guarantee that
per-device headroom (best-fit) protects.

## Validation

- **Per-phase green gate.** Each phase keeps `cargo test --workspace` green;
  single-GPU tests pass byte-identical (1-element Vec). The `llamacpp.rs:610`
  literal-`"0"` assertion stays valid when the test forces `Single(0)`.
- **Synthetic multi-device unit tests** via `set_devices_for_tests(Vec<(u32,
  u32)>)` (added alongside, not replacing, `set_total_mb_for_tests`): 2-device
  budget math, per-device `available_on(idx)`, best-fit picks most-free,
  tie-break lowest index, per-device eviction does not touch the other card,
  `Sharded` chosen only when no single card fits, Python backend rejects
  `Sharded`.
- **Cookbook parity preserved.** `cookbook.rs:545-640` single-4090 numbers must
  not drift (`gpu_vram_gb` stays device[0]); multi-device fit is new tests only.
- **Hardware-validation-pending, explicitly.** Real masking + `--tensor-split` +
  the 70B-on-2x24 case are validated only when a second GPU is installed — until
  then they are merged-but-unproven. We know this decision was right when, on
  multi-GPU hardware, distinct models land on distinct cards by best-fit and a
  model exceeding one card loads sharded; we'd know it was wrong if per-device
  accounting drifts under load or PCIe sharding overhead makes layer-split
  unusable (revisit `--split-mode`/sharding gating then).

