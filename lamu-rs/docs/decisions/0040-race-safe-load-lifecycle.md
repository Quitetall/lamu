# ADR 0040: Race-safe model load lifecycle — one spawn gate + generation-guarded cleanup

## Status

Accepted 2026-06-13

## Context

Reviewing the cherry-picked load_model mid-spawn fix (commit `04d096a`) surfaced
a latent race in the VRAM scheduler's load lifecycle. The per-device
`loaded: HashMap<name, LoadedModel>` was driven by three blind name-keyed ops:
- `mark_unloaded(name)` — unconditional remove,
- `confirm_loaded(name, pid, …)` — writes pid onto whatever entry holds `name`,
- `mark_loading(entry)` — inserts a fresh entry.

A load attempt (gen-1) that gets drained mid-spawn could then, on resume, either
(a) have its stale `confirm_loaded` adopt a *gen-2* entry (a fresh re-load of the
same name → wrong pid written onto someone else's slot), or (b) have its stale
`mark_unloaded` cleanup *erase* the gen-2 entry, making gen-2's `confirm` fail and
tearing down a server that was never actually evicted.

Reachability: the HTTP path was ALREADY safe — `ensure_loaded_with`
(`lamu-core/src/loader.rs`) serializes every spawn through a process-global
`spawn_gate()` async mutex and re-checks loaded-state after acquiring it. But MCP
`handle_load_model` spawned UNGATED (`backend.load().await` with no gate), so an
MCP load could overlap an HTTP `ensure_loaded` of the same name, or interleave
with a `set_routing_mode` cloud-only drain, and hit the blind ops. Low-probability
(needs a simultaneous cross-surface load of one model) but real. Pure MCP-vs-MCP
can't happen (ADR 0024 serial dispatch).

## Decision

Fix both ends, defense in depth (B closes the race by construction now; A is the
invariant that survives a future per-device parallel-load gate).

**B — one spawn gate across all surfaces.** `loader::spawn_gate()` is made `pub`;
MCP `handle_load_model` acquires it (after the cloud-only refusal check, before
plan/reserve) and **re-checks `is_loaded` under the gate** — a concurrent load
that beat us short-circuits to "already loaded." The gate is held across the whole
evict + spawn + confirm body, exactly as HTTP does. MCP and HTTP load paths are now
mutually exclusive: no two loads (same name or otherwise) overlap. This preserves
today's global-serial-load behavior; per-device parallel loading is a future
ADR 0017 optimization.

**Invariant (load-bearing):** drain/evict/unload paths — `set_routing_mode`
cloud-only, eviction of other models, `handle_unload_model`, `reconcile` — MUST
NEVER acquire `spawn_gate`. They mutate scheduler state under the short-held
`state` lock via UNCONDITIONAL `mark_unloaded(name)`, and run *concurrently* with a
gated load by design. A drain that took the gate would deadlock against a load
holding it across its multi-second spawn. The generation guard (A) is what makes
that concurrency safe.

**A — generation token.** `LoadedModel` gains `generation: u64`; `VramScheduler`
gains a monotone `next_gen`. `mark_loading` stamps + returns a fresh gen.
`confirm_loaded(name, slot_gen, …)` confirms only if the entry's gen matches, else
`Err` (the slot was superseded/removed → caller tears the orphaned server down).
`mark_unloaded_gen(name, slot_gen)` removes only on a gen match; the unconditional
`mark_unloaded(name)` stays for operator/drain/evict/reconcile. Gen is threaded
through **load-path self-cleanup only**: the loader's `spawn_one` + `LoadRollback`
guard, and `handle_load_model`'s teardown + confirm. `handle_load_model`'s two
`mark_loading` calls (an eviction-window reservation, then a placement re-reserve)
hand off `reserve_gen → active_gen = load_gen` at the placement step, assigned
*before* `backend.load().await`, so every teardown uses the live gen.
`rollback.armed = false` after a successful confirm keeps the guard's Drop a no-op.

## Rationale

- The root cause was one un-gated spawn path, not a missing token — `spawn_gate`
  already existed and the HTTP path already proved the pattern. B is the contained
  fix that removes the asymmetry.
- A makes the *load-vs-drain* race safe (drain is gate-free by necessity): a drain
  removes the entry; the load's later `confirm` finds no/changed entry → `Err` →
  orphan teardown (which kills the process via `backend.unload()` first, so the
  gen-gated cleanup no-op leaks nothing), and the load's `mark_unloaded_gen` can't
  clobber a newer reservation.
- Operator removal stays unconditional-by-name: an explicit unload/evict/reconcile
  legitimately removes whatever currently holds the name, regardless of generation.

## Alternatives Considered

- **Gate-only (B alone)** — closes load-vs-load but leaves the gate-free
  drain/evict path able to race a gated load's blind cleanup. Rejected: A is cheap
  and the only thing that makes the necessarily-gate-free paths safe.
- **Generation-only (A alone)** — correct but lets two same-name loads both run to
  completion (wasted spawn, one orphaned). B prevents the wasted work. The user
  chose both, B primary.
- **Per-device parallel-load gate now** — the multi-GPU optimization where the
  global gate becomes a bottleneck. Deferred (ADR 0017); A is the safety net that
  makes it correct when built.

## Consequences

- MCP and HTTP loads are globally serialized (unchanged from prior HTTP behavior).
  A future per-device gate can relax this; A keeps it race-safe.
- One theoretical edge — a loader `confirm` that fails gen-check while a *newer*
  same-name entry exists (so `get_loaded` returns the newer lm) — is closed by the
  gate on the HTTP path (no concurrent same-name load can create that newer entry
  while `spawn_one` holds the gate). Documented, not reachable.
- `confirm_loaded` gained a `slot_gen` parameter (2 call sites: loader, handlers);
  `mark_loading` now returns the gen (callers that don't self-clean can ignore it).

## Related Decisions

ADR 0024 (serial MCP dispatch — why pure MCP-vs-MCP can't race, and why the gate
is the cross-surface fix), ADR 0017 (multi-GPU — the future per-device gate where A
is load-bearing), and the `04d096a` review that surfaced this.

## Validation

7 new scheduler tests: `mark_loading` returns strictly-increasing gens;
`confirm_loaded` correct-gen `Ok` / stale-gen `Err` with the entry untouched;
`mark_unloaded_gen` stale → no-op, current → removes; `mark_unloaded`
unconditional; and `race_regression_stale_gen1_cannot_clobber_gen2` — after a
drain + gen-2 re-load, a stale gen-1 `mark_unloaded_gen` is a no-op and a stale
gen-1 `confirm` `Err`s while the gen-2 entry stays intact. lamu-core 216 +
test_scheduler 17 + lamu-mcp 118 green; workspace builds clean. Live gate
(queued, GPU training): a real concurrent MCP-load + HTTP-request for the same
model exercising the gate.
