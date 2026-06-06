# ADR 0023: Backend / module / frontend architecture

## Status

Accepted 2026-06-06

## Context

The ecosystem has grown three kinds of thing that are currently tangled:
the LLM-serving base, capability add-ons (image-gen via ComfyUI, TTS via
fish-speech, research aggregation via JART), and surfaces that drive them
(Claude Code, the MCP/HTTP servers, Odysseus, JART's own TUI). The directive:
**LAMU is the base layer; modules MODIFY/EXPAND the backend; frontends only
DRIVE tools.** JART should be *both* — a backend module (sources/fetch/cache/
summarize/index) plus a bundled-in frontend (TUI/web) that drives it. Use
in-repo workspace crates, not dynamic plugins.

A 5-area investigation + 3-lens adversarial review of the real code grounded
this and corrected several assumptions:

- **Core is already almost modality-agnostic.** The `Modality` enum is read in
  ~6 places, but core *logic* couples to it in exactly ONE: `scheduler.rs:375`
  sorts eviction candidates by `is_llm()`. Modality *routing* already lives
  outside core (`lamu-mcp/image.rs:is_local_image`, `tts.rs:is_local_tts`).
- **The real couplings to break are small:** the closed `BackendType` enum +
  the `make_backend` match (`backends/mod.rs:23-33`) + that one scheduler sort.
- **The load-bearing hard part is the MCP tool seam, NOT a backend seam.**
  `HandlerKind::Stateful` is `fn(&LamuMcpServer, …)` and the image/tts handlers
  call `server.state.lock()`, `scheduler.get_loaded()`, `handle_load_model()`.
  For a module to register a tool without depending on `lamu-mcp`, those needs
  must be abstracted behind a `&dyn ToolCtx`. This is decided here, not deferred.
- **JART is not in this repo** — it is a standalone edition-2021 crate at
  `~/Desktop/jart`. Modularizing it is *import-then-split*, not an in-place move.
- **Odysseus is not a thin frontend** — it owns its own deep-research engine,
  agents, and memory. It is an *external consumer app*, a fourth bucket.

## Decision

Adopt a four-bucket taxonomy and make modules **self-register into core seams**
rather than core enumerating them:

1. **`lamu-core` — base layer (mechanism only).** Owns the `Backend` trait, the
   VRAM scheduler, registry/loader/reconcile, and the child-process hardening
   helpers. It knows HOW to spawn/health/evict/kill a managed subprocess and how
   to tier VRAM — never WHAT a backend does. It stops owning a closed
   `BackendType` dispatch enum and the `make_backend` match.
2. **Modules (`lamu-image`, `lamu-tts`, `lamu-jart`) — expand the backend.**
   In-repo crates that depend on `lamu-core` (never the reverse). Each ships
   `pub fn register(reg: &mut LamuRegistry)` that inserts its backend factory
   (+ eviction tier) and its tool definitions.
3. **Frontends (`lamu-mcp`, `lamu-api`, `lamu-cli`, the bundled `jart` binary)
   — drive tools.** They are composition roots: at startup they call each
   module's `register()`, then expose the assembled registry over their
   transport (stdio JSON-RPC, HTTP `/v1`, TUI). They hold no capability.
4. **External consumer apps (Odysseus, Claude Code, Codex) — out of the
   workspace.** They consume `:8020` / MCP like any client. Odysseus keeps its
   own engine; it is NOT a lamu frontend.

Core grows two seams: a `BackendRegistry` (`backend_kind: String → factory fn +
u8 eviction tier`) and a tool registry that generalizes the already-table-driven
MCP `TOOLS` catalog, with tool handlers taking a `&dyn ToolCtx` (lock entries /
resolve a loaded port / trigger a load) instead of a concrete `&LamuMcpServer`.
The `Modality` enum **stays** as a serde-default(Llm) metadata field — it has
real readers (`/v1/models` modality from ADR-P3 / task #164, the mcp routing
helpers, the cli) — but no core logic branches on it; eviction sorts on a
generic `eviction_tier()`.

## Rationale

- The coupling is tiny (one match, one sort, one enum), so the lowest-friction
  mechanism wins: explicit per-module `register()` at the composition root.
  Rejected heavier options below.
- `backend_kind: String` is serde wire-compatible: the YAML already stores
  snake_case kind strings via the `ModelEntryYaml` DTO, so `llama_cpp` /
  `fish_speech` round-trip unchanged (with `comfyui` keeping its explicit serde
  rename — there are **six** kinds: `llama_cpp, megakernel, dflash,
  dflash_lucebox, fish_speech, comfyui`, not three).
- Abstracting the tool handler over `&dyn ToolCtx` is what actually lets
  `lamu-image`/`lamu-tts`/`lamu-jart` ship MCP tools without a `lamu-mcp`
  dependency. Without it the whole "modules register tools" claim is false.
- JART-as-module-plus-frontend matches the directive literally: its fetch/cache/
  summarize/index becomes `lamu-jart` (research tools), its TUI/web becomes a
  bundled frontend driving them.

## Alternatives Considered

- **Keep the closed `BackendType` enum + `make_backend` match** — rejected: it
  forces core to name every module's backend, the exact coupling to remove.
- **Dynamic plugins (`dlopen`)** — rejected: no infra, ABI fragility, and the
  directive says in-repo crates.
- **`inventory`/`linkme` link-time registration** — rejected: distributed
  registration is harder to reason about than an explicit composition root for a
  3-module system; trades a one-line `register()` call for link-order magic.
- **Treat Odysseus / JART-TUI as ordinary frontends** — rejected: Odysseus owns
  capability (its own research engine), so it is an external app, and JART's
  capability half belongs in a module, not a frontend.

## Consequences

- New public core seam (`LamuRegistry`, `ToolCtx`, `eviction_tier`,
  `register_builtin`) that all frontends must call — a single composition root
  helper should wrap the module `register()` calls so no frontend silently drops
  a module (composition-root drift is a real risk; guard with a test).
- Backend dispatch moves from compile-time-exhaustive (enum match) to
  runtime-keyed (string → factory): an unknown `backend_kind` fails at model
  load, not at compile. Acceptable; the registry returns a clear error.
- `harden_child_command`, `read_log_tail`, `build_payload` get promoted from
  `pub(crate)` to `pub` so extracted backends keep their hardening.
- `media_paths.rs` moves `lamu-mcp → lamu-core` (zero-dep confined-output gate)
  so both media modules share one safety path.
- JART's Python scrapers currently spawn `python3` *without* hardening; folding
  them into `lamu-jart` is an opportunity to adopt `harden_child_command` (new
  behavior, not a move).
- dots.tts (rednote-hilab) is a real candidate engine for `lamu-tts`, but its
  interface (OpenAI `/v1/audio/speech`-shaped server vs Python lib needing a
  wrapper) is **unverified** — gate the second TTS engine on confirming it.

## Migration order (build + `lamu serve` green at every step)

0. **Seams, no behavior change.** Move `media_paths.rs` into core; promote the
   three `pub(crate)` helpers to `pub`; add the `LamuRegistry`/`BackendRegistry`
   + `ToolCtx` types ALONGSIDE the existing enum (nothing uses them yet).
1. **Scheduler decouple.** First WRITE the missing eviction-order test (none
   exists today — pin "media tier evicts before LLM, oldest-first within tier"),
   then add `eviction_tier()` (derived from `Modality` for now) and switch
   `scheduler.rs:375` to it.
2. **Serde-safe `backend_kind`.** Add the string field to `ModelEntryYaml`
   (the wire DTO), round-trip-test all six kinds, route `make_backend` through
   the registry while the enum still exists as a mirror.
3. **Extract `lamu-image` FIRST** (cleanest: the backend moves, the tool either
   stays in mcp against `ToolCtx` or moves with it). Prove the module pattern.
4. **Extract `lamu-tts`** (same pattern; introduce the engine seam).
5. **Add dots.tts** inside `lamu-tts` once its interface is confirmed.
6. **Import JART** into the workspace (edition bump, fix `CARGO_MANIFEST_DIR`
   path resolution), then split `lamu-jart` (module) + `lamu-jart-frontend`.
7. **Delete compat shims** (the `BackendType` enum as dispatch key) once every
   caller uses explicit registration.

## Related Decisions

ADR 0016 (backend orchestrator / BYO frontend — this generalizes it), ADR 0021
(context-occupancy — its `/v1/models` modality reader is why `Modality` stays),
the media-modalities design (TTS/image, now relocated to modules).

## Validation

Right if, after step 3, `cargo build --workspace` + `lamu serve` are green,
image-gen still works end-to-end, `lamu-core` has no `image`/`tts`/`comfyui`
symbols, and `lamu-image` depends on `lamu-core` (not vice-versa). Wrong / revisit
if the `ToolCtx` abstraction can't express what a real tool needs from the server
without leaking `lamu-mcp` types, or if the serde swap changes one byte of an
existing `models.yaml` round-trip.
