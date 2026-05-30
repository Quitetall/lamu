# LAMU Media Modalities — Design Doc

**Status:** proposal (no code yet — design-first per the build decision)
**Scope:** add image-generation + TTS (and a path to other modalities) to LAMU
as managed, VRAM-budgeted inference backends, exposed over MCP + HTTP.
**Author:** drafted with Claude Opus 4.8, 2026-05-29.

---

## 1. Goal & non-goals

### Goal
LAMU already is a Rust **process orchestrator + VRAM manager** for LLM
inference servers (`llama-server`). Extend that exact pattern to non-LLM
modalities:

- **Image generation** — SDXL, SD3.5, Pony, Flux.2, served through **ComfyUI**
  (one engine, many checkpoints, API mode).
- **Text-to-speech** — **fish-speech** first, with room for Kokoro / XTTS / Piper.
- A generic **modality backend** seam so future modalities (image→video, ASR,
  embeddings-as-a-service, etc.) drop in without re-plumbing.

The differentiator is **unified cross-modal VRAM scheduling**: the existing
`VramScheduler` budgets an SDXL checkpoint or a fish-speech model *in the same
pool* as the LLMs, evicting across modalities (drop a 7B LLM to fit Flux, then
reload it) — something neither ComfyUI nor a bare TTS server does on its own.

### Non-goals
- **No reimplementation** of diffusion / vocoder math in Rust. These are Python
  ecosystems; LAMU *manages* them as subprocesses, exactly like `llama-server`.
- **No Python in the Rust build.** The engines are a **runtime** dependency (a
  spawned process), never a cargo dependency. The default `cargo build` must
  stay lean — media lives behind a cargo feature (`media`), like `turbovec`.
- **No bundled weights / no license laundering.** The user downloads weights;
  LAMU detects + manages them. Licensing is the user's to accept (see §8).

---

## 2. Where it slots into the current architecture

```
                         ┌─────────────────────────────────────────┐
                         │            lamu-core                     │
   MCP tools  ┌────────► │  VramScheduler (NVML)  ── budgets ALL ── │
   HTTP compat│          │  Backend trait (spawn/health/proxy/evict)│
              │          │  Registry: ModelEntry { modality, vram } │
              │          └───────────────┬──────────────────────────┘
              │                          │ spawns + health-polls + proxies
   ┌──────────┴─────────┐   ┌────────────┴───────────┐   ┌──────────────────┐
   │ llamacpp / dflash  │   │  lamu-media (feature)  │   │  cloud routing   │
   │  megakernel (LLM)  │   │  ImageBackend(ComfyUI) │   │  (MiMo/DeepSeek) │
   └────────────────────┘   │  TtsBackend(fish-speech)│  └──────────────────┘
                            └────────────┬───────────┘
                                         │ HTTP (localhost)
                            ┌────────────┴───────────┐
                            │ ComfyUI :8188 (Python) │   ← external processes
                            │ fish-speech :8080      │     LAMU spawns + owns
                            └────────────────────────┘
```

Reuse, don't rebuild:
- **`Backend` lifecycle** (`spawn → health-poll → warmup → proxy port → unload`)
  generalizes cleanly. A modality backend is "spawn a server, wait for health,
  register its port, proxy requests, kill on evict." The recent `#16`
  crash-detect-in-poll fix and `#8/#25` process-group reaping apply directly.
- **`VramScheduler`** already does plan_load / LRU eviction by `vram_mb`. Add a
  `modality` tag so eviction policy can be modality-aware (e.g. *pin* the main
  chat LLM, evict image models first) but the accounting is unchanged.
- **MCP dispatch** (`tools.rs` table + `HandlerKind`) — new tools are new rows.
  The `cloud: bool` routing-gate field already exists; media tools are `cloud:
  false` (local-only by nature).

---

## 3. The modality-backend seam

### 3.1 Registry change (`lamu-core/src/types.rs`)
Add to `ModelEntry`:

```rust
pub enum Modality { Llm, Image, Tts }   // serde default = Llm (back-compat)

pub struct ModelEntry {
    // ... existing ...
    #[serde(default)]
    pub modality: Modality,
    // existing vram_mb, port, backend, path, main, pinned, sampling all reused
}
```

`Modality::default() == Llm` keeps every existing `models.yaml` valid.

### 3.2 Backend dispatch (`lamu-core/src/backends/mod.rs`)
`make_backend(entry)` already switches on `entry.backend`. Extend to switch on
`entry.modality` first:

```rust
match entry.modality {
    Modality::Llm   => /* existing llamacpp/dflash/megakernel */,
    Modality::Image => Box::new(lamu_media::ComfyBackend::new(entry)?),
    Modality::Tts   => Box::new(lamu_media::FishSpeechBackend::new(entry)?),
}
```

The `Backend` trait stays the same for spawn/health/unload. Generation is NOT
`generate(prompt) -> text` for media, so we either:
- (a) add modality-specific methods behind `Any` downcasting, or
- (b) **keep `Backend` purely about *lifecycle*** (spawn/health/port/unload) and
  do generation by **proxying to the registered port** from the MCP handler —
  the same way `lamu-api` already proxies LLM chat to a backend port instead of
  calling `generate()` in-process.

**Recommendation: (b).** It's how the HTTP path already works and avoids
widening the `Backend` trait with modality-specific signatures. `lamu-media`
exposes thin client fns (`comfy::txt2img(port, req)`, `fish::tts(port, req)`)
that the MCP handlers call after `ensure_loaded`.

### 3.3 Crate (`lamu-media`, feature-gated)
```
lamu-media/
  src/
    lib.rs          # ImageBackend / TtsBackend + clients
    comfy.rs        # ComfyUI spawn + workflow templating + /prompt client
    fish_speech.rs  # fish-speech spawn + /v1/tts client
    workflows/      # parameterized ComfyUI graph JSON per model family
```
In the workspace, `media = ["dep:lamu-media"]` on `lamu-mcp` / `lamu-cli`,
default OFF. Default build never compiles it (verify: `cargo tree` shows no
media deps) — same discipline as `turbovec`.

---

## 4. Image generation — ComfyUI backend

**Why ComfyUI:** one server backs SDXL / SD3.5 / Pony / Flux via swappable
checkpoints + node graphs; mature HTTP API; huge community workflow library.
LAMU spawns one ComfyUI, points it at a weights dir, and proxies.

### Lifecycle
- **Spawn:** `python main.py --port <P> --output-directory <tmp>
  [--lowvram|--medvram]` from a configured ComfyUI install dir, with
  `CUDA_VISIBLE_DEVICES` set. Hardened child command (PDEATHSIG + process
  group), same as `llama-server`.
- **Health:** poll `GET /system_stats` (or `/` ) until 200.
- **VRAM:** ComfyUI loads the checkpoint lazily on first `/prompt`. Two options:
  (i) charge the registry's declared `vram_mb` up front (simple, slightly
  conservative), or (ii) warm-load one tiny generation at spawn so VRAM is
  resident + measurable via NVML `query_gpu_pids` (matches the LLM warmup path).
  **Start with (i).**
- **Generate:** `POST /prompt` with a **workflow JSON** (templated per model
  family from `workflows/`), poll `GET /history/<id>`, fetch image bytes via
  `GET /view`. Return base64 or a written file path.
- **Unload / evict:** ComfyUI has `POST /free` (unload models, free VRAM)
  without killing the process — use it for soft eviction; SIGTERM the process
  for a hard evict. Lets the scheduler reclaim VRAM for an LLM without paying
  full respawn cost on the next image request.

### Model detection
Scan `<comfyui>/models/checkpoints/**` (and `unet/`, `clip/`, `vae/` for Flux)
→ synthesize `ModelEntry { modality: Image }` rows, the same "detect models in a
folder" UX requested for LLMs. `model_id` = checkpoint filename; the workflow
template is chosen by a `family: sdxl|sd35|pony|flux` hint (from filename or an
explicit field).

### Open question
ComfyUI workflows are verbose graph JSON. We template a *minimal* txt2img graph
per family with holes for `{prompt, negative, steps, cfg, width, height, seed,
sampler}`. Advanced users can drop a custom workflow file and reference it by
name. (Decision needed: how much workflow surface to expose in v1 — see §9.)

---

## 5. Text-to-speech — fish-speech backend

- **Spawn:** fish-speech's API server (`python -m tools.api_server
  --listen 0.0.0.0:<P> --llama-checkpoint-path ... --decoder-checkpoint-path
  ...`), hardened child. (Exact flags pinned to the installed fish-speech
  version at wiring time.)
- **Health:** poll the server's health/docs route until 200.
- **Generate:** `POST /v1/tts` with `{text, reference_audio?, format}` → audio
  bytes (wav/mp3). Voice cloning via an optional reference clip path.
- **VRAM:** declared `vram_mb` (fish-speech ~2–4 GB); evictable like any other.
- **Future TTS:** Kokoro (tiny, fast, no clone), XTTS (multilingual clone),
  Piper (CPU). Each is another `TtsBackend` variant or a generic
  "OpenAI-/HTTP-compatible TTS server" adapter.

---

## 6. MCP + HTTP surface

### MCP tools (in `tools.rs`, `cloud: false`)
```
generate_image {
  prompt, negative?, model?, width?, height?, steps?, cfg?, seed?, n?,
  output: "path" | "base64"
} -> { images: [path|b64], seed, model, elapsed_s }

text_to_speech {
  text, model?, voice_ref?, format: "wav"|"mp3"
} -> { audio: path|b64, model, elapsed_s }

list_media_models {}              -> image + tts entries with vram + loaded state
```
Each handler: `ensure_loaded(model)` (drives the scheduler → may evict an LLM)
→ proxy to the backend port → return result. Routing-mode gate: media tools are
local-only; under `local-only` they run, under `cloud-only` they refuse (no
cloud media providers wired in v1).

### Optional HTTP compat (`lamu-api`)
- `POST /v1/images/generations` (OpenAI Images shape) → ComfyUI.
- `POST /v1/audio/speech` (OpenAI Audio shape) → fish-speech.
This makes LAMU a drop-in OpenAI-compatible *media* endpoint too, not just chat
— low extra cost once the MCP handlers exist. Defer to a second milestone.

---

## 7. VRAM coexistence — the load-bearing integration

The scheduler must treat a 6 GB SDXL load and a 4 GB fish-speech load as
first-class VRAM citizens alongside LLMs:

- `plan_load(image_entry)` may return an LLM as the eviction victim, and vice
  versa. Add **modality-aware eviction preference**: pin the `main` chat LLM,
  prefer evicting image/tts before LLMs (configurable). Falls back to LRU.
- Image gen is **bursty** (load → generate → idle); LLMs are **resident**. A
  short idle-TTL auto-evict for image/tts (configurable) keeps VRAM free for
  chat between image requests, using ComfyUI `POST /free` for cheap soft-evict.
- All of this rides on the *existing* NVML accounting + `query_gpu_pids` drift
  reconciliation; no new VRAM bookkeeping.

This is the piece no single engine gives you and the reason it belongs in LAMU.

---

## 8. Models & licensing (user-accepted, LAMU-managed)

LAMU manages weights the user downloads; it bundles nothing. Flagged so the
user decides before wiring:

| Model     | License                              | Commercial use            |
|-----------|--------------------------------------|---------------------------|
| SDXL 1.0  | CreativeML OpenRAIL++-M              | Yes (with use-restrictions) |
| SD 3.5    | Stability AI Community License       | Free under ~$1M annual rev |
| Pony (XL) | OpenRAIL (SDXL-derived)              | As SDXL                   |
| Flux.2 [dev] | FLUX.2 [dev] Non-Commercial License | **Non-commercial only**   |
| fish-speech | check current repo license (CC-BY-NC-SA historically) | **verify** before commercial use |

LAMU surfaces the declared license per model in `list_media_models` so it's
visible, never silently assumed.

---

## 9. Milestones

1. **M0 — seam (framework only).** `Modality` enum + registry field;
   `make_backend` dispatch; `lamu-media` crate skeleton behind `media` feature;
   scheduler `modality` tag + modality-aware eviction preference. No engine
   wired. Fully unit-testable with a fake modality backend.
2. **M1 — ComfyUI image.** `ComfyBackend` spawn/health/`/free`; one templated
   SDXL workflow; `generate_image` MCP tool; checkpoint folder scan. End-to-end:
   `generate_image` evicts an LLM if needed, renders, returns a path.
3. **M2 — fish-speech TTS.** `FishSpeechBackend`; `text_to_speech` MCP tool.
4. **M3 — breadth.** SD3.5 / Pony / Flux workflow templates; custom-workflow
   passthrough; OpenAI `/v1/images/generations` + `/v1/audio/speech` HTTP compat;
   idle-TTL auto-evict; more TTS engines.

Each milestone is its own reviewed commit chain (MiMo per-commit), green tests
at every step, feature-gated so `main`'s default build is untouched.

## 10. Open questions

1. **ComfyUI install** — assume a user-provided ComfyUI checkout path in config,
   or have LAMU bootstrap it (git clone + venv)? (Lean: user-provided path in
   v1; bootstrap is a later convenience.)
2. **Workflow surface in M1** — single fixed txt2img template per family, or
   expose sampler/scheduler/LoRA knobs immediately? (Lean: fixed template +
   core knobs; custom-workflow passthrough in M3.)
3. **Output handling** — write files to a configured media dir (return paths) vs
   inline base64 in the MCP result. (Lean: files by default — base64 bloats the
   MCP transcript; opt-in base64 for programmatic callers.)
4. **Eviction policy default** — pin main LLM + evict media first, or pure LRU?
5. **Process count** — one ComfyUI for all image models (swap checkpoints) vs one
   per family. (Lean: one ComfyUI, swap via `/free` + reload — fewer processes,
   the scheduler still sees one image backend's VRAM.)
