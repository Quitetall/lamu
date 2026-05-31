# LAMU vs Odysseus — Candid Comparative Assessment

## Positioning (read this first)

These are **different categories** and a single combined score would be dishonest.

- **Odysseus** is a self-hosted, multi-user **AI workspace** — a ChatGPT/Claude-style web app you run on your own hardware. Stack: FastAPI + uvicorn, SQLAlchemy/SQLite, a build-step-free ~129K-LOC hand-written vanilla-JS frontend, ~79K LOC Python backend. It covers chat, an autonomous agent, deep research, email/calendar/contacts, documents, notes/tasks, image gen, and admin-driven model serving.
- **LAMU** is a ~30.7K-LOC **Rust serving backend + MCP orchestrator** with no UI, no email, no multi-user. Its job: own `llama-server` lifecycle, bin-pack one GPU's VRAM via NVML, expose OpenAI/Anthropic/Ollama-compatible HTTP, and offload review/workhorse/compare work to cloud models.

A fair comparison covers **only the overlapping subsystems**: model serving + lifecycle, VRAM scheduling, cloud routing/fallback, model-fit advisory, multi-model compare vs council, memory/retrieval, agent + MCP tooling, and security/sandboxing. LAMU lacking a UI / email / calendar / deep-research is **out of scope by design** — positioning, not a loss. Conversely, Odysseus owning no GPU runtime **is** a real gap on the serving axis where the two genuinely overlap.

All claims below were verified against source in `/home/brianklam/local-llm/`.

## Scorecard (overlapping axes only)

| Dimension | Winner | One-line rationale |
|---|---|---|
| Model serving + lifecycle | **LAMU** | LAMU owns processes with crash-safety; Odysseus is fire-and-forget tmux + regex-scraped status |
| VRAM scheduling / eviction | **LAMU** | Real NVML bin-packer + modality-tiered LRU; Odysseus has no GPU scheduler at all (single-GPU caveat) |
| Cloud routing + fallback | Tie | Odysseus broader (per-role + circuit breaker); LAMU more cost-engineered (cache warmup, tier-gated thinking) |
| Model-fit advisory (cookbook) | **Odysseus** | hwfit roofline engine over 898 models vs LAMU's 3-bucket heuristic over a curated list |
| Compare vs council/review | **LAMU** | Automated ensemble+critic+FP-filter synthesis vs human-only blind voting (Odysseus's A/B is a better *human* instrument) |
| Memory / retrieval | Tie | LAMU temporally correct (valid-time/supersede); Odysseus better hybrid BM25+vector fusion |
| Agent + MCP tooling | Tie | Odysseus is the end-user agent; LAMU is the backend other agents drive — both well-built for their role |
| Agent execution sandboxing | **LAMU** | bubblewrap (net-off, workdir-only-write) vs Odysseus's unsandboxed host shell (`cwd=$HOME`) |
| HTTP server security / auth | **Odysseus** | Full 2FA/Fernet/CSP/bearer stack vs LAMU's zero auth behind a loopback bind |
| Prompt-injection boundary | **Odysseus** | Centralized, tested `UNTRUSTED_SOURCE_DATA` envelope vs LAMU's per-surface hardening |
| Code quality / maintainability | Tie | Both strong solo projects with characteristic debt (details below) |

## Where LAMU is genuinely better

1. **Cross-modal VRAM scheduling** (`lamu-core/src/scheduler.rs`). The modality-tiered LRU bin-packer evicts idle TTS/image models before LLMs and computes `available_mb = max(registered, NVML-actual)` so orphan servers can't be handed out as free VRAM. **Odysseus has no GPU scheduler whatsoever** — verified; the only LRU in the codebase is a search-result cache. This is LAMU's reason to exist and Odysseus structurally cannot match it because it doesn't own the model processes.
2. **Process lifecycle + crash-safety** (`backends/mod.rs`, `loader.rs`). PDEATHSIG with a `getppid()==1` fork-race re-check, scheduler rollback on spawn failure, single-flight load gate. Odysseus launches engines into tmux and regex-scrapes `capture-pane` for state (`_parse_serve_phase`) — brittle to engine log-format drift and unable to cleanly evict.
3. **Agent sandboxing** (`sandbox/launcher.rs`). bubblewrap with `--unshare-net`, `--clearenv`, workdir as the only writable bind, HOME repointed; firejail fallback with `--private-etc` allowlist. Odysseus's `ShellService` runs raw `create_subprocess_shell` with `cwd=$HOME` (verified) — a prompt-injection against an admin session is host RCE by design.

## Where Odysseus is genuinely better

1. **hwfit fit-scoring engine** (`services/hwfit/fit.py`, 463 LOC, verified). MoE active-param speed estimate, a GPU memory-bandwidth roofline table (~70 GPUs), GGUF-single-GPU vs sharded-prequant VRAM logic, context-halving budget search, prequant bit-width matching, over a vendored 898-model DB, with `hardware.py` grouping identical GPUs for tensor-parallel. **LAMU's `fit_bucket` is a 3-state heuristic** (FitsNow/AfterUnload/TooBig) over a small curated list. Not close — this is the biggest single capability gap and the top port candidate.
2. **HTTP auth / security stack** (`core/auth.py`, `src/secret_storage.py`, `core/middleware.py`). bcrypt + TOTP 2FA + single-use backup codes, `ody_` bearer tokens (bcrypt + cache), loopback-only internal-tool token, Fernet-encrypted secrets with idempotent migration, CSP nonces, login rate-limiting, deleted-user session purge — pinned by `test_security_regressions` / `test_auth_regressions`. **LAMU has zero auth**: it is secure only because it binds `127.0.0.1`, and its own code notes binding `0.0.0.0` "exposed an unauthenticated API from day one."
3. **Centralized prompt-injection envelope** (`src/prompt_security.py`). All retrieved/tool/web/memory content is wrapped in an `UNTRUSTED_SOURCE_DATA` envelope demoted from system to user role, and regression-tested. LAMU's injection hardening is per-surface (git refs, paths) rather than one uniform boundary.

## Honest liabilities on both sides

**LAMU**
- Single-GPU by construction: `scheduler.rs` only ever reads `device_by_index(0)` (lines 25/76/92). The celebrated bin-packer is one GPU only.
- Hardcoded developer-machine paths: `megakernel.rs:38-40` and `config.rs:28` point at `~/local-llm/...`. Several backends are unusable off the author's machine without editing source.
- A **load-bearing stale comment** ("Streaming SSE is not yet implemented — non-streaming only", `openai_compat.rs:898`) sits directly above working streaming code — exactly the drift the project claims to forbid.
- Lossy Anthropic `tool_use`/`tool_result` → prose bridging drops `tool_call_id`; multimodal image blocks silently dropped.
- The review-every-commit loop relies on cloud reviewers with an admitted ~30% FP rate; quality is unverifiable from inside.

**Odysseus**
- Single squashed git commit — no history for regression archaeology.
- ~1189 `except Exception` blocks (16 bare `except: pass`) swallow failures app-wide.
- God-files: `tool_implementations.py` (4035), `agent_loop.py` (2106), `email_routes.py` (3038).
- **Live drifted duplication**: `services/search/` and `src/search/` are ~20-line-drifted forks, **both imported** (11 + 7 sites verified). A fix in one won't propagate.
- **Dead/broken facade**: `services/memory/service.py` `MemoryService` calls `add_memory`/`search_memories`/`get_memories`, which **do not exist** (real API is `add_entry`/`get_relevant_memories`). Verified.
- Serve-state detection is regex over tmux scrollback — brittle across vLLM/llama.cpp versions.

## Philosophy contrast

Odysseus is **batteries-included product engineering**: delegate execution to the environment (tmux, SSH, host shell, cloud APIs), reinvent nothing it can borrow (ChromaDB, SearXNG, opencode patterns, llmfit-derived fit logic, Tongyi-DeepResearch loops), and spend its own effort on integration, UI, auth, and a hard-won agent prompt. Threat model: trust the single admin; power is privilege-gated, not sandboxed; ethos is degrade-don't-crash. Cost: it owns no runtime — no VRAM management, no eviction, regex-scraped state, and unsandboxed host RCE by design.

LAMU is **lean composable-backend engineering**: "manage processes, don't reimplement engines" — but it *does* own the one runtime that matters for its niche (VRAM bin-packing, model lifecycle with crash-safety, bubblewrap-sandboxed agents). Threat model: loopback-only single-user, so zero auth and trust the network boundary; ethos is fail-loud and "one definition can't drift." Cost: narrowness (single GPU, single user, Linux/CUDA-only, hardcoded paths) and a self-referential quality loop.

Neither is wrong; they optimize for different owners of different problems.

## Bottom line

On the overlapping mission, **LAMU wins the three things that are its reason to exist** — VRAM scheduling with cross-modal eviction, model-process lifecycle with crash-safety, and agent sandboxing — and Odysseus structurally cannot match the first two. **Odysseus wins fit-advisory and HTTP security/auth**, both of which LAMU under-invests in. The rest are honest ties with complementary strengths.

So: we are **better on the core serving/scheduling/sandboxing mission and worse on fit-advisory and on any deployment that leaves loopback.** The actionable conclusions: (1) port hwfit's roofline fit-scoring — the gap is real and embarrassing for a tool that ships a `cookbook` command; (2) adopt Odysseus's dead-host circuit breaker and a uniform untrusted-content envelope; (3) add a minimal bearer-token so `LAMU_BIND_HOST=0.0.0.0` isn't an instant open endpoint; and (4) fix the narrowness/hygiene tells (single-GPU assumption, hardcoded `~/local-llm` paths, the stale streaming comment) before claiming the code is more disciplined than Odysseus's — on those specific points it currently isn't.
