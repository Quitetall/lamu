# LAMU bug audit — 2026-06-01 (adversarial hunt)

Full-tool adversarial bug hunt: 16 finder surfaces → dedup → refute-by-default verify.
**57 candidates → 51 confirmed (42 after de-dup) → 6 refuted.** Severity is the verifier's
corrected severity. Items marked ✅ FIXED have landed; the rest are open, ranked by impact.

## FIXED — 38 of 42 (all blockers + all 16 majors + 20 of 24 minors)

Blockers: **B1** tool_calls passthrough, **B2** cancellation-safe ensure_loaded.
Majors (all 16): **M1** backend-status propagation, **M2** anthropic stop_reason,
**M3** random_hex→getrandom, **M4** ollama num_predict negative, **M5** strip
`<think>` from ollama/anthropic streams, **M6** charge stream reserve after
resolve, **M7** immediate cross-process revoke, **M8** python backends bind
loopback, **M9** per-device eviction deficit, **M10** router capability bypass,
**M11** deterministic main, **M12** backend generate() status, **M13** OpenRouter
provider label, **M14** api-keys.env 0600, **M15** model-label cardinality clamp,
**M16** restart/quarantine counters wired.
Minors (20): m1 embeddings charged, m2 gateway model, m3 anthropic tool_result
array, m5 ollama done_reason, m8 dedup GPU indices, m9 cross-device placement
remove, m10 port-exhaustion Option, m11 Backend::stream status, m12 GGUF
ftype→quant table, m13 write_atomic dir fsync, m14 GPU_INDEX parse, m15 router
tie determinism, m16 empty-model match, m18 parallel_query thinking key, m19
truncated-thinking error flag, m20 media symlink canonicalization, m21 MCP
empty-key reject, m22 routing-mode validation, m23 failure-metric attribution,
m24 gauge series leak.

Plus a separate `feat`: the TUI model selector is now grouped by provider.

## OPEN — 4 minors (low-value / design-nuanced)

- **m4** streaming quota reserve uses raw `max_tokens`, not the resolved/locked
  sampler-profile `eff_max_tokens` (minor billing precision when a profile
  overrides the cap).
- **m6** OpenAI streaming buffers ≤21-char outputs until end-of-stream (a tiny
  tagless reply arrives in one final chunk; touches the reasoning-tag state
  machine — deferred to avoid regressing the untested OpenAI stream).
- **m7** `/metrics` is unauthenticated (standard for Prometheus) and carries the
  `user` label, so off-loopback it exposes the key roster. Design call: requiring
  auth breaks scrapers, dropping the label undoes P2b — recommend firewalling
  `/metrics` off-loopback, or a future `LAMU_METRICS_USER_LABEL=0` opt-out.
- **m17** non-TTY `lamu repl <url>` fallback sends `Bearer sk-local` + OpenAI
  wire-format, ignoring a resolved cloud key/provider (niche: piped REPL against
  a cloud URL with a config api_key).

| Sev | File:line | Bug | Trigger |
|-----|-----------|-----|---------|
| blocker | loader.rs:190 | ensure_loaded is not cancellation-safe: a dropped load future leaks a permanent Loading entry AND an orphaned llama-server subprocess | POST /v1/chat/completions for an unloaded model, then disconnect the client during the cold load (llama-server load+warmup is up to ~90s). Axum drops the handle |
| blocker | openai_compat.rs:1500 | Non-streaming /v1/messages drops all tool_calls (turns tool-only completions into a 502) | POST /v1/messages with stream:false and a `tools` array, where the model decides to call a tool (e.g. Claude Code's non-streaming tool agent loop). Backend emit |
| major | cloud_sync.rs:51 | cloud sync mislabels OpenRouter anthropic/* models as provider="anthropic", causing every such model to be sent in Anthropic wire-format (x-api-key + /v1/messages) to OpenRouter and fail | Run `lamu cloud sync` (pulls the OpenRouter catalog, auto-adding anthropic/* rows), then call `cloud_query({model: "claude-opus-4.8"})` (or open that row in the |
| major | dflash.rs:84 | dflash & megakernel backends bind their inference server to 0.0.0.0 (all interfaces) with no opt-out and no auth | Load any model whose registry entry has `backend: dflash` (or `dflash_lucebox`) or `backend: megakernel` on a host with a routable/LAN interface. The spawned in |
| major | keys.rs:374 | Key revocation is NOT immediate for a running `lamu serve` — cross-process cache incoherence; CLI falsely reports immediacy | Operator runs `lamu serve` (KeyStore mode, >=1 active key). A key leaks / an employee leaves. Operator runs `lamu auth revoke lamu_<prefix>`; CLI prints 'it sto |
| major | main.rs:862 | `lamu scan` blindly overwrites registry, destroying all operator customizations (sampling, notes, pinned, status, main, backend, capabilities) | Operator edits models.yaml to set `sampling.lock: true`/`temperature: 0.0`, a `main:` choice, `status: recommended`, or a backend for a safetensors entry, then  |
| major | metrics.rs:93 | backend_restarts_total and backend_quarantined_total are registered + documented but never incremented (always 0) | Cause a backend to crash and be supervisor-restarted, or drive a backend to 5 consecutive errors so record_error sets HealthState::Quarantined, then scrape /met |
| major | openai_compat.rs:1038 | random_hex is a truncated timestamp, not random — colliding completion/message/tool IDs | Two concurrent /v1/chat/completions (or /v1/messages) requests arriving within ~65us, OR a single streaming /v1/messages response where the backend emits >=2 to |
| major | openai_compat.rs:1389 | Streaming /v1/messages and /api/chat charge the full max_tokens reserve BEFORE model resolution — failed/unloadable models permanently burn quota with no refund | A KeyStore-authenticated user (daily_token_quota=Some(n)) sends a streaming POST /v1/messages (Claude Code streams by default) or POST /api/chat with model set  |
| major | openai_compat.rs:1530 | Anthropic stop_reason never reports max_tokens (silent truncation reported as end_turn) | POST /v1/messages with a small max_tokens (or a long generation that hits the 16384 default) so the backend returns finish_reason:"length". Client sees stop_rea |
| major | openai_compat.rs:1757 | Synthesized tool_use ids collide within a single response (random_hex keeps slow-changing high bits) | Streaming POST /v1/messages where the model emits two or more tool calls in one turn and the OpenAI-compat backend omits the `id` field on the tool_call deltas  |
| major | openai_compat.rs:1792 | Anthropic streaming/non-streaming stop_reason hardcoded to end_turn — max_tokens truncation reported as a clean finish | POST /v1/messages with `"stream": true` (or non-stream) and a max_tokens small enough that the model is truncated mid-answer, e.g. {"model":"qwen3.6-27b","max_t |
| major | openai_compat.rs:1848 | Ollama `num_predict: -1` / `-2` (infinite / fill-context sentinels) fail to deserialize, rejecting the entire request | POST /api/chat with body `{"model":"lamu","messages":[{"role":"user","content":"hi"}],"options":{"num_predict":-1}}` (the default many native Ollama clients sen |
| major | openai_compat.rs:1848 | Standard Ollama num_predict:-1 (and other negative options) 422s with a non-Ollama plain-text body | POST /api/chat with body {"model":"lamu","messages":[{"role":"user","content":"hi"}],"options":{"num_predict":-1}} from any Ollama client (AnythingLLM / Open We |
| major | openai_compat.rs:2081 | Ollama streaming path leaks raw <think> reasoning into visible message content (and all three streaming paths drop reasoning_content), unlike the OpenAI/Ollama non-streaming paths | POST /api/chat (Ollama surface, default stream=true) against a reasoning model with default enable_thinking, e.g. {"model":"qwen3.6-27b","messages":[{"role":"us |
| major | openai_compat.rs:2087 | Ollama streaming path leaks raw `<think>...</think>` reasoning blocks into the assistant message | POST /api/chat with `stream` omitted (defaults true) or `stream:true`, against a thinking-capable model with no `enable_thinking` field, e.g. `{"model":"lamu"," |
| major | openai_compat.rs:378 | Unbounded `model` label cardinality: client-supplied model string flows verbatim into requests_total | Any client (authenticated, or unauthenticated on loopback) sends a stream of POST /v1/chat/completions with a fresh random `model` each time, e.g. {"model":"jun |
| major | openai_compat.rs:500 | Unbounded Prometheus label cardinality: raw client-supplied `req.model` used as metric label on quota_exceeded / gpu_locked early returns (before registry validation) | KeyStore mode: an authenticated user whose daily_token_quota bucket is exhausted (so every request hits the quota_exceeded branch) sends a stream of POST /v1/ch |
| major | openai_compat.rs:693 | Backend HTTP error status is swallowed; real 4xx/5xx becomes generic 502 backend_returned_empty | POST /v1/chat/completions with messages whose prompt exceeds the loaded model's context_max, so llama-server replies 400 with a JSON error body. |
| major | router.rs:127 | model=None main-model preference ignores requested capabilities (capability-filter bypass) | mcp__local-llm__query({ model: null, capabilities: ["vision"] }) (or ["embedding"], ["reasoning"]) while a text-only main model is loaded and healthy. handlers. |
| major | router.rs:128 | Nondeterministic main resolution in the no-model path (uses find() instead of lowest-name like the alias path) | models.yaml with two entries flagged `main: true` (the YAML permits this; auto_promote_main only acts when none are flagged). A frontend calling /v1/chat/comple |
| major | scheduler.rs:297 | Concurrent same-model load_model double-spawns on the same port (is_loaded ignores Loading state) | Two MCP clients (or one client firing parallel tool calls) invoke load_model({name:"qwen35-27b"}) for the same not-yet-loaded model within the spawn window (~te |
| major | scheduler.rs:356 | plan_load rejects a model as VramExhausted on multi-GPU even when eviction on a single device would make room | Run with LAMU_GPU_INDICES=0,1 (two managed GPUs), each device partially occupied by an evictable Loaded model so e.g. device0 free=5000MB, device1 free=5000MB ( |
| major | scheduler.rs:363 | plan_load uses aggregate available_mb() for deficit but a model must fit on ONE device — wrong eviction decisions on multi-GPU | load_model for a model that fits on no single GPU but whose vram is <= aggregate free across the multi-GPU pool (LAMU_GPU_INDICES with >=2 devices, each partly  |
| major | settings.rs:64 | TUI save_api_key writes ~/.config/lamu/api-keys.env without 0600 perms — API keys created world/group-readable | Open `lamu`, select a cloud model, press 'a', paste an API key, press Enter — when ~/.config/lamu/api-keys.env did not already exist (or its perms were loosened |
| minor | auth.rs:90 | /metrics is unauthenticated and exposes per-user `user` labels (username roster + per-user request/token counts) | Off-loopback KeyStore deployment (LAMU_BIND_HOST=0.0.0.0 + keys.db with active keys). Any host on the network runs `curl http://<host>:<port>/metrics` with NO A |
| minor | cloud.rs:259 | MCP cloud_query accepts an exported-but-empty API key and sends a blank Bearer/x-api-key, producing a confusing 401 (regression of fix #32, which only patched the TUI/config layer) | `export DEEPSEEK_API_KEY=` (set but empty), then `cloud_query({model: "deepseek-v4-flash", prompt: "hi"})`. |
| minor | config.rs:109 | LAMU_GPU_INDEX typo diverges: backends spawn on device 0 while scheduler manages zero devices | Start lamu with a malformed LAMU_GPU_INDEX (typo, leading sign, or a name instead of an index): backends launch on GPU 0 but the VRAM scheduler tracks no device |
| minor | handlers.rs:281 | handle_query returns a non-error 'thinking truncated' string that is not flagged isError, masking an empty answer | query (or a local parallel_query/council task) against a reasoning model where the response is all <think> and gets truncated before any answer token, with incl |
| minor | handlers.rs:758 | parallel_query drops the per-task thinking toggle for LOCAL tasks (wrong arg name) | mcp__local-llm__parallel_query({tasks:[{prompt:"...", model:"<local model>", thinking_enabled:false}]}) — the local task runs with thinking ON despite thinking_ |
| minor | llamacpp.rs:459 | Backend generate() ignores HTTP status code — server error envelopes are masked as 'missing choices[0].message' | Call the local-LLM `query`/inference tool with a prompt that overflows the model's context, or hit the backend while it returns a 4xx/5xx (e.g. the server repli |
| minor | llamacpp.rs:487 | Backend::stream() ignores HTTP status code — non-2xx responses become a silently-empty successful stream | Invoke the trait `stream()` path against a backend that returns 400/503 (e.g. context overflow or busy slot). The consumer receives a successful, empty stream w |
| minor | loader.rs:79 | pick_backend_port returns a possibly-occupied PORT_SIDECAR on candidate exhaustion, causing bind failure or cross-model port aliasing | Load >9 distinct models concurrently (feasible on a multi-GPU rig per ADR 0017) so all 10 candidate ports are occupied, then request a further model. pick_backe |
| minor | media_paths.rs:37 | Media output confinement lacks the symlink/canonicalization check that write_file has | text_to_speech/generate_image with output_path='link/x.png' where 'link' is (or becomes) a symlink under <data_dir>/lamu/tts (or images) pointing outside the co |
| minor | metrics.rs:147 | Gauge leak: vram_used_mb and queue_depth keep stale per-model series after a model is unloaded | Load model A (vram_used_mb{model="A"} becomes e.g. 18000), unload A (MCP unload / eviction), then scrape /metrics. lamu_vram_used_mb{model="A"} still reads 1800 |
| minor | openai_compat.rs:1305 | Anthropic tool_result content arrays injected as raw JSON instead of extracted text | POST /v1/messages with a user message containing a tool_result block whose content is an array of text blocks (Claude Code's standard tool-result shape), e.g. { |
| minor | openai_compat.rs:1902 | Streaming quota reservation ignores the per-model sampling profile's max_tokens, under-charging when a profile raises the cap | POST /api/chat with `stream:true`, no `options.num_predict`, against a model whose registry `sampling` profile sets a large `max_tokens` (esp. with `lock:true`) |
| minor | openai_compat.rs:2116 | Ollama responses hardcode `done_reason: "stop"` and `prompt_eval_count: 0`, misreporting truncation and prompt size | POST /api/chat with `stream:true` and a small `num_predict` so the backend truncates at `length`; the NDJSON `done` frame reports `done_reason:"stop"` and `prom |
| minor | openai_compat.rs:271 | /v1/embeddings checks quota but never charges it — metered keys get unlimited embeddings | KeyStore mode with a key that has daily_token_quota set; repeatedly POST /v1/embeddings — the bucket never goes down. |
| minor | openai_compat.rs:403 | requests_total mis-attributes early-failure requests to user="anon" even for authenticated principals | An authenticated user (Principal with user="alice") sends a chat request whose model can't fit in VRAM or fails to spawn; the resulting lamu_requests_total{..., |
| minor | openai_compat.rs:634 | LAMU_GATEWAY_URL path forwards no model when request omits model — gateway cannot route | Set LAMU_GATEWAY_URL=http://bifrost:PORT, then POST /v1/chat/completions with a body that omits the model field (e.g. Claude Code default, or any harness relyin |
| minor | openai_compat.rs:650 | Streaming reserve charges the raw request max_tokens, ignoring the resolved/locked sampler profile max_tokens that is actually sent to the backend | Issue a streaming request to a model that has a locked sampling profile whose max_tokens differs from the client-supplied (or default) value. e.g. locked profil |
| minor | openai_compat.rs:909 | OpenAI streaming generator buffers short (<22-char) outputs and never streams them incrementally — entire reply arrives in one chunk at end-of-stream | POST /v1/chat/completions with "stream": true, "enable_thinking": false and a prompt that yields a short answer (<=21 chars), e.g. messages=[{"role":"user","con |
| minor | registry.rs:188 | GGUF file_type → quant mapping is wrong (IQ2_XXS/IQ2_XS/Q2_K_S mislabeled as Q2_K/Q3_K_S/Q3_K_M) and misses Q5_0/Q5_1/Q2_K/Q3_K_* | `lamu scan` over a directory containing any IQ2_XXS/IQ2_XS/Q2_K_S/Q2_K/Q3_K_*/Q5_0/Q5_1 GGUF. e.g. an IQ2_XXS .gguf with an opaque filename → registry records q |
| minor | registry.rs:583 | write_atomic claims crash safety but never fsyncs the parent directory after rename | Power loss / kernel panic in the window after std::fs::rename returns but before the filesystem flushes the directory inode (e.g. ext4 default 5s commit interva |
| minor | repl.rs:281 | Non-TTY REPL fallback always sends bearer 'sk-local' and OpenAI wire-format, ignoring the resolved cloud key/provider | lamu repl https://api.deepseek.com/v1/chat/completions / tee out.txt  (stdout not a TTY) then type any prompt; or any cloud chat whose stdout is redirected. The |
| minor | router.rs:179 | Nondeterministic unloaded-candidate selection on equal (capability-count, vram) ties | Two registry models with identical capability sets and identical vram_mb (e.g. two Q4 7B chat models at the same vram_mb) and neither loaded, with a capability  |
| minor | router.rs:208 | find_model substring fallback silently resolves to an arbitrary/wrong model (empty string and partial names) | Client sends `model: ""` to /v1/chat/completions (common from frontends that don't omit the field) against a single-model registry → routed to that model; or `m |
| minor | scheduler.rs:284 | register_loaded does not remove a prior placement of the same model — double-counts VRAM across devices | Call register_loaded twice for the same model name on a multi-device pool whose occupancy changed between the two calls so placement_for resolves to a different |
| minor | scheduler.rs:64 | Duplicate LAMU_GPU_INDICES entries create duplicate DeviceBudgets — VRAM double-counted and same physical GPU over-committed | Run any lamu binary with `LAMU_GPU_INDICES=0,0` (e.g. a copy-paste / scripting error), then load two models that each fit individually. |
| minor | server.rs:98 | LAMU_ROUTING_MODE env value is stored without validation; a typo silently degrades to auto-like behavior | Start the MCP server with `LAMU_ROUTING_MODE=local_only` (underscore) or any misspelling intending to lock to local; cloud tools still execute. |

## Refuted (false positives, not bugs)
- loader.rs:209 — confirm_loaded error silently swallowed leaks a spawned subprocess holding VRAM with no scheduler entry
- lib.rs:91 — serve() binds an IPv4-only socket, so any IPv6 LAMU_BIND_HOST fails to start
- main.rs:1010 — resolve_entry bidirectional substring match treats an empty registry name as matching every query
- rag.rs:339 — index_repo assumes OpenAI /embeddings preserves input order; out-of-order response misaligns embeddings with chunks
- mod.rs:150 — Backend child PR_SET_PDEATHSIG is parent-THREAD-relative, not process-relative — tokio worker-thread exit SIGKILLs a healthy llama-server (VRAM/proc loss)
- cloud.rs:989 — review_commit accepts an attacker-supplied `repo` path used verbatim as git's working directory (arbitrary-directory git execution / info disclosure)
