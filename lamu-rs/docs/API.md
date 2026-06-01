# LAMU HTTP API

LAMU is a lean Rust **backend orchestrator**. It owns the things that are hard to
share — model lifecycle, VRAM, and routing across a pool of local GGUF models —
and exposes that pool behind broadly-compatible HTTP so that **any frontend you
already use becomes the LAMU frontend**.

There is no LAMU UI. You bring the harness: Claude Code, Open WebUI, AnythingLLM,
Continue, LibreChat, a RAG front-end, the OpenAI/Anthropic SDKs, or plain `curl`.
LAMU speaks three HTTP dialects at once — **OpenAI**, **Anthropic Messages**, and
**Ollama** — so whichever client you point at it just works.

> The orchestration/agent plane (council, commit review, lifetime memory, routing
> control, cloud models, TTS/image, fine-tuning) lives on a **separate MCP/stdio
> contract**, not on this HTTP surface. See [MCP is the agent plane](#mcp-is-the-agent-plane).
> The HTTP surface is deliberately "dumb": pure inference + model listing.

---

## Contents

- [The model: backend orchestrator, bring-your-own-frontend](#the-model)
- [Base URL & binding](#base-url--binding)
- [Authentication](#authentication)
- [LAMU extensions](#lamu-extensions)
- [Endpoints](#endpoints)
  - [GET /health](#get-health)
  - [GET /metrics](#get-metrics)
  - [GET /v1/models](#get-v1models)
  - [POST /v1/chat/completions](#post-v1chatcompletions)
  - [POST /v1/embeddings](#post-v1embeddings)
  - [POST /v1/messages](#post-v1messages-anthropic)
  - [GET /api/tags](#get-apitags-ollama)
  - [POST /api/chat](#post-apichat-ollama)
- [Status codes & error envelopes](#status-codes--error-envelopes)
- [Point your frontend at LAMU](#point-your-frontend-at-lamu)
- [Footguns](#footguns)
- [MCP is the agent plane](#mcp-is-the-agent-plane)

---

## The model

LAMU separates two planes cleanly:

| Plane | Crate | Transport | Purpose | Who calls it |
|-------|-------|-----------|---------|--------------|
| **Inference (this doc)** | `lamu-api` | HTTP | Stateless multi-dialect inference proxy over the **local** model pool | Any OpenAI/Anthropic/Ollama client |
| **Agent / orchestration** | `lamu-mcp` | MCP over stdio | council, review, memory, routing, cloud models, TTS/image, training, RAG | An orchestrator (e.g. Claude Code) |

Key consequence: **cloud models are not on HTTP.** `GET /v1/models` and the chat
surfaces serve only the *local* registry (`models.yaml`). DeepSeek / Claude / MiMo
are reached **exclusively** through the MCP `cloud_query` tool, which talks directly
to the provider. The one HTTP escape off-box is `LAMU_GATEWAY_URL` (Bifrost-style),
which an operator wires up explicitly. Asking the OpenAI surface for
`model: "deepseek-v4-flash"` returns `503 model_not_available` unless that name is a
local registry entry.

---

## Base URL & binding

Default bind is **loopback**: `http://127.0.0.1:<port>`.

| Client family | Base URL to configure |
|---------------|-----------------------|
| OpenAI / SDKs / Open WebUI / Continue (OpenAI mode) | `http://127.0.0.1:<port>/v1` |
| Anthropic / Claude Code | `http://127.0.0.1:<port>` (hits `/v1/messages`) |
| Ollama clients (AnythingLLM, Open WebUI Ollama mode) | `http://127.0.0.1:<port>` (hits `/api/tags` + `/api/chat`) |
| RAG embeddings | `http://127.0.0.1:<port>/v1/embeddings` |

**Off-loopback exposure is gated.** Set `LAMU_BIND_HOST=0.0.0.0` and `lamu serve`
**hard-fails at startup unless a token is configured** (`LAMU_API_TOKEN` env or
`~/.config/lamu/api-token`). The explicit, loud escape hatch is
`LAMU_ALLOW_INSECURE=1`. This is ADR 0012.

**CORS** is permissive (any origin, any header, any method; no credentials) and
sits outermost, so browser-based frontends — and their preflight `OPTIONS` —
work even with a bearer token set (preflight is answered before auth). The
Bearer token travels as a header, not a cookie, so `Allow-Origin: *` is safe.

Other runtime facts: IPv4 socket with `SO_REUSEADDR`, backlog 1024; graceful
shutdown on SIGINT/SIGTERM; an RAII pidfile at
`$XDG_RUNTIME_DIR/lamu-serve-{port}.pid` (else `/tmp/...`); backend HTTP timeout
300s.

---

## Authentication

Single static **bearer token**, calibrated to a single-user, loopback-default
threat model. No accounts, sessions, or per-token DB.

- **Auth OFF** when no token is configured — frictionless loopback (the common case).
- **Auth ON** when a token is set: every route **except `/health` and `/metrics`**
  requires `Authorization: Bearer <token>`.
- Token resolution order: `LAMU_API_TOKEN` (trimmed, non-empty) →
  `~/.config/lamu/api-token` (trimmed). Resolved **once** at startup.
- Comparison is **constant-time** (`subtle::ConstantTimeEq`). It short-circuits on
  length mismatch (token length is public by design).
- Header parse is lenient (RFC 7235): scheme is case-insensitive (`bearer` ok),
  whitespace tolerated, split on the first space. An empty presented token fails.

```bash
curl http://127.0.0.1:8020/v1/models \
  -H "Authorization: Bearer $LAMU_API_TOKEN"
```

**401 response** — surface-correct: Anthropic shape on `/v1/messages`, Ollama flat
`{"error":"..."}` on `/api/*`, OpenAI shape elsewhere. OpenAI example:

```
HTTP/1.1 401 Unauthorized
WWW-Authenticate: Bearer
Content-Type: application/json

{"error":{"message":"unauthorized","type":"invalid_request_error"}}
```

---

## LAMU extensions

These are LAMU-specific behaviors layered on the standard dialects:

### Model aliases
Send `model: "lamu"` (also `"main"` / `"default"`, case-insensitive) and LAMU
resolves it to the registry entry flagged `main: true`. Or **omit `model` entirely**
— the router auto-selects (lands on the loaded `main:true` entry when no capability
is requested). This lets a frontend pin a stable name and stay agnostic of which
model actually backs it. Explicit registry names also work. The chosen name is
reflected in the response `model` field, which may differ from what you requested.

### `enable_thinking` (Qwen3.6/3.5 reasoning toggle)
Accepted at the top level on **all three chat surfaces**. When set, LAMU injects
`chat_template_kwargs.enable_thinking` into the backend payload; the Qwen3.6/3.5
chat templates honor it. `false` skips the `<think>` block to shave latency.
**Omit it** and the backend template default applies (typically thinking ON for
Qwen3.6). For non-Qwen model families the flag is forwarded but may be a no-op.

### `<think>` / reasoning handling differs by surface
- `/v1/chat/completions` **non-stream**: reasoning is returned as
  `message.reasoning_content` (only present when non-empty). LAMU prefers the
  backend's structured `reasoning_content`; if absent it splits the raw content on
  the model's marker.
- `/v1/chat/completions` **stream**: reasoning is **stripped and discarded** — never
  emitted to the client (handles tags split across token boundaries).
- `/v1/messages` **stream** and `/api/chat` **stream**: the marker is **ignored** —
  raw token content is forwarded, so literal `<think>...</think>` may appear. On
  these surfaces `enable_thinking: false` is the only reliable suppression.

### VRAM-aware auto-load (and the no-auto-evict rule)
Request a model that isn't loaded and LAMU spawns the `llama-server` subprocess on
a free port (single-flight: N concurrent requests for the same model collapse to
one spawn). **The HTTP path refuses to auto-evict**: if fitting the new model would
require evicting another, it errors rather than silently killing a loaded model
(ADR 0006). A GPU lock held by `lamu-train` returns `503 gpu_locked` before any
VRAM work. `main:true` is fire-and-forget preloaded at startup.

### Sampling profiles
`temperature / top_k / top_p / min_p / repeat_penalty / max_tokens` merge through a
per-model `SamplingProfile` (registry `sampling:` key). Precedence:
**locked-profile field > request value > unlocked-profile field > builtin default**.
`lock: true` pins the operator value over client values. Only set samplers are
emitted downstream (no null fields). Builtin defaults: `max_tokens=16384`,
`temperature=0.7`.

---

## Endpoints

The full route table (all routes except `/health` and `/metrics` carry the bearer
middleware):

| Method | Path | Family |
|--------|------|--------|
| GET  | `/health` | (auth-exempt) |
| GET  | `/metrics` | (auth-exempt) |
| GET  | `/v1/models` | OpenAI |
| POST | `/v1/chat/completions` | OpenAI |
| POST | `/v1/embeddings` | OpenAI |
| POST | `/v1/messages` | Anthropic |
| GET  | `/api/tags` | Ollama |
| POST | `/api/chat` | Ollama |

There is **no** `/v1/completions` (legacy), `/v1/models/{id}`, model-management
routes, or Ollama `/api/generate` / `/api/pull` / `/api/show` / `/api/version` /
`/api/embeddings`. Tools that probe those will 404.

---

### GET /health

Liveness. No auth. Always 200.

```bash
curl http://127.0.0.1:8020/health
```
```json
{"status":"ok","models_loaded":2}
```

---

### GET /metrics

Prometheus scrape. No auth. `Content-Type: text/plain; version=0.0.4; charset=utf-8`.

Series include `lamu_requests_total{model,status}`,
`lamu_request_duration_seconds{model,phase}`,
`lamu_tokens_generated_total{model,kind}`, `lamu_vram_used_mb{model}`,
`lamu_vram_total_mb`, `lamu_queue_depth{model}`,
`lamu_backend_health_state{model}` (2=healthy, 1=degraded, 0=dead, -1=quarantined),
`lamu_backend_restarts_total{model}`, `lamu_backend_quarantined_total{model}`,
`lamu_metrics_scrapes_total`. `status` label values: `ok`, `no_candidate`,
`spawn_failed`, `gpu_locked`, `backend_error`, `backend_empty`.

---

### GET /v1/models

OpenAI model list, augmented with LAMU fields. Lists the **local** registry only.

```bash
curl http://127.0.0.1:8020/v1/models -H "Authorization: Bearer $TOKEN"
```
```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen3.6-27b",
      "object": "model",
      "owned_by": "local",
      "loaded": true,
      "params_b": 27,
      "vram_mb": 18000,
      "capabilities": ["chat","code","reasoning","long_context"]
    }
  ]
}
```

Capability strings: `chat`, `code`, `reasoning`, `routing`, `vision`,
`long_context`, `embedding`. Note: LAMU adds `loaded`/`params_b`/`vram_mb`/
`capabilities` and **omits** OpenAI's `created`. Discover live names here rather
than hardcoding.

---

### POST /v1/chat/completions

OpenAI chat. Streaming and non-streaming.

**Request fields** (`ChatRequest`):

| Field | Type | Notes |
|-------|------|-------|
| `model` | string \| null | Optional. Omit → router picks. Aliases `lamu`/`main`/`default`. |
| `messages` | array | **Required** (omit → 422). `content` may be a string, an OpenAI-Vision parts array (text parts concatenated with `\n`; image/tool parts dropped), `null` → `""`, or other JSON → stringified. |
| `max_tokens` | int | Optional; default 16384. |
| `temperature` | float | Optional; default 0.7. |
| `stream` | bool | Default **false**. |
| `top_k` / `top_p` / `min_p` / `repeat_penalty` | num | Optional; emitted only if set. |
| `enable_thinking` | bool | LAMU extension. Omit → backend template default. |
| `tools` / `tool_choice` | array / value | OpenAI tools forwarded verbatim. |

**Minimal request:**
```bash
curl http://127.0.0.1:8020/v1/chat/completions \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hi"}]}'
```

**Full request:**
```json
{
  "model": "qwen3.6-27b",
  "messages": [
    {"role":"system","content":"be terse"},
    {"role":"user","content":"hi"}
  ],
  "max_tokens": 256, "temperature": 0.7, "stream": false,
  "top_k": 40, "top_p": 0.95, "min_p": 0.05, "repeat_penalty": 1.1,
  "enable_thinking": false,
  "tools": [{"type":"function","function":{"name":"f","parameters":{}}}],
  "tool_choice": "auto"
}
```

**Non-stream 200 response:**
```json
{
  "id": "chatcmpl-9f2a1b3c4d5e",
  "object": "chat.completion",
  "created": 1748736000,
  "model": "qwen3.6-27b",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Hello.",
      "reasoning_content": "..."
    },
    "finish_reason": "stop"
  }],
  "usage": { "prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15 },
  "timings": { "predicted_per_second": 101.7 }
}
```
- `reasoning_content` present only when non-empty.
- `usage` is passed through from the backend (or `null`).
- `timings` is a non-standard llama.cpp passthrough; present only if the backend emits it.

**Streaming** (`stream: true`) — SSE, `Content-Type: text/event-stream`:
```
data: {"id":"chatcmpl-..","object":"chat.completion.chunk","created":1748736000,"model":"qwen3.6-27b","choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null}]}

data: {"id":"chatcmpl-..","object":"chat.completion.chunk","created":1748736000,"model":"qwen3.6-27b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```
Reasoning is **not** streamed on this surface (stripped). Stream error events:
```
data: {"error":"backend: <msg>"}
data: [DONE]
```
```
data: {"error":{"type":"backend_returned_empty","message":"backend produced no content and no legitimate finish reason"}}
data: [DONE]
```

**Errors** (non-stream):
| Status | Body |
|--------|------|
| 503 | `{"error":{"message":"<lock msg>","type":"gpu_locked"}}` |
| 503 | `{"error":{"message":"No model: <reason>","type":"model_not_available"}}` |
| 503 | `{"error":{"message":"Failed to load '<name>': <e>","type":"spawn_failed"}}` |
| 500 | `{"error":{"message":"internal: lost loaded model after spawn"}}` |
| 502 | `{"error":{"message":"Backend unreachable: <e>"}}` |
| 502 | `{"error":{"message":"Bad JSON from backend: <e>"}}` |
| 502 | `{"error":{"type":"backend_returned_empty","message":"backend on :<port> returned no usable content (finish_reason='<fr>')"}}` |
| 500 | `{"error":{"message":"LAMU_GATEWAY_URL is not a valid HTTP URL","type":"config"}}` |

`backend_returned_empty` fires when the message is missing, OR content+reasoning are
empty AND there are no tool_calls AND `finish_reason` is not in
`{stop, length, tool_calls, content_filter}`.

**Gateway:** if `LAMU_GATEWAY_URL` is set (validated http/https, no userinfo), this
surface forwards to `{gw}/chat/completions` with the client's `model` passed
through. This is the **only** HTTP path to non-local backends, and it applies to
this surface only (not Anthropic-stream, Ollama-stream, or embeddings).

---

### POST /v1/embeddings

OpenAI embeddings, proxied to a local `llama-server --embedding`. Near-blind
passthrough: the request body is raw JSON (no struct validation), the response is
the backend's bytes with the backend's status code and forced
`Content-Type: application/json`.

Model resolution: the request `model` if it is an Embedding-capability entry, else
the first registry entry with `Capability::Embedding`.

```bash
curl http://127.0.0.1:8020/v1/embeddings \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"model":"bge-m3","input":"hello world"}'
```

**Errors:**
| Status | Body |
|--------|------|
| 503 | `{"error":{"message":"no embedding model in registry — add one with capability 'embedding' (llama-server --embedding)","type":"model_not_available"}}` |
| 503 | `{"error":{"message":"failed to load embedding model '<n>': <e>","type":"spawn_failed"}}` |
| 502 | `{"error":{"message":"embeddings backend read failed: <e>"}}` / `{"error":{"message":"embeddings backend: <e>"}}` |

---

### POST /v1/messages (Anthropic)

Anthropic Messages shim. Translates to the internal `ChatRequest`, reuses the
OpenAI pipeline, maps the response back. Full streaming in Anthropic event shape.

**Request fields** (`AnthropicRequest`):

| Field | Type | Notes |
|-------|------|-------|
| `model` | string \| null | Optional. |
| `messages` | array | **Required**. `content` is a string or content-block array. |
| `system` | string \| array | Optional. Array of blocks → text concatenated with `\n`. |
| `max_tokens` | int | Anthropic spec requires it; LAMU is lenient, default 16384. |
| `temperature` / `top_k` / `top_p` / `min_p` / `repeat_penalty` | num | Optional. |
| `stream` | bool | Default **false**. |
| `tools` | array | Anthropic `{name,description,input_schema}` → OpenAI function tools. |
| `tool_choice` | value | `{type:auto\|any\|tool\|none,name?}` → OpenAI `auto`/`none`/`required`/`{type:function,...}`. |
| `enable_thinking` | bool | LAMU extension (not Anthropic spec). |

Message expansion: text blocks concatenated; a `tool_use` block becomes assistant
text `[Tool call id=<id>] name(<json args>)`; a `tool_result` block becomes a `tool`
message `[Tool result id=<id>]\n<body>`; image/unknown blocks ignored.
**Inbound `tool_call_id` wiring is lost** (internal Message is string-only) — new
tool calls the model emits *are* returned structurally.

**Minimal / Full requests:**
```json
{"messages":[{"role":"user","content":"hi"}]}
```
```json
{
  "model": "lamu", "system": "be terse",
  "messages": [{"role":"user","content":[{"type":"text","text":"hi"}]}],
  "max_tokens": 1024, "temperature": 0.7, "stream": false,
  "enable_thinking": false,
  "tools": [{"name":"get_weather","description":"...","input_schema":{"type":"object","properties":{}}}],
  "tool_choice": {"type":"auto"}
}
```

**Non-stream 200 response:**
```json
{
  "id": "msg_9f2a1b3c4d5e", "type": "message", "role": "assistant",
  "model": "qwen3.6-27b",
  "content": [
    {"type":"text","text":"Sunny."},
    {"type":"tool_use","id":"toolu_abc","name":"get_weather","input":{"city":"SF"}}
  ],
  "stop_reason": "tool_use",
  "stop_sequence": null,
  "usage": {"input_tokens": 12, "output_tokens": 6}
}
```
Text block first (if non-empty), then a `tool_use` per tool call. `stop_reason` is
`tool_use` iff any tool_use emitted, else `end_turn`. Note: this surface does **not**
expose `reasoning_content` — only `/v1/chat/completions` does.

**Streaming** — SSE with **named events**, terminated by `message_stop` (**no
`[DONE]` sentinel**):
```
event: message_start
data: {"type":"message_start","message":{"id":"msg_..","type":"message","role":"assistant","model":"qwen3.6-27b","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Sun"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":6}}

event: message_stop
data: {"type":"message_stop"}
```
Tool calls stream as further `content_block_start` (`tool_use`) / `input_json_delta`
/ `content_block_stop` at index ≥ 1. Backend error: `event: error` with
`{"type":"error","error":{"type":"backend_error","message":"<e>"}}`. Note:
streamed `output_tokens` is a naive per-delta count, not the backend tokenizer count.

**Errors:**
| Status | Body |
|--------|------|
| 502 | `{"type":"error","error":{"type":"backend_returned_empty","message":"backend produced neither text nor tool_use blocks"}}` |
| 500 | `{"type":"error","error":{"type":"internal","message":"body read: <e>"}}` |
| 502 | `{"type":"error","error":{"type":"bad_response","message":"json: <e>"}}` |

> Delegated operational errors (gpu_locked / no-model / spawn_failed /
> backend-unreachable) are translated into the **Anthropic** error envelope
> (`{"type":"error","error":{"type":"...","message":"..."}}`) on this surface,
> with the `type` mapped from the HTTP status.

---

### GET /api/tags (Ollama)

Lists the registry in Ollama shape.

```bash
curl http://127.0.0.1:8020/api/tags -H "Authorization: Bearer $TOKEN"
```
```json
{
  "models": [{
    "name": "qwen3.6-27b", "model": "qwen3.6-27b",
    "modified_at": 1748736000,
    "size": 18874368000,
    "details": {"family":"qwen3","parameter_size":"27B","quantization_level":"Q4_K_M"}
  }]
}
```
`size` is `vram_mb * 1048576`. `modified_at` is a unix int (real Ollama uses an
RFC3339 string — minor drift for strict clients).

---

### POST /api/chat (Ollama)

Ollama chat. **`stream` defaults to TRUE** when omitted (matches real Ollama).

**Request fields** (`OllamaChatRequest`):

| Field | Type | Notes |
|-------|------|-------|
| `model` | string \| null | Optional. |
| `messages` | array | **Required**. `{role, content}` — string only, no array form. |
| `stream` | bool | **Default true.** |
| `options` | object | `{temperature, top_p, top_k, min_p, repeat_penalty, num_predict}`; `num_predict` → `max_tokens`. |
| `enable_thinking` | bool | LAMU extension (top-level). |

**Minimal request** (returns an ndjson stream — `stream` defaults true):
```json
{"messages":[{"role":"user","content":"hi"}]}
```
**Non-stream request:**
```json
{
  "model": "lamu",
  "messages": [{"role":"user","content":"hi"}],
  "stream": false, "enable_thinking": false,
  "options": {"temperature":0.7,"top_p":0.95,"top_k":40,"min_p":0.05,"repeat_penalty":1.1,"num_predict":256}
}
```

**Non-stream 200 response:**
```json
{
  "model": "qwen3.6-27b",
  "created_at": "2026-05-31T00:00:00Z",
  "message": {"role":"assistant","content":"Hello."},
  "done_reason": "stop", "done": true,
  "total_duration": 0, "load_duration": 0,
  "prompt_eval_count": 12, "eval_count": 3, "eval_duration": 0
}
```
Duration fields are hardcoded 0 (not real measurements).

**Streaming** — **NDJSON** (`Content-Type: application/x-ndjson`), one JSON object
per line, **no `data:` prefix, no `[DONE]`**, terminated by a `"done":true` line:
```
{"model":"qwen3.6-27b","created_at":"2026-05-31T00:00:00Z","message":{"role":"assistant","content":"Hel"},"done":false}
{"model":"qwen3.6-27b","created_at":"2026-05-31T00:00:00Z","message":{"role":"assistant","content":""},"done_reason":"stop","done":true,"total_duration":0,"load_duration":0,"prompt_eval_count":0,"eval_count":3,"eval_duration":0}
```
Backend error line: `{"error":"backend: <e>"}`. Empty-backend gate line:
`{"model":"<n>","created_at":"...","error":"backend_returned_empty: ...","done":true}`.

---

## Status codes & error envelopes

| Status | When |
|--------|------|
| 200 | Success (all surfaces). Also used on `/v1/embeddings` passthrough where the backend's own status is forwarded (can be non-200). |
| 400 | Malformed JSON body (axum default). |
| 401 | Bad/missing bearer when a token is configured (all routes except `/health`, `/metrics`). |
| 422 | JSON deserialization failure (missing `messages`, wrong types) — **axum default body, not a surface envelope** (see footguns). |
| 500 | Internal: lost model after spawn; `LAMU_GATEWAY_URL` config error; Anthropic body-read error. |
| 502 | Backend unreachable / bad JSON / `backend_returned_empty` / embeddings backend read fail. |
| 503 | `gpu_locked` / `model_not_available` / `spawn_failed`. |

Envelope shapes by family:
- **OpenAI / Ollama (non-stream):** `{"error":{"message":"...","type":"<type>"}}` (some
  502/500 bodies omit `type`).
- **Anthropic:** `{"type":"error","error":{"type":"...","message":"..."}}`.
- **Ollama (its own errors):** flat `{"error":"<string>"}`.

> 4xx parse failures (400/422) do **not** use these envelopes — they are axum's
> default `text/plain` prose. Do not assume `resp.json()` succeeds on a 4xx parse error.

---

## Point your frontend at LAMU

| Frontend | Surface | How to configure |
|----------|---------|------------------|
| **Claude Code** | Anthropic `/v1/messages` | Point `ANTHROPIC_BASE_URL` at `http://127.0.0.1:<port>`; send `model: "lamu"`. Tools translate both directions; single-turn tool loops work. |
| **Open WebUI** (OpenAI mode) | `/v1/chat/completions` + `/v1/models` | Add an OpenAI connection with base URL `http://127.0.0.1:<port>/v1`. |
| **Open WebUI / AnythingLLM** (Ollama mode) | `/api/tags` + `/api/chat` | Set the Ollama URL to `http://127.0.0.1:<port>`. (No `/api/version`/`/api/show` — capability probing may misbehave.) |
| **AnythingLLM** (OpenAI/generic) | `/v1/chat/completions` | Generic OpenAI provider, base URL `http://127.0.0.1:<port>/v1`. |
| **Continue / LibreChat** | OpenAI or Anthropic | Same as the matching family above. |
| **OpenAI / Anthropic SDKs, curl** | `/v1/*` or `/v1/messages` | Set base URL + bearer token (if configured). |
| **RAG front-ends** (Chroma, etc.) | `/v1/embeddings` | Point `EMBEDDING_URL` at `http://127.0.0.1:<port>/v1/embeddings`; ensure a registry entry has the `embedding` capability. |

Pin your frontend's model name to `lamu` (or omit it) to stay agnostic of which
model actually backs the request. If running off-loopback, configure the token first.

---

## Footguns

1. **Three different streaming wire formats.** `/v1/chat/completions` = SSE
   terminated by `data: [DONE]`; `/v1/messages` = SSE with **named events**, no
   `[DONE]` (ends at `message_stop`); `/api/chat` = **NDJSON**, no `data:` prefix,
   no `[DONE]` (ends at a `"done":true` line).
2. **`/api/chat` `stream` defaults to TRUE.** Omitting `stream` yields an ndjson
   stream, not a single JSON object. OpenAI and Anthropic surfaces default `false`.
3. **Cloud is not on HTTP.** Local registry only. Cloud → MCP `cloud_query`. The
   only off-box HTTP path is `LAMU_GATEWAY_URL`, and it applies to
   `/v1/chat/completions` only.
4. **HTTP auto-load never auto-evicts.** Switching models when VRAM is full errors
   instead of swapping (ADR 0006). MCP's load path can evict; HTTP cannot.
5. **Reasoning handling is inconsistent across surfaces** (see
   [LAMU extensions](#lamu-extensions)). `reasoning_content` is exposed only on
   `/v1/chat/completions` non-stream.
6. **Vision/image content is silently dropped** on all surfaces — only text parts of
   a content array survive, with no error, even for `vision`-capable entries.
7. **Multi-turn tool history degrades to text** on the Anthropic bridge: replayed
   `tool_use`/`tool_result` become plain-text strings; `tool_call_id` linkage is lost
   inbound. New (outbound) tool calls are structured.
8. **Validation errors (400/422) are not surface envelopes** — they are axum's
   default `text/plain` prose. `resp.json()` on a parse-failure 4xx throws.
9. **Remaining error-envelope gaps:** `/v1/messages` operational errors and 401s
   are now surface-correct (Anthropic envelope / path-aware). Still imperfect:
   Ollama `/api/chat` passes some delegated errors through in the OpenAI shape
   rather than flat `{"error":"..."}`, and validation 4xx (next item) remains
   axum's default. (Tracked.)
10. **Synthetic timings on Ollama/Anthropic.** Ollama duration fields are 0;
    Anthropic streamed `output_tokens` is a per-delta count. Only the OpenAI surface
    passes the backend's real `usage`/`timings`.
11. **No `/v1/completions`, `/api/generate`, `/api/show`, `/api/version`,
    `/api/embeddings`.** Probing those 404s.
12. **`enable_thinking` requires a template that honors it** (Qwen3.6/3.5). For other
    families it is a forwarded no-op; there is no per-model declaration in the response.

---

## MCP is the agent plane

Everything *agentic* lives on a separate **MCP/stdio** contract (`lamu-mcp`), not on
HTTP. None of these are reachable over `curl` to `lamu serve`:

- **Inference + lifecycle:** `query`, `plan_query`, `list_models`, `load_model`,
  `unload_model`, `vram_status`, `scan_models`, `queue_status`
- **Cloud / orchestration:** `cloud_query`, `council`, `parallel_query`,
  `list_cloud_models`, `review_commit`, `review_diff`, `warmup`, `consolidate_memory`
- **Routing:** `set_routing_mode` (`auto` / `local-only` / `cloud-only`), `routing_status`
- **RAG:** `search_repo`, `index_repo`
- **Memory:** `remember`, `recall_memory`, `forget_memory`, `export_memory_graph`,
  `recall_conversation`
- **Media / train / fs:** `generate_image`, `text_to_speech`,
  `train_from_conversations`, `write_file`

`routing_mode` (MCP state, default `auto`) gates backend availability:
`local-only` refuses cloud tools; `cloud-only` drains local backends to free VRAM
and refuses local queries; `auto` uses local for matching capabilities and cloud for
the rest. Cloud models live in a separate `~/.config/lamu/cloud-models.yaml` and are
reached directly by the provider's native wire format (Anthropic Messages or
OpenAI-compat), never through this HTTP server.

The HTTP contract stays deliberately dumb: a client that already speaks
OpenAI/Anthropic/Ollama points its base URL at `lamu serve` and works with zero
LAMU-specific knowledge. See ADR 0001 (MCP-first) and ADR 0016 (backend orchestrator
/ bring-your-own-frontend).
