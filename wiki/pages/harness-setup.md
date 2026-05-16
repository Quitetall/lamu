# Harness Setup

Lamu = universal local LLM endpoint. Any "harness" (Claude Code, Codex, Crush, Cursor, Aider, Continue, pi, Hermes, AnythingLLM, Open WebUI) gets pointed at `http://localhost:8020` and talks to whichever model is marked `main: true` in `config/models.yaml`.

One launcher (`just open` / `scripts/open-harness.sh`) reads `config/harnesses.yaml`, sets the right env vars for that harness's API flavor, and execs it.

---

## API surfaces lamu exposes

All three live on the same port (default `:8020`), routed by path:

| Surface | Routes | Use when |
|---------|--------|----------|
| **OpenAI-compatible** | `POST /v1/chat/completions`, `GET /v1/models` | Codex, Cursor, Aider, Continue, pi, custom OpenAI clients |
| **Anthropic Messages** | `POST /v1/messages` (SSE + tool_use) | Claude Code, Crush, Hermes, anything that speaks Anthropic |
| **Ollama compat** | `POST /api/chat`, `GET /api/tags` (NDJSON streaming) | AnythingLLM, Open WebUI, Ollama-only tools |

The Anthropic shim translates `tool_use` ↔ OpenAI `tool_calls` server-side, so harnesses that hand-roll Anthropic JSON still work end-to-end against a local model.

---

## Default model — `main: true`

In `config/models.yaml`, exactly one entry should have `main: true`:

```yaml
qwen3.6-27b-uncensored-heretic-v2-q4_k_m:
  ...
  status: recommended
  main: true
```

When a harness omits the `model` field, or passes `model: "default" | "main" | "lamu"`, lamu's router resolves to whichever entry has `main: true`. No harness-side config needed.

Change the daily driver:
```bash
# edit config/models.yaml — flip main: true to a different entry
# restart lamu
just serve
```

---

## Harness registry — `config/harnesses.yaml`

```yaml
harnesses:
  claude-code:
    default: true              # 'just open' with no args launches this
    flavor: anthropic          # which env-var family to set
    cmd: claude                # shell command to exec

  codex:
    flavor: openai
    cmd: codex

  aider:
    flavor: openai
    cmd: aider
    extra_env:
      AIDER_MODEL: openai/lamu
```

Fields:

| Field | Meaning |
|-------|---------|
| `default: true` | Exactly one entry. `just open` (no args) picks this one. |
| `flavor` | `anthropic` \| `openai` \| `ollama` — picks env vars to set. |
| `cmd` | Shell command. Extra argv after `just open <name>` appended verbatim. |
| `extra_env` | Optional. Map of `KEY: value` env vars set before exec. |

Per-flavor env vars set by the launcher:

| Flavor | Env vars exported |
|--------|-------------------|
| `anthropic` | `ANTHROPIC_BASE_URL=http://localhost:8020`, `ANTHROPIC_API_KEY=lamu-local` (if unset) |
| `openai` | `OPENAI_BASE_URL=http://localhost:8020/v1`, `OPENAI_API_KEY=lamu-local` (if unset) |
| `ollama` | `OLLAMA_BASE_URL=http://localhost:8020`, `OLLAMA_HOST=localhost:8020` |

Override the base URL globally:
```bash
LAMU_URL=http://192.168.1.10:8020 just open
```

---

## Usage

```bash
just open                # launches default (claude-code)
just open codex          # launches named entry
just open aider --model openai/lamu file.py   # extra argv forwarded
just open list           # show configured harnesses + which is default
```

`just open list` output:
```
claude-code    anthropic    claude  (default)
crush          anthropic    crush
codex          openai       codex
...
```

The launcher pre-checks `GET /v1/models` and warns (but doesn't abort) if lamu isn't reachable.

---

## Adding a new harness

1. Edit `config/harnesses.yaml`:
   ```yaml
   my-harness:
     flavor: openai
     cmd: my-harness-cli
     extra_env:
       MY_HARNESS_MODEL: lamu
   ```
2. Done. `just open my-harness` picks it up immediately — no rebuild.

If the harness uses a non-standard env var, add it under `extra_env` instead of editing the launcher. The launcher only knows the three flavors; everything else is per-harness config.

---

## Harnesses that read config files, not env vars

Some harnesses ignore `OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL` and instead read a config file. The env-var trick in `open-harness.sh` won't work for them. lamu ships per-harness setup scripts that patch the config file once; afterwards the harness just runs.

### `pi` (Earendil pi-coding-agent)

Reads `~/.pi/agent/models.json`. Run once to register `lamu` as a custom OpenAI-compatible provider with the full lamu registry pulled in:

```bash
bash scripts/setup-pi.sh
pi --provider lamu --model qwen3.6-27b-uncensored-heretic-v2-q4_k_m "hello"
```

Or set `lamu` as the default in `pi config` (interactive). Re-running `setup-pi.sh` refreshes the model list against the current lamu registry — useful after `lamu scan`.

Requires `jq`.

### `hermes` (Nous Research agent)

Reads `~/.hermes/config.yaml`. Run once to patch the three relevant keys under `model:` (`default`, `provider`, `base_url`):

```bash
bash scripts/setup-hermes.sh
hermes -z "hello" chat
```

`default` is set to the `"lamu"` alias so lamu's router picks the `main: true` entry. Override per-call with `-m <name>`. Backup of the original config goes to `~/.hermes/config.yaml.bak`.

### Other config-file harnesses

| Harness | Config file | Field to set |
|---|---|---|
| **Cursor** | Settings UI → "OpenAI API Key" + "OpenAI Base URL" | `http://localhost:8020/v1` |
| **Aider** | `--openai-api-base http://localhost:8020/v1 --model openai/lamu` or `~/.aider.conf.yml` | `openai-api-base: http://localhost:8020/v1` |
| **Continue** | `~/.continue/config.json` → `models[].provider: openai` + `apiBase: http://localhost:8020/v1` | model object |
| **AnythingLLM** | Settings → LLM Preference → "Generic OpenAI" → `http://localhost:8020/v1` | `apiBase` |
| **Open WebUI** | Settings → Connections → OpenAI → URL `http://localhost:8020/v1` | URL field |

Pattern: any "custom OpenAI-compatible endpoint" field accepts lamu at `http://localhost:8020/v1`. Lamu's `/v1/chat/completions`, `/v1/messages`, and `/api/chat` cover the three wire formats most tools speak. Pick whichever flavor the tool natively supports — translation is server-side.

---

## Thinking toggle (`enable_thinking`)

Qwen3.6 reasoning can be turned off per-request. Two paths exposed:

**HTTP — pass `enable_thinking: false` in the request body:**
```bash
curl :8020/v1/chat/completions -d '{
  "model": "lamu",
  "messages": [{"role":"user","content":"Hi"}],
  "enable_thinking": false
}'
```

Lamu injects `chat_template_kwargs.enable_thinking` into the upstream llama-server call. Works on `/v1/chat/completions`, `/v1/messages`, `/api/chat`.

**MCP — pass the same field to the `query` tool:**
```python
mcp__local-llm__query({prompt: "...", enable_thinking: false})
```

Wall-time effect on Qwen3.6-27B-bee: ~4× faster on tiny prompts (no `<think>...</think>` preamble), ~1.2× on 8k-prompt + 200-token cap. Caveat: `enable_thinking: true` + low `max_tokens` can produce 0 visible tokens (all the budget burned in reasoning).

---

## How the pieces fit

```
┌─────────────┐    ANTHROPIC_BASE_URL    ┌──────────────────┐
│ claude-code │ ──────────────────────▶  │                  │
└─────────────┘                          │                  │
┌─────────────┐    OPENAI_BASE_URL       │   lamu :8020     │   llama-server :8081
│   codex     │ ──────────────────────▶  │  ┌────────────┐  │   ┌──────────────┐
└─────────────┘                          │  │ /v1/msgs   │  │   │              │
┌─────────────┐    OLLAMA_BASE_URL       │  │ /v1/chat   │ ─┼──▶│  Qwen3.6-27B │
│ open-webui  │ ──────────────────────▶  │  │ /api/chat  │  │   │  (main)      │
└─────────────┘                          │  └────────────┘  │   │              │
                                         │   router         │   └──────────────┘
                                         │   (main: true)   │
                                         └──────────────────┘
```

Every harness sees its native API. Lamu is the universal translator + router.

---

## Files

- `config/models.yaml` — `main: true` flag picks default model.
- `config/harnesses.yaml` — harness registry, `default: true` flag picks default harness.
- `scripts/open-harness.sh` — launcher; reads yaml, sets env, execs.
- `justfile` — `open` recipe wraps the launcher.
- `lamu-rs/lamu-api/src/openai_compat.rs` — all three HTTP surfaces.
- `lamu-rs/lamu-api/src/lib.rs` — `serve` entry point: pidfile, SO_REUSEADDR, preload, graceful shutdown.
- `lamu-rs/lamu-core/src/loader.rs` — request-driven backend spawn shared by HTTP + MCP.
- `lamu-rs/lamu-core/src/router.rs` — `main`/`default`/`lamu` alias resolution.
- `lamu-rs/lamu-core/src/backends/llamacpp.rs` — `enable_thinking` → `chat_template_kwargs` injection.

---

## Troubleshooting

### `lamu status` shows `🟡 :8020 — http up, no model loaded`
The HTTP layer is live but no backend (llama-server) is loaded yet. Send a chat completion — `lamu serve` spawns the `main: true` model on first request (or via the startup preload, in flight). Watch the log:
```bash
tail -f /tmp/lamu-serve.log
```
If the preload failed (e.g. `vram exhausted`), pick a smaller `main: true` entry in `config/models.yaml` or unload other GPU workloads.

### Harness gets 503 `model_not_available` or `spawn_failed`
- `model_not_available` → router couldn't resolve. Check `lamu status` shows the registry and that `main: true` is set on exactly one entry.
- `spawn_failed` → the underlying `llama-server` couldn't start. Check `/tmp/lamu-serve.log`. Common causes: GGUF path missing on disk, `$LAMU_LLAMACPP_DIR` unset, VRAM exhausted.

### Harness gets 502 `backend_returned_empty`
Backend produced no content + no tool_calls + no valid `finish_reason`. Usually means the llama-server crashed or returned malformed JSON. Restart: `pkill -f llama-server; pkill -f "lamu serve"` then re-launch.

### `lamu serve` refuses with `lamu serve already running on :8020 (pid N)`
A previous instance still owns the port via its pidfile. `kill N` (or `pkill -f "lamu serve"`) and retry. Pidfile lives at `$XDG_RUNTIME_DIR/lamu-serve-{port}.pid` or `/tmp/lamu-serve-{port}.pid`. Stale pidfiles from SIGKILLed predecessors are auto-cleaned on the next start.

### Multiple stale `lamu start` MCP daemons in `ps aux`
On Linux, `lamu start` now installs `PR_SET_PDEATHSIG` so the kernel kills it when the parent harness dies. Older binaries don't have this — `pkill -f "lamu start"` and rebuild (`cargo install --path lamu-rs/lamu-cli`).

### End-to-end smoke
```bash
# All three surfaces; expect non-empty responses.
curl -s :8020/v1/chat/completions -d '{
  "model":"lamu","messages":[{"role":"user","content":"reply ok"}],
  "max_tokens":20,"enable_thinking":false}'
curl -s :8020/v1/messages -d '{
  "model":"lamu","max_tokens":20,
  "messages":[{"role":"user","content":"reply ok"}],"enable_thinking":false}'
curl -s :8020/api/chat -d '{
  "model":"lamu","stream":false,
  "messages":[{"role":"user","content":"reply ok"}],"enable_thinking":false}'
```
