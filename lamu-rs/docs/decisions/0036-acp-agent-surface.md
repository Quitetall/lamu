# ADR 0036: Native ACP agent surface (`lamu acp`)

## Status

Accepted 2026-06-12

## Context

ACP (Agent Client Protocol) is Zed's editor↔agent protocol: the editor
spawns an agent over stdio, streams session updates, and brokers
permissions and file access. LAMU had four serving surfaces but no way
for an ACP editor to drive a LOCAL model with LAMU's tools (research,
memory, web search) directly — the "both layers" user decision: native
ACP now, with the katana harness fronting the same provider later.

## Decision

`lamu acp` — an IN-CLI module (`lamu-cli/src/acp/`), not a crate: it is
another CLI mode like `lamu start`, and lamu-cli is already the ADR 0023
composition root coupling lamu-mcp + modules, so a `lamu-acp` crate
would have created exactly the frontend→frontend edge ADR 0029 exists to
prevent. Protocol layer HAND-ROLLED, with wire shapes pinned against
Zed's official `agent-client-protocol` crate v0.14.0 source (framing
verified there: newline-delimited JSON-RPC 2.0, not LSP Content-Length).
The crate itself was rejected: ~10 new transitive deps and a
runtime-agnostic inverted-control connection layer that fights the tokio
stdio-loop pattern, on a 0.x line with 69 releases in 11 months.

Dispatch: `session/prompt` runs in a spawned task so `session/cancel`
is honored MID-turn — a scoped, documented deviation from ADR 0024
(which is MCP-loop-scoped); every other method stays serial inline. The
agent loop streams the loaded model's port: `delta.content` →
`agent_message_chunk`, `delta.reasoning_content` (ADR 0037's split) →
`agent_thought_chunk`; tool_calls accumulate ToolAcc-style and execute
through `dispatch_tool_text` — a function extracted from the MCP
dispatcher so ACP rides the EXACT same path (local-only gate, module
fallback), zero divergence. Curated tool subset: query, web_search,
research, recall_memory, remember, write_file (MCP schemas translated
to OpenAI function shapes). Write-effecting tools gate on
`session/request_permission` with allow/reject once/always semantics
(allow_always cached per session+tool); `write_file` routes through the
client's `fs/write_text_file` when advertised, else the local journaled
handler. Turn cap 10 → `max_turn_requests` (the spec's stop reason for
exactly this, verified in the schema crate — not an invented note).

## Rationale

- Pinning shapes to the official crate's source gives spec fidelity
  without inheriting its dependency and control-flow choices; the pin is
  recorded so a future crate adoption is a mechanical swap.
- Reusing `dispatch_tool_text` means ACP tool calls inherit every MCP
  dispatch guarantee (routing gate, wire conventions) and never fork.
- The permission gate is the editor's UI affordance — LAMU defers to the
  client rather than inventing local policy.

## Alternatives Considered

- **Official crate end-to-end** — dep weight + inverted control;
  rejected with specifics above.
- **Separate lamu-acp crate** — frontend→frontend dependency or
  duplicated composition; rejected.
- **Strictly serial dispatch (ADR 0024 verbatim)** — would make
  session/cancel dead until turn end, violating the protocol's
  expectations. Scoped deviation instead.

## Consequences

- Zed (any ACP client) can register `lamu acp` and drive local models
  with thought streaming, tool lifecycle, and permission prompts.
- v1 gaps doc-noted: `fs/read_text_file` unwired (no read tool in the
  curated set), local-fallback write_file journals against process cwd,
  `mcpServers` accepted-ignored, `session/load` unadvertised.
- Protocol shape drift risk rides Zed's 0.x churn; the version pin makes
  drift detectable.

## Related Decisions

ADR 0023/0029 (why in-cli), ADR 0024 (the scoped deviation), ADR 0037
(reasoning split the thought chunks consume), ADR 0011 (tool results
remain fenced at consuming boundaries).

## Validation

7 tests over an in-memory duplex + scripted SSE stub: updates precede
the prompt response; thought/message separation; split-args ToolAcc
reassembly; tool lifecycle ordering; mid-stream cancellation →
`cancelled`; permission reject-once-continue + allow_always caching
(1 request for 2 writes) + client-fs assertions; schema translation.
Workspace 792 green. Live gate: real Zed `settings.json` registration
driving a tool call + permission prompt (queued).
