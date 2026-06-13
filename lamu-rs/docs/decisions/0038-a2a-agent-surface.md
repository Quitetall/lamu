# ADR 0038: A2A (Agent2Agent) agent surface (`lamu a2a`)

## Status

Accepted 2026-06-13

## Context

A2A (Agent2Agent) is the Linux-Foundation-governed agent↔agent interop
protocol (originated at Google): an HTTP JSON-RPC 2.0 surface where one
agent discovers another via an Agent Card and drives it as a remote peer
— `message/send`, `message/stream` (SSE), `tasks/get`, `tasks/cancel`,
with a task lifecycle (submitted/working/input-required/completed/
failed/canceled). LAMU had five serving surfaces (MCP, OpenAI-compat
HTTP, ACP, memory HTTP, the CLI) but none let ANOTHER AGENT drive
LAMU's agent loop over the network. For katana fleets this is the
inter-agent fabric: a katana instance (or any A2A client) discovers and
tasks LAMU's local-model agent remotely.

W6 (`lamu acp`, ADR 0036) already built the hard half — an agent loop
that streams the local model, accumulates tool_calls, dispatches via
`lamu_mcp::server::dispatch_tool_text`, and honors a cancellation
watch-channel. A2A is a second transport over that same loop shape.

## Decision

`lamu a2a [--port 8022] [--bind 127.0.0.1]` — an IN-CLI module
(`lamu-cli/src/a2a/`) plus a shared `lamu-cli/src/agent_core/` module,
NOT crates. lamu-cli is the ADR 0023/0029 composition root that already
couples lamu-mcp + the module crates; the A2A loop needs lamu-mcp's tool
dispatch, which lamu-api must NEVER depend on, so A2A lives beside ACP in
lamu-cli, not in lamu-api. An `a2a-rs`/official-crate dependency was
rejected for the same reasons ADR 0036 rejected the ACP crate (new dep
tree, control-flow mismatch); wire shapes are hand-rolled and the spec
version is pinned.

**Spec pin:** A2A Protocol Specification **v1.0.0** (Major.Minor `1.0`),
sourced from `https://a2a-protocol.org/latest/specification/` on
2026-06-13; normative protobuf at `github.com/google-a2a/A2A`. Only the
HTTP/JSON-RPC surface is implemented (no gRPC). Recorded in
`a2a/protocol.rs` module docs so a future bump is a mechanical diff.

**`agent_core`** holds two traits the loop is parameterized on —
`UpdateSink` (emit a `LoopEvent`: message/thought chunk, tool-call,
tool-call-update) and `PermissionGate` (`async fn request → Allowed /
Rejected / CancelledTurn`) — plus stock `AlwaysAllow` and `DenyWrites`
gates. `run_prompt_turn` here is the protocol-neutral loop: it reuses
ACP's LEAF helpers (`mcp_to_openai_tool`, `ToolAcc` reassembly,
`WRITE_EFFECTING_TOOLS`, `is_tool_error_text`, `prompt_blocks_to_text`)
so tool-schema translation and the MCP-failure heuristic stay
single-sourced, and dispatches through the exact `dispatch_tool_text`
path with a fail-closed out-of-subset guard.

**The A2A surface** (`a2a/{mod,protocol,sink}.rs`):
- `GET /.well-known/agent-card.json` (the v1.0.0 canonical path; renamed
  from `agent.json` in 0.2.x, which plus `/agent.json` stay as aliases),
  auth-EXEMPT — the Agent Card advertises the EFFECTIVE bind URL
  (threaded from `serve()`), `capabilities {streaming: true,
  pushNotifications: false}`, text-only I/O modes, and one skill per
  curated tool + a `chat` skill.
- `POST /` JSON-RPC: `message/send` (run the loop to completion, return
  the final Task with the assistant message as a text artifact),
  `message/stream` (SSE — `submitted` → `working` → per-chunk message
  events + thought DataParts → final completed Task), `tasks/get`
  (in-memory store, oldest-evicted at a 256 cap), `tasks/cancel` (flip
  the task's watch-channel → `canceled`).
- `contextId` → a persistent per-context session whose history carries
  across tasks; `taskId` per send — the same session/turn split as ACP.
- **DenyWrites v1:** the curated tool subset EXCLUDES `write_file` (no
  human present to answer a permission prompt); `recall_memory`,
  `remember`, `query`, `web_search`, `research`, `deep_research` only.
  A forged `write_file` tool_call hits the `DenyWrites` gate and fails
  closed even though it is not advertised.
- **Auth:** loopback bind is frictionless; an off-loopback bind is
  REFUSED at startup unless `LAMU_A2A_TOKEN` is set (mirrors ADR
  0005/0012 shape, inlined — not imported from lamu-api). When a token
  is set, every route except the card requires `Authorization: Bearer`,
  compared in constant time (`subtle`).

## Rationale

- A second transport over one loop shape, with `agent_core` traits as
  the seam, keeps ACP and A2A from forking their model-interaction
  behavior while letting each own its wire.
- Reusing `dispatch_tool_text` + `is_tool_error_text` means A2A tool
  calls inherit every MCP dispatch guarantee and the same failure
  heuristic as MCP and ACP.
- DenyWrites is the honest default for an unattended network peer: the
  A2A spec's `input-required` state is the eventual home for write
  prompts, but v1 ships read/research/memory-only rather than invent a
  side channel for approvals.
- Pinning to spec v1.0.0 and hand-rolling the JSON-RPC subset matches
  the ADR 0036 precedent and keeps the dep tree lean.

## Alternatives Considered

- **A2A in lamu-api** — would create the lamu-api→lamu-mcp edge ADR
  0029 forbids (the loop needs tool dispatch). Rejected.
- **`a2a-rs`/official crate end-to-end** — dep weight + control-flow
  mismatch, same as ADR 0036's ACP-crate rejection. Rejected; shapes
  pinned to the spec instead.
- **Allowing write_file with an auto-approve gate** — silently granting
  filesystem writes to an unauthenticated network peer. Rejected;
  DenyWrites + curated subset, with `input-required` as the documented
  path to attended writes.
- **Truly moving `run_prompt_turn` out of ACP** — would require
  abstracting ACP's client-filesystem `write_file` routing
  (`fs/write_text_file`) through the generic gate, risking the 7
  byte-stable ACP tests. Deferred (see gaps) — `agent_core` holds the
  shared loop A2A consumes; ACP keeps its copy until convergence is a
  separate, test-guarded change.

## Consequences

- Any A2A client (katana included) can discover `lamu a2a` via the card
  and task LAMU's local model with streaming, thought DataParts, tool
  lifecycle, and read/research/memory tools — write-free.
- **Known v1 gaps (documented):**
  - **SSE events are emitted BARE**, not wrapped in JSON-RPC
    `{jsonrpc, id, result}` response envelopes the spec expects for
    `message/stream`. A strict client keyed on the JSON-RPC frame will
    not parse them; the request-id is not yet threaded to the sink.
    First interop follow-up.
  - **ACP's loop is NOT converged onto `agent_core`** — a second copy of
    the loop lives in `acp/agent_loop.rs` (untouched, 7 tests
    byte-stable). SSE-parse / ToolAcc-reassembly fixes must currently be
    applied in both. Convergence is a guarded follow-up.
  - **`input-required` is unused** — write-needing turns are denied, not
    escalated to a prompt.
  - `message/send` blocks its axum worker for the whole turn (acceptable
    on a loopback default).
  - A client disconnect ends the SSE body but the spawned turn runs to
    completion (bounded resource use, not a correctness bug).
  - Concurrent `message/send` on the SAME `contextId` interleaves
    history (no per-context turn serialization in v1).
- `Command::A2a` adds axum + async-stream + subtle + async-trait to
  lamu-cli.

## Related Decisions

ADR 0036 (ACP — the loop shape and hand-rolled-pinned-shapes precedent),
ADR 0023/0029 (why in-cli, not a crate; no frontend→frontend edge),
ADR 0024 (the loop's serial-dispatch lineage), ADR 0005/0012 (the
loopback-default + bearer-token auth shape mirrored here), ADR 0037
(`reasoning_content` split the thought DataParts consume), ADR 0011
(tool results stay fenced at the dispatch boundary).

## Validation

9 A2A tests over a real port-0 listener + the ACP scripted-SSE fake
model: agent-card golden, JSON-RPC envelope parse/reject, `message/send`
→ completed Task with text artifact, `message/stream` SSE ordering
(submitted → working + chunks → thought DataParts → final Task),
`tasks/get` retained, `tasks/cancel` mid-stream → `canceled` + stream
closes, `write_file` absent from skills AND a forged call fails closed,
auth-exempt card + 401 RPC without token, off-loopback bind without
token refused at startup. The 7 ACP tests pass byte-unchanged (the
refactor gate). lamu-cli 126 green; workspace green. Live gate: a real
A2A client (or katana) driving a card-discovered tool call (queued).
