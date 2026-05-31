# ADR 0001: MCP-first orchestration; HTTP serve as a thin compat shim

## Status
Accepted 2026-05-31

## Context
LAMU has two front doors that expose the same pool of local + cloud models, and they serve different callers.

- The agentic caller is Claude Code (and peer agents) speaking MCP over stdio. It needs more than "generate a completion": it routes work to cheaper models, convenes multi-model councils, runs commit reviews, drives lifetime memory, generates media, flips routing modes, and fans out parallel jobs. These are orchestration verbs, not chat turns.
- The dumb caller is any tool that hardcodes an OpenAI / Anthropic / Ollama HTTP client — RAG front-ends, IDE plugins, `EMBEDDING_URL` consumers. They want a URL that behaves like the API they already target, nothing more.

At the time this was written the MCP catalog held 25 tools (`lamu-mcp/src/tools.rs:537-741`), the large majority of which have no HTTP equivalent. The HTTP layer (`lamu-api/src/openai_compat.rs::build_app`, lines 131-149) exposed exactly seven routes: `/health`, `/metrics`, `/v1/models`, `/v1/chat/completions`, `/v1/embeddings`, the Anthropic `/v1/messages` shim, and the Ollama `/api/tags` + `/api/chat` shims. The two surfaces had already diverged hard in capability; the question was whether to keep investing orchestration into HTTP, fold MCP into an HTTP gateway, or formalize the split.

## Decision
LAMU's primary control plane is the MCP server (`lamu-mcp`): a hand-rolled JSON-RPC-2.0 server over stdin/stdout (`lamu-mcp/src/server.rs:104-204`) that dispatches a table-driven tool catalog (`lamu-mcp/src/tools.rs::TOOLS`). All orchestration lives there and only there — `cloud_query`, `council`, `review_commit`/`review_diff`, `text_to_speech`, `generate_image`, `remember`/`recall_memory`/`consolidate_memory`/`forget_memory`/`export_memory_graph`, `parallel_query`, `set_routing_mode`/`routing_status`, `search_repo`/`index_repo`, `train_from_conversations`, `write_file`. `lamu serve` (`lamu-api/src/lib.rs::serve`, `openai_compat.rs::build_app`) is a deliberately-thin OpenAI/Anthropic/Ollama HTTP compat surface: it resolves a model, ensure-loads it, and proxies the payload to the backend's local llama-server `/v1/chat/completions` (`openai_compat.rs:496`) or an optional external gateway (`LAMU_GATEWAY_URL`, lines 459-494), doing only mechanical translation (reasoning-marker splitting, sampling-profile merge, Anthropic/Ollama envelope mapping). The HTTP path performs no council, no review, no memory, no media, no routing-mode control, and never exposes those verbs as routes.

## Rationale
- The orchestration verbs are inherently agentic — `review_commit`, `council`, `consolidate_memory` are tool calls an agent decides to make, not chat completions a client streams. MCP's `tools/list` (`server.rs:319-331`) is the native discovery + invocation protocol for that; OpenAI chat-completions has no slot for "review this commit and verify findings."
- The table-driven catalog makes MCP the cheap place to add capability: one `ToolDef` entry in `TOOLS` is picked up by both the dispatcher (`server.rs:263-274`) and `tools/list` automatically (`tools.rs:1-8`, `49-58`). Adding a route to the HTTP app is bespoke handler wiring each time (`build_app`, lines 131-149).
- Cross-cutting policy that the orchestrator needs — the `local-only`/`cloud-only` routing gate (`server.rs:258-268`), per-model FIFO queues for concurrent agents (`server.rs:294-302`), the rollback journal on `write_file` (`server.rs:345-413`) — is enforced at the MCP dispatch boundary where `&LamuMcpServer` state is in scope. The stateless HTTP proxy has no place to hang it.
- The HTTP surface earns its keep precisely by being boring: a client that already speaks OpenAI/Anthropic/Ollama points its base URL at `lamu serve` and works with zero LAMU-specific knowledge. The Anthropic shim translates `/v1/messages` into the internal `ChatRequest` and reuses the same pipeline (`anthropic_messages`, lines 1090-1143); the Ollama shims do likewise. Compatibility is the whole product of that surface.
- Keeping orchestration out of HTTP keeps the unauthenticated network surface (bound 127.0.0.1 by default, `lib.rs:18-33`) small. A council or a `write_file` reachable over an unauthenticated socket would be a much larger blast radius than a chat proxy.

## Alternatives Considered
- **HTTP-first design (Ollama model).** Make the HTTP server the brain: expose orchestration as REST endpoints and treat MCP as optional. Rejected because the primary caller is an MCP agent, and HTTP chat-completions is the wrong shape for tool semantics — there is no native discovery/typed-arg/`isError` channel; we'd reinvent `tools/list` and the `isError` flag (`server.rs:279-291`) as ad-hoc JSON conventions. It would also force the routing gate, per-model queues, and write journal onto an unauthenticated network socket rather than a stdio pipe owned by the parent harness.
- **External gateway (Bifrost / LiteLLM) as the brain.** Route everything through a third-party gateway that fans out to providers. Rejected as the control plane: a generic LLM gateway has no concept of LAMU's domain verbs — VRAM-aware eviction, blind-judge council, verify-before-fix commit review, the temporal lifetime-memory store. LAMU still *interoperates* with such a gateway, but only as a downstream egress: when `LAMU_GATEWAY_URL` is set the HTTP proxy forwards chat completions through it (`openai_compat.rs:459-494`, with scheme/userinfo validation). That is the gateway's correct role here — a forwarding target for the dumb surface, not the orchestrator.

## Consequences
- Two surfaces stay permanently asymmetric, and that asymmetry is intentional, not a gap. A feature reachable from Claude Code is not reachable over `curl` to `lamu serve` unless it is also a chat completion. Anyone expecting `lamu serve` to expose council/review/memory will be surprised; the route table (`build_app`) is the contract.
- The catalog-as-single-source-of-truth (`TOOLS`) is load-bearing. Tests pin it: `cloud_flag_matches_routing_policy` (`tools.rs:791-813`) and `critical_tools_present` (`tools.rs:816-828`) enforce that the `cloud` gate and the externally-depended-on tool names don't drift. Removing or renaming a tool is a breaking change for the agent contract.
- The MCP transport is a hand-rolled JSON-RPC loop (`server.rs:133-192`), not an SDK. We own protocol-version strings (`initialize_response`, `server.rs:306-316`) and framing. Upside: zero dependency surface and full control over dispatch/gating. Downside: any future MCP protocol evolution (resources, prompts, richer capabilities) is our manual port.
- The HTTP proxy must keep tracking upstream API quirks by hand — content-as-array flattening (`openai_compat.rs:40-70`), Anthropic tool_use/tool_result block expansion (`anthropic_message_to_openai`, lines 1006-1088), streaming `<think>`-tag splitting (`stream_response`, lines 626-794). This shim code is permanent maintenance owed to "behaves like the real API," and it is where compat bugs will live.
- Because orchestration is MCP-only, headless/CI use of the orchestration verbs requires an MCP client, not a `curl`. That is a deliberate cost: the dumb surface stays dumb.

## Related Decisions
ADR 0006 — HTTP path never auto-evicts; eviction is an MCP-only op (the same surface split applied to the VRAM scheduler).
ADR 0005 — Bind 127.0.0.1 by default (keeps the small HTTP surface unauthenticated-safe).
ADR 0007 — Unified cloud routing across OpenAI / Anthropic / OpenRouter (the egress that `cloud_query`/`council`/`review_*` MCP tools sit on top of).
ADR 0008 — Headless multi-model council instead of a compare UI (a flagship MCP-only orchestration verb).
ADR 0002 — Lean Rust backend, not a batteries-included framework (the philosophy this split expresses).

## Validation
This decision is right as long as: (1) new capabilities keep landing as `ToolDef` entries rather than HTTP routes — measure by the ratio of MCP tools to HTTP routes staying lopsided; (2) the HTTP route table in `build_app` stays small and changes only to track upstream API compat, not to add LAMU-specific verbs; (3) the orchestration gates (routing mode, queues, journal) remain enforceable at the single MCP dispatch site. Revisit if a credible non-agent caller needs an orchestration verb (e.g. a web UI wanting council results), at which point the right move is likely a separate authenticated service, not bolting verbs onto the unauthenticated OpenAI-compat surface. Also revisit if the hand-rolled JSON-RPC loop starts costing more in protocol-drift bugs than an MCP SDK would.

