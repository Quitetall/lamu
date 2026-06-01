# ADR 0016: LAMU is a backend orchestrator; bring your own frontend via broad API compat

## Status
Accepted 2026-05-31

## Context
By the time this was written LAMU had a stable HTTP surface
(`lamu-api/src/openai_compat.rs::build_app`, lines 135-159) exposing eight routes —
`/health`, `/metrics`, `/v1/models`, `/v1/chat/completions`, `/v1/embeddings`, the
Anthropic `/v1/messages` shim, and the Ollama `/api/tags` + `/api/chat` shims — and a
26-tool MCP catalog (`lamu-mcp/src/tools.rs::TOOLS`) for orchestration. ADR 0001
already established that orchestration is MCP-first and HTTP is a thin compat shim.
What ADR 0001 did *not* state as a first-class product decision is the inverse, equally
load-bearing commitment: **LAMU ships no frontend of its own, and never will.** The
project is a backend that owns the genuinely hard, genuinely shared concerns — model
lifecycle, VRAM, and routing across a single GPU — and deliberately delegates the
human-facing surface to whatever harness the user already runs.

This needed to be written down because three forces kept pushing toward a built-in UI:
(1) every comparable local-LLM project (Ollama, LM Studio, Open WebUI, Jan) ships a UI
and users expect one; (2) the three HTTP dialects we already maintain are most of the
work a UI would need; (3) it is tempting to "just add a chat page." Committing to
*not* doing that — and instead doubling down on multi-dialect compatibility so that
**any** frontend is a first-class LAMU frontend — is the decision. The cost of building
a UI is not the page; it is the permanent obligation to a second product (theming,
auth/session for browsers, settings sync, accessibility, a release cadence) that
competes for the same single-maintainer attention as the scheduler and the reviewer.

The technical facts that make "be the backend" viable were already in place: the bind
defaults to loopback with a constant-time bearer gate (ADR 0005, ADR 0012); cloud
routing is a separate provider-direct egress, not an HTTP concern (ADR 0007); the HTTP
path never auto-evicts so a dumb client can't thrash VRAM (ADR 0006). The HTTP layer is
explicitly described in-source as a "direct port of `lamu/api/openai_compat.py`" —
boring on purpose.

## Decision
LAMU is a **backend orchestrator with no frontend of its own**. Its job is to own
model lifecycle (spawn/preload/health/quarantine of local `llama-server`
subprocesses), VRAM (the single-GPU scheduler with no-auto-evict on the HTTP path),
and routing (alias/capability resolution, the cloud egress). It exposes that pool
through **three concurrent HTTP dialects on one port** — OpenAI
(`/v1/chat/completions`, `/v1/models`, `/v1/embeddings`), Anthropic Messages
(`/v1/messages`), and Ollama (`/api/tags`, `/api/chat`) — plus `/health` and
`/metrics`. Any tool that already speaks one of those dialects becomes the user's
frontend by pointing its base URL at `http://127.0.0.1:<port>` (or `/v1`) and,
optionally, setting a bearer token. A stable alias namespace (`lamu` / `main` /
`default`, or an omitted `model`) lets a frontend pin its config without knowing which
model backs it. LAMU will not add a built-in web UI, chat page, or settings console;
the agent/orchestration plane stays on MCP (ADR 0001). The product surface a human
touches is, by design, **the user's choice of existing client**.

## Rationale
- **Compatibility is cheaper to maintain than a UI, and it composes.** Three dialect
  shims (the translation lives in `anthropic_message_to_openai`, `ollama_chat`, and the
  Vision-array `Message` deserializer) reach Claude Code, Open WebUI, AnythingLLM,
  Continue, LibreChat, every OpenAI/Anthropic SDK, and RAG front-ends via
  `/v1/embeddings` — for the cost of one HTTP crate. A built-in UI would reach exactly
  the users who like that one UI, while incurring a perpetual second-product tax.
- **The hard parts are the backend parts.** VRAM-aware single-flight loading, the
  no-auto-evict guarantee (ADR 0006), health/quarantine, and the GPU lock handshake
  with `lamu-train` are problems no frontend can solve and every frontend benefits from.
  Concentrating effort there is the highest leverage for a single-GPU, single-maintainer
  system.
- **Stable indirection via aliases decouples the frontend from the model.** Because
  `lamu`/`main`/`default` resolve to the `main:true` registry entry (and omitting
  `model` routes the same way), a user can swap the underlying GGUF without touching any
  frontend config. The response's `model` field still reports the real resolved name for
  observability.
- **Loopback-default + optional bearer makes "any client" safe by construction.** The
  unauthenticated surface is only reachable on `127.0.0.1`; going off-loopback
  hard-fails without a token (ADR 0005, ADR 0012). So "point your frontend at LAMU" is a
  one-line config on the common (local) path and a guarded, deliberate act on the LAN
  path.
- **It keeps the blast radius small.** Refusing a UI keeps browser-specific attack
  surface (CSP, sessions, cookies, CSRF) entirely out of scope, consistent with the
  minimal-bearer threat model. The orchestration verbs that *would* be dangerous on an
  open socket stay on the stdio MCP plane (ADR 0001).
- **Cloud stays off the HTTP contract on purpose.** Cloud models are reached only via
  the MCP `cloud_query` egress and a separate `cloud-models.yaml` (ADR 0007). The HTTP
  surface therefore makes one honest promise — "I serve the local pool" — and a frontend
  pointed at it can reason about cost and latency accordingly.

## Alternatives Considered
- **Ship a built-in web UI (the Ollama / LM Studio model).** A bundled chat page +
  model manager. Rejected: it is a second product with its own indefinite maintenance
  surface (theming, auth/session, settings, accessibility, release cadence) that
  competes with the scheduler/reviewer for single-maintainer time, and it would reach
  only the subset of users who prefer that one UI. The same engineering invested in
  dialect compat reaches every existing client. It would also drag browser security
  concerns into a deliberately minimal threat model.
- **Pick one dialect (OpenAI-only) and tell users to adapt.** Rejected: Claude Code
  speaks Anthropic Messages natively and AnythingLLM/Open WebUI frequently hardcode the
  Ollama surface. Supporting only OpenAI would exclude the two harnesses we most want as
  first-class frontends; the marginal cost of the Anthropic + Ollama shims is small
  relative to that reach.
- **Build a thin first-party adapter/proxy per popular frontend.** Rejected as
  N-products-instead-of-one: per-client adapters drift independently and each is its own
  bug surface. Implementing the *dialects* once, inside `lamu serve`, makes every client
  that already speaks a dialect work without a LAMU-specific shim.
- **Expose orchestration over HTTP so a generic UI could drive council/review/memory.**
  Rejected by ADR 0001 and reaffirmed here: that would put dangerous verbs on an
  unauthenticated socket and reinvent MCP's typed-arg/discovery/`isError` semantics as
  ad-hoc JSON. If a non-agent UI ever needs an orchestration verb, the right move is a
  separate authenticated service, not bolting it onto the dumb compat surface.

## Consequences
- **The HTTP route table is the frontend contract** and must stay small and
  compatibility-driven. Adding LAMU-specific verbs to it is a regression of this ADR;
  changes to `build_app` should only track upstream dialect compat.
- **We owe perpetual fidelity to three moving upstream specs.** Quirks like Vision
  content-array flattening, Anthropic tool block expansion, `<think>` splitting,
  Ollama's `stream`-defaults-true and NDJSON framing are permanent maintenance, and
  they are where compat bugs live. Cross-surface inconsistencies (error-envelope drift
  on `/v1/messages` and `/api/chat`, missing CORS for browser frontends, the
  no-`/api/version` gap) are the *known liabilities* this decision creates and obligates
  us to fix — they are tracked as API-hygiene work, not accepted as permanent.
- **No UI means no first-party answer to "I just want to click and chat."** The user
  must install a frontend. We accept that friction in exchange for not owning a UI; the
  docs carry an explicit "point your frontend at LAMU" table to minimize it.
- **Frontends inherit backend constraints they cannot see.** Auto-load won't
  auto-evict, vision content is dropped, multi-turn tool linkage degrades on the
  Anthropic bridge, and cloud models are absent from `/v1/models`. These must be
  documented loudly (they are, in docs/API.md "Footguns") because a frontend author has
  no other signal.
- **Browser-origin frontends are currently dead on arrival** (no CORS layer; tower-http
  is not a dependency). This is the single largest immediate gap created by promising
  "any frontend," and is the top item in the hygiene backlog.

## Related Decisions
ADR 0001 — MCP-first orchestration; HTTP serve as a thin compat shim (this ADR is its
product-framing complement: 0001 says orchestration stays on MCP, 0016 says the human
frontend stays off LAMU entirely).
ADR 0005 — Bind 127.0.0.1 by default (makes "any client" safe on the common path).
ADR 0012 — Minimal bearer auth (the off-loopback gate that makes LAN frontends a
deliberate, guarded act).
ADR 0006 — HTTP path never auto-evicts (the guarantee that lets a dumb frontend not
thrash VRAM).
ADR 0007 — Unified cloud routing (cloud is a provider-direct MCP egress, deliberately
absent from the HTTP frontend contract).

## Validation
This decision is right as long as: (1) real third-party frontends keep working against
`lamu serve` with zero LAMU-specific code — track by periodically smoke-testing Claude
Code (Anthropic), Open WebUI (OpenAI + Ollama modes), AnythingLLM, and a Chroma RAG
flow against the live server; (2) the `build_app` route table changes only to track
upstream dialect compat, never to add LAMU verbs; (3) the known compat liabilities
(CORS, per-surface error envelopes, validation-error shape) get closed rather than
accreting, measured by the API-hygiene backlog shrinking. We would know it was wrong if
a built-in UI became necessary because no third-party frontend could express something
LAMU needs to surface (at which point the right response is still a *separate*
authenticated service, not a UI welded onto the compat shim), or if maintaining three
dialects started costing more than the reach they buy. Revisit if a fourth dialect is
demanded by a frontend with meaningful adoption, or if browser-frontend support (CORS +
envelope consistency) cannot be made clean without compromising the loopback threat
model.
