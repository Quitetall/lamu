# ADR 0027: Typed `ToolCtxError` at the ToolCtx seam, "error:" strings stay wire-only

## Status

Accepted 2026-06-11

## Context

The ToolCtx seam (ADR 0023) was stringly-typed: `ensure_loaded` and
`generate` returned plain `String`s and signalled failure with an
`"error:"` prefix that ~65 call sites across 7 crates had to sniff. The
sniffing was inconsistent — lamu-jart checked
`.trim_start().to_lowercase().starts_with("error:")`, lamu-image/lamu-tts
checked `.starts_with("error")` without the colon and could false-match
prose ("Error bars on the chart…"). A missed check silently treated an
error message as model output; the compiler couldn't help. `embed`
already returned `Result<_, String>` — a third convention.

A constraint shapes the migration: the MCP wire contract is one text
string per tool response, with `isError` inferred from the `"error:"`
prefix at the dispatcher (server.rs). Tool handlers
(`ModuleToolHandler -> String`) ARE the wire layer.

## Decision

`ToolCtx::ensure_loaded` and `generate` return
`Result<String, ToolCtxError>`; `embed` returns
`Result<Vec<Vec<f32>>, ToolCtxError>`. `ToolCtxError` is a three-variant
enum (`Load`/`Generate`/`Embed`) carrying the bare failure message (no
prefix); `Display` prints the message only. Tool handlers keep returning
wire strings and compose their own framing
(`format!("error: <step>: {e}")`), so `ModuleToolHandler`, the dispatcher
`isError` heuristic, and every MCP client-visible string shape are
unchanged. The server impl bridges legacy handlers
(`handle_load_model`/`handle_query`/`handle_cloud_query`, which still
speak the prefix convention internally) via two shared helpers in
`tools_ext`: `is_wire_error` (the one canonical sniffer, colon required)
and `ToolCtxError::strip_wire_prefix`.

## Rationale

- Type safety where it was breaking: consumers of the SEAM can no longer
  forget an error check — `Result` forces the match, and the ~20 seam
  call sites dropped their hand-rolled sniffing.
- The colon-less `starts_with("error")` bug class in lamu-image/tts is
  structurally retired (those sites now match on `Err`).
- Keeping `ModuleToolHandler -> String` bounds the blast radius to the
  seam: no dispatcher change, no wire change, no MCP client change,
  dispatch_smoke's wire-shape assertions pass untouched.
- Bare-message payloads end the nested `"error: step: error: inner"`
  stutter — handlers add exactly one prefix at the wire boundary.
- Defensive `"error:"` checks on MODEL OUTPUT (parse_queries,
  parse_subquestions, normalize_verdict) deliberately stay: a model can
  literally generate the text "error: …", and those parsers treat it as
  an undecidable reply, which is about content, not the seam.

## Alternatives Considered

- **Migrate `ModuleToolHandler` to `Result` too** — would push typed
  errors one layer further, but the handler return IS the wire string;
  the dispatcher would convert straight back. Churn in every tool
  registration for zero information gain. Revisit only if the MCP layer
  ever wants structured errors.
- **Rich error struct (operation + model + cause fields)** — more
  structure than any consumer uses; every site just formats the message.
  The variant already names the operation. YAGNI.
- **`lamu_core::Error` reuse** — conflates backend/infra errors with the
  seam contract and would force seam consumers to handle variants that
  can't occur there.

## Consequences

- The trait broke once: all impls (LamuMcpServer, FakeCtx in
  lamu-jart-frontend) and ~20 seam call sites updated in one commit
  (intermediate states can't compile across a trait change).
- Internal handler plumbing (handlers.rs, cloud.rs) still produces
  prefixed strings; the seam bridges them. Migrating handler internals to
  typed errors is possible later without touching the trait again.
- Error text on the wire is slightly cleaner (inner prefixes stripped),
  same information.
- GenerateOpts widening (next deferred item) now lands as a pure
  signature extension over already-`Result`-shaped call sites.

## Related Decisions

ADR 0023 (the seam being typed), ADR 0024 (dispatch boundary), audit
fixes A3/A4 (routing gate + empty-cloud flag now return `Err`).

## Validation

- `cargo test` across lamu-core / lamu-mcp / lamu-jart /
  lamu-jart-frontend / lamu-image / lamu-tts green; dispatch_smoke wire
  assertions unchanged and passing.
- Unit tests pin `is_wire_error` (colon required), `strip_wire_prefix`
  (leading only), and Display (bare message).
- Revisit trigger: MCP protocol-level structured errors → migrate
  `ModuleToolHandler` then.
