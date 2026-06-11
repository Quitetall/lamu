# ADR 0026: `backend_kind` string dispatch key + composition-root drift test

## Status

Accepted 2026-06-11

## Context

ADR 0023's migration order called for a string dispatch field on the
registry wire DTO so module backends (lamu-image's `comfyui`, lamu-tts's
`fish_speech`, and anything future) can be targeted by registry entries
without core naming them. Until now `make_backend` matched only the
`BackendType` enum, with two variants hard-coded to forward into the
string-keyed module registry — two dispatch namespaces, glued by hand.
Nothing verified the glue: adding an enum variant or a module backend
without registering it at the composition root surfaced only as a runtime
`"not registered"` error on first load.

## Decision

`ModelEntry`/`ModelEntryYaml` gain `backend_kind: Option<String>`
(`skip_serializing_if = None` — old registries stay byte-stable).
`make_backend` dispatches on ONE string: `backend_kind` when present (a
mismatch with the enum logs a warning and the string wins — never a silent
pick), else `BackendType::as_kind_str()`, a canonical-string method pinned
to the serde wire names by a unit test. Builtin kinds resolve to core
backends; every other string resolves through the ADR 0023 module
registry. `merge_registry` preserves `backend_kind` as a curated field
(operator-set wins; a scan never sets it). A composition-root drift test
(`lamu-cli/tests/composition_root.rs`) mirrors `main()`'s `register()`
calls and proves all six kinds resolve without the `"not registered"`
error, plus that an unknown kind still fails loudly.

## Rationale

- One dispatch namespace ends the enum-vs-registry split: the enum is now
  a typed alias for a canonical string, and the string is the key.
- `Option<String>` (not a widened enum) keeps core decoupled from modules
  per ADR 0023 — core cannot enumerate kinds it doesn't know.
- Backward compatibility is structural: absent field → enum path,
  byte-identical to pre-0026 behavior; absent field also never serializes.
- The drift test is the load-bearing piece: registration happens at the
  composition root by convention, and conventions need tripwires. The
  serde-name ↔ `as_kind_str` unit test closes the other drift axis
  (renaming a serde tag without the matching string).

## Alternatives Considered

- **Widen `BackendType` with a `Custom(String)` variant** — makes the enum
  non-`Copy`, churns every match, and still needs the string registry for
  resolution. More invasive for the same dispatch outcome.
- **String-only (drop the enum)** — loses compile-time exhaustiveness for
  the builtin kinds and would churn every registry file and construction
  site at once. The enum stays as the typed default; deprecating it can be
  its own ADR if module kinds ever dominate.
- **Registry-side validation (reject unknown `backend_kind` at load)** —
  rejected: registration order is a composition-root property, and the
  registry loads before modules register in some paths; load-time
  validation would false-positive. Dispatch-time loud failure + drift test
  covers it.

## Consequences

- Operators can route an entry to any module backend via one YAML line;
  typos fail loudly at load time with the kind named.
- `BackendType::as_kind_str` must grow an arm per new variant (compiler
  enforces) and the drift test list must grow with it (test names the
  variants explicitly, so a new variant fails the count nowhere — keep the
  list in sync when adding kinds; the serde-agreement loop is the
  tripwire that does fail).
- Conflicting `backend` + `backend_kind` resolves to the string with a
  warning — documented, deterministic, never silent.

## Related Decisions

ADR 0023 (module architecture — the registry being keyed), ADR 0025
(registry relocation — where the YAML lives).

## Validation

- `cargo test -p lamu-core`: serde-name agreement for all six kinds,
  unregistered-kind loud failure, enum-fallback unchanged, YAML roundtrip
  (+ absent stays absent), merge preservation.
- `cargo test -p lamu-cli --test composition_root`: all six kinds resolve
  after `register()`; unknown kind still fails loudly.
- Revisit trigger: a third dispatch consumer (beyond make_backend +
  list_models) wanting kind metadata → consider a structured descriptor.
