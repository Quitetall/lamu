# ADR 0032: Memory-as-a-service HTTP surface + owner scoping

## Status

Accepted 2026-06-12

## Context

Katana's harness spec makes durable memory a kernel NON-goal: memory is
an out-of-process extension its agents consume over a protocol (§10).
LAMU already owns a temporal fact store (valid-time, supersede, hybrid
recall — ADRs 0028/0030/0031) but it was reachable only through MCP
tools, and every row belonged to the implicit single user. ADR 0018
deferred its P4 (memory owner-scoping) "until a shared/HTTP memory
service is built" — this is that service.

## Decision

Four JSON routes on `lamu serve`, inside the existing bearer/quota
layers: `POST /v1/memory/{remember,recall,forget,supersede}`. Owner =
`Principal.user` under `AuthMode::KeyStore`, `"local"` under
StaticToken/Off — and `"local"` for every MCP/autocapture/CLI path
(`LOCAL_OWNER`), so MCP-written facts and an unauthenticated serve see
one namespace, while each API key is an isolated tenant. The owner
parameter is plumbed through every lamu-memory read/write leg — vector,
FTS (filtered on the JOIN side; the FTS table carries no owner), recency
fallback, novelty dedup, hydration, corpus export. Cross-owner ids
behave exactly like missing ids: `forget` → `forgotten:false`,
`supersede` → 404 `not_found`, with NOTHING inserted (its return became
`Option<i64>` so a failed ownership check can't half-apply). Novelty
dedup is owner-scoped — cross-owner dedup would leak existence. The
persistent .tv index stays per-(store, model) WITHOUT owner partitioning;
the SQLite post-filter (now owner-aware) is the fence, with per-owner
over-fetch scaling noted as the multi-tenant follow-up. HTTP `remember`
is the PLAIN write (novelty is the autocapture path's concern); writes
charge quota by the existing len/4 approximation; recall is free in v1.
`lamu clean` reports lamu.db + index/ as unmanaged live state and gains
an explicit `--legacy-dbs` category (NOT in `--all` — deleting databases
deserves its own flag) for the import-source leftovers, offered only
once lamu.db exists.

## Rationale

- Owner enforcement lives in the storage layer's SQL, not in handlers —
  a future route can't forget it, and MCP keeps single-user semantics by
  passing a constant.
- Cross-owner-as-missing (vs 403) prevents id-space probing from
  enumerating other tenants' facts.
- Reusing the bearer/quota/audit middlewares means the memory surface
  inherits ADR 0018's identity, limits, and who-did-what logging for
  free.

## Alternatives Considered

- **Memory over MCP only** — katana's MCP adapter could bridge it, but
  per-key tenant isolation doesn't exist on stdio MCP (one process, one
  user), and HTTP is the surface external harnesses already hold a
  bearer for. Rejected as the only path; MCP tools remain for local use.
- **Per-owner .tv indexes** — isolation already guaranteed by the SQL
  fence; index-per-tenant is an optimization to revisit under real
  multi-tenant load. Deferred.
- **403 on foreign ids** — leaks existence. Rejected.

## Consequences

- An external harness gets remember/recall/forget/supersede with per-key
  isolation and documented degradation (no embedder → FTS+recency).
- `MemoryHit` gained `valid_until` (additive; MCP wire unchanged).
- Reembed is deliberately owner-less (operator action; model identity is
  store-wide).
- docs/API.md gains the Memory API section; the provider-embedding
  contract points at it.

## Related Decisions

ADR 0018 (identity/quotas/audit — resolves its deferred P4), ADR 0028
(owner column), ADR 0030 (embedder chain the routes ride), ADR 0011
(content fencing stays at consuming boundaries).

## Validation

11 new tests: owner isolation across every recall leg (vector via a
const embedder so ONLY the owner filter separates tenants, FTS,
recency), cross-owner forget/supersede as-missing, owner-scoped novelty,
export filtering; HTTP integration — 401 on all routes, two-key
isolation e2e, Off-mode sees MCP-written local facts, malformed-body
400, k clamp; clean gating/pairing. Workspace 767 green.
