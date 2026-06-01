# ADR 0018: Multi-user — per-token identity, per-user memory namespacing, and quotas (supersedes ADR 0012)

## Status

Accepted 2026-05-31

**Supersedes ADR 0012** (single static bearer). The 0012 single-token code path
survives as one auth mode; see Consequences.

## Context

ADR 0012 added one optional static bearer token and explicitly stated the revisit
trigger: *"if LAMU grows multiple clients needing distinct keys (→ a per-token
store, a new ADR superseding this)."* That trigger has fired. The operator now
wants genuine multi-user support: distinct, revocable credentials per user;
per-user quotas and fairness; usage attribution; and the ability to scope memory
by owner. ADR 0012 was not wrong — it was correctly scoped to single-user — but
its premise no longer holds.

The single most important fact for this design is the **structural split between
the two surfaces**, established by ADR 0001:

- **lamu-api (HTTP)** is the OpenAI/Anthropic/Ollama-compat inference proxy. It
  has the *only* auth mechanism (one static bearer, `auth.rs:31 resolve_token`,
  `:52 require_bearer`, constant-time compare). `AppState`
  (`openai_compat.rs:115`) holds nothing user-related. Critically, this path
  **never touches memory** — it forwards straight to the loaded llama-server
  (`:469`, `resolve_and_ensure_loaded:286`).
- **lamu-mcp (stdio JSON-RPC)** owns **all** memory and the per-model queues.
  Per CLAUDE.md and verified: **each Claude Code instance spawns its own
  lamu-mcp subprocess.** There is no bearer, no HTTP, no network identity here —
  the principal is implicitly *the OS user who launched the process.*

Memory is global, no owner column: conversation memory (`memory.rs`,
`~/.local/share/lamu/conversations.db`) keyed by caller-supplied
`conversation_id` (path-traversal allowlist at `:50`); lifetime memory
(`lifetime_memory.rs`, `~/.local/share/lamu/memory.db`) a global fact store with
schema `memories(... source, ts, valid_from, valid_until, supersedes)` — `source`
is freeform, **not an authenticated principal**, and `recall_memory` searches
across all facts.

The queue already has `Strategy::Priority` with `QueueRequest.priority`
(`queue.rs:14-27,92`) — but it is MCP-side only, HTTP never enqueues, and
priority is always 0. Metrics label by `(model, status)` only — no per-user
attribution exists. The bearer authenticates the *connection*, identifies no
*principal*, and propagates nowhere (memory lives in the other process). No
accounts/sessions/bcrypt/totp/api_keys store exists anywhere (greenfield).

The structural catch, stated loudest: **auth lives on HTTP; memory lives in a
per-process stdio MCP with no remote identity.** Today every Claude Code user
already has memory isolation *for free* — each runs their own lamu-mcp against
their own `~/.local/share` files (OS-user isolation). There is no shared
multi-tenant memory server to secure. So "multi-user" is primarily an
HTTP-inference-surface feature; "multi-user memory" only becomes a real problem
if a shared/HTTP memory service is built.

## Decision

Add **per-token identity + quotas + audit on the HTTP surface**, calibrated to a
machine-to-machine JSON API — **the API key *is* the account**, like OpenAI's
`sk-...`. Explicitly **out of scope** (a browser threat model LAMU does not
have, same reasoning ADR 0012 used and still valid): bcrypt passwords, TOTP/2FA,
backup codes, browser sessions, CSP/nonces, login forms. Memory owner-scoping is
**designed and recorded but built only when a shared memory service appears.**

1. **Key store** (new `keys.db`, 0600, alongside `api-token`):
   `api_keys(id, user, token_hash, token_prefix, created_at, last_used_at,
   revoked_at, daily_token_quota, priority)`. Store **SHA-256 of the full
   `lamu_<64hex>`**, never plaintext (a 256-bit random token has no brute-force
   surface, so SHA-256 suffices; no bcrypt/argon2 needed). `issue(user)` returns
   plaintext **shown once**; `revoke`, `list` (prefix only), `verify(token) ->
   Option<Principal{user, key_id, priority, quota}>` by hash lookup.

2. **AuthMode.** Replace `AppState.auth_token: Arc<Option<String>>` with
   `auth: Arc<AuthMode>` where `AuthMode = Off | StaticToken(String) |
   KeyStore(Arc<KeyStore>)`. **`StaticToken` stays the default** so every
   ADR-0012 deployment is byte-identical; `KeyStore` is opt-in.

3. **require_bearer branches on AuthMode** (`auth.rs:52`): `Off` → pass;
   `StaticToken` → existing `ct_eq` path; `KeyStore` → `verify(presented)`, on
   success insert `Principal` into request extensions for downstream handlers,
   on failure the same surface-correct 401 envelopes (`:86-92`). Update
   `last_used_at` fire-and-forget; reject revoked keys; dummy-hash on miss to
   avoid user-enumeration timing. An in-memory `HashMap<token_hash, Principal>`
   cache (invalidated on revoke) avoids a per-request DB read.

4. **Quotas + fairness** (new `quota.rs`): extract `Principal` from extensions
   in the chat/embeddings handlers; an in-memory token-bucket keyed by user
   (refilled by `daily_token_quota`) → **429** on exhaustion. Then, optionally
   and behind a flag, wrap the HTTP forward path with the **existing lamu-core
   `RequestQueue` `Strategy::Priority`** keyed per model, enqueuing at
   `principal.priority` — the only place HTTP gains an admission queue.

5. **Audit**: add a bounded-cardinality `user` label to `requests_total` /
   `tokens_generated_total` (`metrics.rs:50/67`), and emit a per-request
   structured tracing event `{user, key_prefix, model, route, status,
   prompt_tokens, completion_tokens, ts}` to journald — **that is the audit
   trail**; an optional durable `usage` table in `keys.db` is a later add.

6. **Memory owner-scoping (designed, deferred):** add
   `owner TEXT NOT NULL DEFAULT 'default'` to `lifetime_memory.memories` via an
   idempotent `ALTER` mirroring `migrate_temporal_columns:73`, thread `owner`
   through `remember`/`recall_memory`/`forget` so global search filters
   `WHERE owner = ?`; conversation memory scoped via owner-prefixed
   `conversation_id` (the existing allowlist already validates it). **Built only
   when a single process serves multiple principals' memory — which the
   stdio-per-process MCP does not do today.**

7. **CLI**: `lamu auth issue --user NAME`, `lamu auth list`, `lamu auth
   revoke <prefix>`, extending the existing `lamu auth init`.

## Rationale

- **The credential is the account.** Machine clients cannot do interactive 2FA
  and never render LAMU's HTML (there is none). Passwords/TOTP/CSP defend a
  browser threat model LAMU still doesn't have — exactly ADR 0012's reasoning,
  unchanged. A random 256-bit key revocable per row *is* genuine multi-user for
  an API.
- **Hash the tokens.** ADR 0012's single token is plaintext in a 0600 file (ADR
  0013 deferred at-rest encryption). A multi-user `keys.db` is a juicier target,
  so store SHA-256 and show plaintext once. Hashing also removes the constant-
  time requirement (hash lookup is content-independent); a dummy-hash on miss
  still defeats user-enumeration timing.
- **Reuse, don't reinvent, fairness.** `queue.rs` already has a tested
  `Strategy::Priority`. Wiring `principal.priority` into a per-model HTTP queue
  reuses it verbatim; the token bucket is plain in-memory arithmetic. Negligible
  perf cost (one cached principal lookup, in-memory bucket).
- **Bounded-cardinality `user` label is safe for Prometheus** (few users) and
  gives per-user attribution; the structured tracing event is the durable
  who-did-what without a new audit DB at this scale.
- **`StaticToken` default = zero migration.** Existing `LAMU_API_TOKEN` /
  `api-token` deployments and `tests/http.rs` are byte-for-byte unaffected;
  `KeyStore` is opt-in. The 0012 code path is *kept*, not deleted.
- **Don't build memory plumbing speculatively.** OS-user + per-process isolation
  already gives each Claude Code user private memory. The owner column matters
  only for a shared/HTTP memory service that doesn't exist; recording the
  idempotent migration makes it a clean later add (ADR-0014-style honesty about
  not building hardware/services that aren't here).

## Alternatives Considered

- **Port Odysseus's full stack** (users, bcrypt passwords, sessions, TOTP,
  backup codes, CSP/nonces, login rate-limit). Rejected: every piece exists for
  a multi-user *browser* app. LAMU serves a JSON API to harnesses; pulling these
  in defends threats it doesn't face — the identical rejection ADR 0012 made,
  still correct.
- **Keep the single static token.** Rejected: it authenticates a connection, not
  a principal — no per-user quota, attribution, or revocation. This is precisely
  the revisit trigger ADR 0012 named.
- **Store tokens in plaintext (as 0012 does for the single token).** Rejected
  for multi-user: a stolen `keys.db` would leak every user's live credential.
  Hashing costs nothing for a random token and contains the blast radius.
- **Build memory owner-scoping now.** Rejected/deferred: each principal already
  runs their own MCP process against their own files; there is no shared memory
  surface to scope. Building the owner plumbing speculatively adds churn across
  `remember`/`recall_memory`/`forget` for a service that doesn't exist. Recorded
  as a clean migration for if/when an HTTP memory API lands.
- **Per-user rate limiting in a reverse proxy instead of in-process.** Rejected
  as not real multi-user: a proxy can rate-limit by IP/token but cannot read
  per-key `daily_token_quota` from the LAMU key store or feed `principal.priority`
  into the existing queue. Identity must reach the handler.

## Consequences

- **`StaticToken` survives as one `AuthMode` variant** — 0012's path is not
  deleted, just demoted to the default. Loopback no-token users see zero change.
- **Every `AppState` constructor** (incl. the `tests/http.rs` fixture) must set
  the new `auth` field — the same churn 0012 already paid for `auth_token`.
- **The off-loopback hard-fail gate (`lib.rs:34`) must treat
  `KeyStore`-with-≥1-active-key as "auth configured"**; an **empty** key store
  off-loopback must still hard-fail exactly like no-token. This is the one
  security subtlety that, if missed, reopens ADR 0012's closed liability.
- **Revocation must be immediate** — the in-memory principal cache is invalidated
  on revoke. Plaintext is shown once on issue, never again (list shows prefix).
- **The `user` metrics label touches every `with_label_values` site** for the
  two affected series; quota 429s must not leak other users' usage.
- **Multi-user memory is explicitly NOT delivered** by this ADR. It is a designed,
  recorded migration that activates only with a shared memory service; until
  then OS-user/per-process isolation is the memory boundary.
- **New direct dependencies/surface**: a SQLite `keys.db`, `quota.rs`, three CLI
  verbs. Phases 1-3 are a few hundred LOC concentrated in lamu-api — medium, not
  the Odysseus stack.

## Related Decisions

ADR 0012 — **superseded by this ADR** (its single-token path lives on as
`AuthMode::StaticToken`; update its Status line to "Superseded by 0018"). ADR
0005 — loopback-default bind; together with this they remain LAMU's network
posture, and the off-loopback gate must understand the new `KeyStore` mode. ADR
0013 — at-rest encryption still deferred, but this ADR resolves the worse half by
hashing stored tokens. ADR 0001 — the MCP/HTTP split that makes multi-user an
HTTP-surface feature and per-process MCP single-principal. ADR 0016 — the BYO-
frontend contract; per-token keys make multiple frontends/users first-class
without LAMU growing a UI.

## Validation

- **`tests/http.rs`** keeps all single-token cases and adds: issue→verify
  roundtrip, revoked key → 401, unknown key → 401, `StaticToken` mode still
  passes existing cases, over-quota user → 429, higher-priority request dequeues
  first (mirror `queue.rs` `priority_first`). The empty-keystore-off-loopback
  hard-fail is asserted explicitly.
- **Metrics** carry the bounded `user` label; the per-request structured event is
  emitted with `user`/`key_prefix`/token counts.
- We know this is right when multiple users hold distinct revocable keys with
  enforced per-user quotas and attributable usage, and existing single-token
  deployments are unchanged. We'd know it was wrong if quota/attribution leaks
  across users, if the off-loopback gate regresses, or if a real shared-memory
  multi-tenant requirement emerges (→ build the deferred owner-scoping, possibly
  a further ADR for the memory service itself).

