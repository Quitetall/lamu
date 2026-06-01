# ADR 0012: Minimal single-token HTTP bearer auth for lamu-api

## Status

Accepted 2026-05-31 · **Superseded by [ADR 0018](./0018-multi-user-per-token-identity.md)** 2026-05-31 — the single static token survives as one auth mode within 0018's per-token-identity model.

## Context

`lamu serve` exposed an OpenAI/Anthropic/Ollama-compat HTTP API with **no
authentication of any kind**. It was safe only because it bound `127.0.0.1`
by default (ADR 0005); the moment an operator set `LAMU_BIND_HOST=0.0.0.0`
(a documented, supported option), the API became an open inference + cloud-
credit-spending endpoint on the LAN, guarded by nothing but a
`tracing::warn!`. The comparison against Odysseus
(`docs/comparison-odysseus.md`) scored this a decisive loss: Odysseus ships
bcrypt + TOTP 2FA + per-token DB + sessions + CSP, all regression-tested.

But Odysseus is multi-user and browser-facing; LAMU is single-user, loopback-
default, and serves a JSON API to harnesses (Claude Code, RAG front-ends).
Porting accounts/sessions/2FA/CSP would be machinery defending a threat model
LAMU does not have.

## Decision

Add one optional static bearer token. `auth.rs::resolve_token()` reads
`LAMU_API_TOKEN` (env) then `~/.config/lamu/api-token` (0600 file); `None` →
auth off. `build_state` resolves it once into `AppState.auth_token`. One
`axum::middleware::from_fn_with_state` layer (`require_bearer`) on `build_app`
enforces it: no token configured → pass (frictionless loopback); token set →
every route except `/health` + `/metrics` requires `Authorization: Bearer
<token>`, compared in **constant time** (`subtle::ct_eq`); failure → 401 in
the OpenAI error-envelope shape. `serve()` **hard-fails at startup** if bound
off-loopback with no token, unless `LAMU_ALLOW_INSECURE=1`. `lamu auth init`
mints a `lamu_<64-hex>` token, writes it 0600, and prints it once.

## Rationale

- The loopback default (ADR 0005) is the trust boundary for the common case;
  auth must not add friction there, hence no-token → pass.
- The one real risk is an off-loopback bind. A startup hard-fail (not a warn)
  is the only thing that actually prevents the accident; the warn was ignored
  by definition (you don't see it until you've already exposed the port).
- Constant-time compare is the single correctness trap in a token check; a
  naive `==` leaks the token byte-by-byte via timing. `subtle` was already in
  the tree.
- `/health` + `/metrics` are exempt because probes (load balancers, Prometheus)
  legitimately carry no credentials and reveal nothing sensitive.
- OpenAI error-envelope 401 so compat clients (Claude Code, Open WebUI,
  AnythingLLM) surface the failure correctly instead of choking on a bespoke
  shape.

## Alternatives Considered

- **Port Odysseus's full stack** (users, bcrypt passwords, sessions, TOTP,
  per-token SQLite + prefix-bucket cache, CSP/headers, login rate-limit,
  owner-404s). Rejected: every piece exists for multi-user/browser; LAMU is
  single-user with a JSON API. It would be hundreds of LOC of attack surface
  defending threats LAMU doesn't face.
- **Auto-generate a token on first off-loopback bind.** Rejected: a server
  that mints a token nobody sees is useless — better to hard-fail with the
  exact command (`lamu auth init`) to run.
- **Keep the warn-only status quo.** Rejected: it's the documented liability
  this ADR exists to close; a warning does not prevent exposure.
- **Encrypt the token / cloud keys at rest now (Fernet-style).** Deferred (see
  Consequences): keys + token are already 0600; at-rest encryption defends a
  stolen-backup-file, a lower-priority threat. The `enc:-` idempotent-prefix
  migration pattern is recorded for if/when that threat enters scope.

## Consequences

- Loopback users see zero change. Off-loopback users must run `lamu auth init`
  (or set `LAMU_API_TOKEN`) or explicitly opt into `LAMU_ALLOW_INSECURE=1`.
- `AppState` gains `auth_token: Arc<Option<String>>`; every `AppState`
  constructor (incl. the `tests/http.rs` fixture) must set it.
- The token is plaintext in env or a 0600 file — adequate for single-user;
  at-rest encryption is a recorded follow-up, not built.
- No rotation/expiry/revocation list — rotation is "delete the file + `auth
  init` again". Acceptable for one operator; would need rethinking for
  multi-client key management (which would itself warrant a superseding ADR).
- `subtle` becomes a direct lamu-api dependency.

## Related Decisions

ADR 0005 (loopback-default bind — this builds on it; together they are LAMU's
network posture), ADR 0001 (the small HTTP surface this guards), ADR 0011 (the
other half of this security pass — the injection boundary).

## Validation

`tests/http.rs` pins the middleware: token set → no-bearer 401, wrong-bearer
401, right-bearer 200, `/health` 200; token unset → no-bearer 200. Revisit if
LAMU grows multiple clients needing distinct keys (→ a per-token store, a new
ADR superseding this), or if at-rest encryption becomes warranted.
