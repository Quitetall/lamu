# ADR 0013: At-rest encryption of cloud keys / API token deferred

## Status

Accepted 2026-05-31

## Context

The security pass (ADR 0011, ADR 0012) added an injection boundary and bearer
auth. A remaining question from the Odysseus comparison: Odysseus encrypts
secret columns at rest with Fernet + an idempotent `enc:-` prefix migration.
LAMU stores cloud API keys in `~/.config/lamu/api-keys.env` and the bearer
token in `~/.config/lamu/api-token`, both written at mode `0600`, read into
process env / memory.

## Decision

Do NOT build at-rest encryption now. Keys and the token stay as `0600`
plaintext files. Record the design to use IF the stolen-backup-file threat
enters scope: a single app key at `~/.config/lamu/.app-key` (0600, `getrandom`
on first run); `encrypt()` no-ops on already-`enc:`-prefixed values,
`decrypt()` passes through un-prefixed (legacy plaintext) and fails soft to
`None` on bad data — making a one-time "encrypt in place" migration safe to run
on every startup with zero version tracking. `aes-gcm` or `age`, ~40 LOC.

## Rationale

- The threat at-rest encryption defends is a *stolen backup / leaked file
  whose `0600` perm was stripped*. On a single-user box where the same user
  runs LAMU and owns the files, an attacker who can read `0600` files in
  `~/.config/lamu` already has the user's session — encryption buys little.
- Every key is also reachable from the process environment at runtime; a local
  attacker at that tier wins regardless.
- The marginal security gain does not justify the key-management complexity
  (where does the app key live? it's plaintext too) for the current threat
  model. Recording the *design* now means the decision is deliberate, not an
  oversight, and the migration shape is ready if the calculus changes.

## Alternatives Considered

- **Build Fernet-style encryption now (port Odysseus).** Rejected: solves a
  threat LAMU doesn't currently face; the app key is itself an at-rest secret,
  so it mostly moves the problem. Complexity > benefit for single-user.
- **OS keyring (Secret Service / libsecret).** Rejected for now: adds a daemon
  dependency + headless/CI breakage; revisit only if encryption is warranted.

## Consequences

- Anyone with read access to `~/.config/lamu` (same-user, or a backup whose
  perms leaked) can read the keys + token. Documented, accepted for
  single-user.
- If/when built, the `enc:-` idempotent-prefix trampoline is the chosen shape —
  this ADR is superseded by the one that implements it.

## Related Decisions

ADR 0012 (bearer token — the other secret on disk), ADR 0005 (loopback default
— the primary network defense that makes this lower-priority).

## Validation

Revisit when LAMU runs somewhere the at-rest file is exposed to a different
trust principal than the LAMU process (shared host, synced/backed-up config,
multi-user). Until then, `0600` is the bar.
