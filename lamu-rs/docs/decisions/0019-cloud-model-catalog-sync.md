# ADR 0019: Cloud-model catalog auto-sync from OpenRouter + per-provider endpoints

## Status

Accepted 2026-06-01

## Context

`cloud-models.yaml` (the cloud-model registry, ADR 0007) was hand-maintained, so
it went stale the moment a provider shipped a model — e.g. `claude-opus-4.8`
landed and the operator would have to add the row by hand. The operator asked
for an automatically-current list, ideally one someone else maintains that LAMU
just pulls.

Two facts make this easy:
- **OpenRouter `https://openrouter.ai/api/v1/models`** is an **auth-free**,
  continuously-maintained cross-provider catalog — every provider's models in
  one JSON array with `id` (`provider/name`), `context_length`, pricing, and
  `knowledge_cutoff`. Verified live: it already lists `anthropic/claude-opus-4.8`,
  `openai/gpt-5.4`, `deepseek/deepseek-v4-pro` (345 models total at time of
  writing). LAMU already routes OpenRouter ids through the OpenAI-compat path
  (ADR 0007), so a pulled entry is immediately callable.
- **Each direct provider** exposes its own `/v1/models` (OpenAI-compat Bearer;
  Anthropic uses `x-api-key` + `anthropic-version`) — the authoritative list of
  what *that* key can actually call.

## Decision

Add `lamu cloud sync` (lamu-cli `cloud_sync.rs` + `cloud_config::save_models`).
It (1) pulls the OpenRouter catalog (auth-free) into `openrouter`-routed
`CloudModel`s, (2) pings each distinct provider already configured in
`cloud-models.yaml` that has a present key for its own `/v1/models`, and (3)
**merges preservation-first** into `cloud-models.yaml`: an existing entry keeps
every hand-set field (name, model_id, base_url, api_key_env, chat_path, notes,
quota); only an unset `context_max` is filled; new models are appended; nothing
is deleted. Flags: `--no-ping`, `--no-openrouter`, `--dry-run`. Run on demand or
on a schedule (cron / `/schedule`) to stay current.

## Rationale

- **Don't maintain a model DB — pull the one that exists.** OpenRouter curates
  the cross-provider catalog continuously; new models (Opus 4.8) appear within
  days without any LAMU change. That is exactly the "someone else maintains it"
  the operator wanted.
- **Pull + ping are complementary.** OpenRouter gives breadth + metadata
  (context, pricing) but routes through OpenRouter; per-provider `/v1/models`
  gives the authoritative directly-callable list for each configured key. Both
  feed one merge.
- **Preservation-first merge is non-negotiable.** The operator's aliases,
  routing prefs, and notes must survive a sync — so the merge only *adds* and
  *fills unset context*, never overwrites a hand-set field or deletes a row.
- **Reuses existing routing.** OpenRouter entries are callable via ADR 0007's
  OpenRouter branch with no new plumbing; direct-provider entries carry their
  base_url + api_key_env so they call direct.

## Alternatives Considered

- **Keep it manual.** Rejected — that's the staleness problem.
- **Vendor a static model DB (models.dev / LiteLLM's
  `model_prices_and_context_window.json`).** Good secondary sources for
  pricing/context cross-checks, but they're snapshots to re-vendor; OpenRouter's
  live endpoint is fresher and needs no commit to update. Kept as optional
  enrichment, not the primary.
- **Per-provider `/v1/models` only (no OpenRouter).** Misses cross-provider
  discovery + requires a key for every provider just to *see* its models.
  OpenRouter needs no key to enumerate.
- **Auto-write on startup.** Rejected — a network fetch on every boot is
  surprising + slow; sync is an explicit command (schedulable).

## Consequences

- A full sync adds the entire OpenRouter catalog (~345 rows) to
  `cloud-models.yaml` — comprehensive but large; the TUI model list grows
  accordingly. `--no-openrouter` (ping-only) or post-sync pruning keeps it
  curated if desired.
- OpenRouter-routed entries require `OPENROUTER_API_KEY` to actually *call*
  (enumeration is free; inference is not).
- Sync is best-effort: a provider ping or the OpenRouter pull failing logs a
  warning and continues with whatever else succeeded (never corrupts the file).
- Staying current is now a scheduled command, not a code change.

## Related Decisions

ADR 0007 (unified cloud routing — OpenRouter branch makes pulled entries
callable), ADR 0018 (provider keys / api-keys.env this reads), ADR 0016 (the
broad-compat backend-orchestrator stance this serves).

## Validation

`cloud_sync::tests` pin the merge contract (existing fields preserved, new added,
unset context filled, `/v1/models` URL built tolerating a `/v1` suffix). A live
`lamu cloud sync --dry-run` reports counts without writing. Revisit if a provider
drops `/v1/models` or OpenRouter's schema changes (the minimal `{data:[{id,
context_length}]}` parse is resilient to added fields).
