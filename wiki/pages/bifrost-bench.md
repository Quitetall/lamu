# Bifrost gateway overhead

Empirical measurements of Bifrost (`:8080`) as a proxy in front of the
local model on `:8020`. Run `bash scripts/bench-bifrost.sh` to append a
new entry to this file.

The v3 path-consolidation plan keeps Bifrost iff overhead is under 3%
of total request latency (LAN-local proxy on the same host). Above 3%,
Bifrost gets stripped from the runtime path.

## Run on 2026-05-06T05:30:00Z

| Path | Mean (ms) | Median (ms) | Tokens (10×32) | Tokens/s |
|------|----------:|------------:|---------------:|---------:|
| Direct (:8020)         | 833 | 833 | 320 | 38.4 |
| Through Bifrost (:8080) | 847 | 840 | 320 | 37.8 |

**Bifrost overhead: +1.67%.** Verdict: **KEEP**.

- Model: heretic-v2-Q4_K_M (`qwen/qwen3.6-27b-uncensored` route on Bifrost; `qwen3.6-27b` on direct).
- N=10 runs · max_tokens=32 · temperature=0 · prompt: `Reply with only the digit 7.`
- Bifrost configured with provider routes: anthropic, openai (cloud), qwen→:8020, dflash→:8000, sglang→:8001, gpt2→:9001. Useful as a unified cloud+local gateway.
- v3 daemon does NOT route through Bifrost — Bifrost is parallel infrastructure for clients that want cloud + local under one OpenAI surface.

**Decision:** Bifrost stays in `scripts/serve-bifrost.sh` (not legacy). Document its role in `wiki/pages/mcp-setup.md` as the cloud+local unified gateway, distinct from `lamu serve` which is local-only.
