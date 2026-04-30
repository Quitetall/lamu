# Token Efficiency — 70-80% Savings

## The Strategy

Cloud AI (Claude) plans and reviews. Local AI (Qwen) implements. You pay only for the thin reasoning layer.

## Token Breakdown Per Swarm Run

```
PLANNER (Claude Opus):     ~2K tokens    (short JSON plan)
WORKERS (local Qwen):      ~8-16K tokens (bulk code generation — FREE)
CRITIC (Claude Sonnet):    ~1K tokens    (short JSON review)
```

Workers do 80% of generation, all local. Cloud handles the 20% that needs real reasoning.

## MCP Delegation Pattern

In Claude Code, use `query_local_llm` for:
- "Write a function that does X" → local generates, Claude reviews
- "Refactor this to use Y pattern" → local drafts, Claude verifies
- "Generate tests for this module" → local writes, Claude checks coverage

## Daily Cost Impact

```
Without local:  ~500K cloud tokens/day = ~$7-15/day
With local:     ~100K cloud tokens/day = ~$1.50-3/day
Savings:        70-80%
```

## Graphify Additional Savings

Knowledge graph reduces file searching by 71.5x (measured on mixed corpus). Instead of grepping through every file, Claude checks the graph first and navigates directly to relevant code.

## Wiki Additional Savings

Accumulated findings prevent re-derivation. "Why doesn't vLLM work on 24GB?" is answered from the wiki in ~200 tokens instead of re-discovering it through trial and error (thousands of tokens of failed attempts).
