# LAMU — Claude Code Configuration

## Token Efficiency Rules

These rules minimize wasted tokens. Follow them strictly.

### 1. Check knowledge layers BEFORE reading files

Order of operations for any codebase question:
1. **Wiki first** — `wiki/index.md` has accumulated findings. Check before re-deriving.
2. **Graph second** — `graphify-out/GRAPH_REPORT.md` has architecture structure.
3. **Files last** — only read raw code if the wiki and graph don't have the answer.

### 2. Delegate bulk work to the local model

Use the `query_local_llm` MCP tool for:
- Drafting implementations (then review the output yourself)
- Generating boilerplate code
- Writing tests from existing implementations
- Refactoring existing code to a new pattern
- Any task where "write 100+ lines" is the main work

Keep for yourself (Claude):
- Architecture decisions and planning
- Debugging complex multi-file interactions
- Reviewing local model output for correctness
- Security-sensitive code review

This saves 70-80% of tokens — local model generates, you review.

### 3. Navigate by structure, not grep

For "how does X relate to Y" questions:
```bash
graphify query "<question>"
graphify path "<A>" "<B>"
graphify explain "<concept>"
```
These traverse the knowledge graph's edges. Faster and cheaper than grepping every file.

### 4. After modifying code

Run `graphify update .` to keep the graph current (AST-only, no API cost).

## Project Context

LAMU (Local Agent Model Utility) — framework for running AI models as local agents.

Key files:
- `server/serve.py` — production model server (think-block middleware, health endpoint)
- `server/mcp_qwen.py` — MCP server exposing local model to Claude Code
- `server/client.py` — Python SDK for the local model
- `server/poincare.py` — Poincaré ball embedding visualization
- `agents/swarm.py` — agentic coding swarm (plan→implement→test→review)
- `agents/trainer.py` — QLoRA fine-tuning pipeline
- `agents/bench.py` — benchmark runner
- `legacy/cli/chat_repl.py` — terminal REPL (v1; superseded by `lamu repl`)
- `web/app.py` — Chainlit web frontend
- `justfile` — all commands (43 recipes)
- `config/models.yaml` — model registry
- `lamu-rs/lamu-cli/` — canonical Rust binary (`lamu` on $PATH after `just install`)

Hardware: RTX 4090 (24GB), 64GB RAM, Arch Linux.
Model: Qwen3.6-27B Dense Uncensored Heretic v2, GGUF Q5_K_S, 108K context.
Engine: llama-cpp-python with flash attention + Q8_0 KV cache.
Fast mode: native llama-server with ngram-mod speculation (40-137 t/s).

## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md
- For cross-module questions, prefer `graphify query` / `graphify path` / `graphify explain` over grep
- After modifying code files, run `graphify update .` to keep the graph current
