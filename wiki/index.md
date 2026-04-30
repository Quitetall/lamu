# LAMU Wiki — Knowledge Base

Accumulated findings from building and optimizing the local LLM stack.

## Architecture
- [[serving-engine]] — Why llama-cpp-python wins on single 4090
- [[262k-context]] — How to achieve 262K context on 24GB VRAM
- [[ngram-speculation]] — ngram-mod speculative decoding (40-137 t/s)
- [[model-selection]] — Qwen3.6-27B Dense vs MoE vs Qwen3.5

## Hardware Constraints
- [[vram-budget]] — RTX 4090 VRAM breakdown for different configs
- [[vllm-limitations]] — Why vLLM can't fit 27B on 24GB

## Training
- [[eagle-training]] — EAGLE-3 head training pipeline
- [[training-loop]] — Self-improving feedback loop from swarm runs

## Integration
- [[mcp-setup]] — Claude Code MCP integration
- [[token-efficiency]] — How to save 70-80% of cloud tokens
