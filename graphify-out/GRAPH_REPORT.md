# Graph Report - local-llm  (2026-05-04)

## Corpus Check
- 27 files · ~24,677 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 314 nodes · 405 edges · 32 communities detected
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 17 edges (avg confidence: 0.62)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]

## God Nodes (most connected - your core abstractions)
1. `SQLiteDataLayer` - 26 edges
2. `_run()` - 15 edges
3. `LocalLLM` - 11 edges
4. `WikiRAG` - 11 edges
5. `main()` - 8 edges
6. `PoincareBallEmbedding` - 6 edges
7. `main()` - 6 edges
8. `ThinkStripASGI` - 6 edges
9. `probe_endpoint()` - 6 edges
10. `invoke_swarm()` - 6 edges

## Surprising Connections (you probably didn't know these)
- `chat_completions()` --calls--> `generate()`  [INFERRED]
  server/gpt2_proxy.py → server/eagle_server.py
- `WikiRAG` --uses--> `Probe all known endpoints and return available models.`  [INFERRED]
  server/rag.py → server/mcp_qwen.py
- `WikiRAG` --uses--> `Send a chat completion to Bifrost (routes to the right backend).`  [INFERRED]
  server/rag.py → server/mcp_qwen.py
- `WikiRAG` --uses--> `Probe all known endpoints and return available models.`  [INFERRED]
  server/rag.py → server/mcp_qwen.py
- `WikiRAG` --uses--> `Send a chat completion to Bifrost (routes to the right backend).`  [INFERRED]
  server/rag.py → server/mcp_qwen.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.08
Nodes (20): BaseDataLayer, build_llm(), extract_think(), on_chat_resume(), on_message(), python_repl(), Chainlit frontend — local-llm stack. Agentic: tool-calling loop with think displ, Agentic loop for dflash models:       - Shows thinking as a collapsible cl.Step (+12 more)

### Community 1 - "Community 1"
Cohesion: 0.09
Nodes (28): build_swarm(), commit_node(), critic_node(), fail_node(), integrator_node(), invoke_swarm(), llm(), load_context() (+20 more)

### Community 2 - "Community 2"
Cohesion: 0.12
Nodes (14): chat(), get_default(), LocalLLM, models(), Local LLM Python client — import from anywhere.      from server.client import L, Send a multi-turn conversation. Returns the response text., Stream tokens from a chat completion., Check if the LLM backend is reachable. (+6 more)

### Community 3 - "Community 3"
Cohesion: 0.12
Nodes (13): call_tool(), _chat(), _discover_models(), MCP server — exposes all local LLMs as tools for Claude Code (or any MCP client), Probe all known endpoints and return available models., Probe all known endpoints and return available models., Send a chat completion to Bifrost (routes to the right backend)., Send a chat completion to Bifrost (routes to the right backend). (+5 more)

### Community 4 - "Community 4"
Cohesion: 0.16
Nodes (11): chat(), ChatMessage, draft(), EagleHead, get_hidden_state(), LeanDecoder, main(), Lean EAGLE speculative decoding — llama-cpp GGUF + PyTorch EAGLE head.  Uses lla (+3 more)

### Community 5 - "Community 5"
Cohesion: 0.17
Nodes (10): chat_completions(), ChatMessage, ChatRequest, EagleHead, generate(), main(), predict_next(), Custom speculative decoding server using the trained EAGLE head.  Loads both the (+2 more)

### Community 6 - "Community 6"
Cohesion: 0.17
Nodes (12): create_poincare_plot(), load_graphify_json(), load_networkx_from_codebase(), main(), PoincareBallEmbedding, Poincaré ball embedding for knowledge graphs.  Takes graphify's graph.json and e, Load graphify's graph.json into a NetworkX graph., Build a graph from Python AST. Filters stdlib/third-party noise by default. (+4 more)

### Community 7 - "Community 7"
Cohesion: 0.17
Nodes (7): draft_multi(), EagleV3, HiddenStateDataset, main(), Predicts next hidden state from current hidden state + token embedding., hidden_state: [batch, seq, H] or [batch, H]         token_emb: [batch, seq, H] o, ResidualMLP

### Community 8 - "Community 8"
Cohesion: 0.19
Nodes (14): BaseModel, AnthropicMessage, AnthropicMessagesRequest, build_app(), ChatMessage, ChatRequest, main(), _parse_tool_calls() (+6 more)

### Community 9 - "Community 9"
Cohesion: 0.17
Nodes (9): chat_completions(), chat_completions(), ChatRequest, format_prompt(), generate(), get_decoder(), Message, OpenAI-compatible server for Qwen3.5-0.8B megakernel. 462+ tok/s on RTX 4090. Ru (+1 more)

### Community 10 - "Community 10"
Cohesion: 0.25
Nodes (14): auto_start(), cmd_models(), cmd_status(), discover_models(), get_available_models(), main(), probe_endpoint(), Start the LLM stack if nothing is running. (+6 more)

### Community 11 - "Community 11"
Cohesion: 0.21
Nodes (12): collect(), export(), main(), prepare(), Training pipeline — collect swarm data + QLoRA fine-tune local models.  Feedback, Run QLoRA fine-tuning with unsloth (optimized for consumer GPUs)., Merge LoRA + export to GGUF (for DFlash) or HF format (for vLLM)., Save a successful (task → implementation) pair for fine-tuning. (+4 more)

### Community 12 - "Community 12"
Cohesion: 0.22
Nodes (12): compare(), load_swebench_tasks(), main(), Agentic benchmark runner — compare Opus solo vs swarm on coding tasks.  Three mo, Load SWE-bench Lite tasks. Requires swebench package., Run a task with just Opus (cloud-only, no local workers)., Run a task with the full swarm (Opus planner + local workers)., Run a benchmark suite with a given config. (+4 more)

### Community 13 - "Community 13"
Cohesion: 0.23
Nodes (7): ctx_for_quant(), find_gguf(), kv_type_for_quant(), main(), Production Qwen3.6 server — native chat template, think-block middleware, health, Wraps the llama-cpp-python ASGI app and strips think blocks from responses., ThinkStripASGI

### Community 14 - "Community 14"
Cohesion: 0.18
Nodes (4): Dataset, DS, EagleV2, EAGLE v2 training — standalone script, no datasets library.

### Community 15 - "Community 15"
Cohesion: 0.2
Nodes (7): get_config(), Return a LangChain RunnableConfig dict with active callbacks., chat_node(), ChatState, Minimal single-node LangGraph chat agent.  Usage:     python agents/simple.py "y, SwarmState, TypedDict

### Community 16 - "Community 16"
Cohesion: 0.4
Nodes (3): Convert PyTorch EAGLE head to binary format for llama.cpp., write_tensor(), main()

### Community 17 - "Community 17"
Cohesion: 1.0
Nodes (1): SGLang launcher with qwen35 GGUF patch. Applies the monkey-patch then starts SGL

### Community 18 - "Community 18"
Cohesion: 1.0
Nodes (1): Monkey-patch transformers to support qwen35 GGUF architecture.  Qwen3.5/3.6 uses

### Community 20 - "Community 20"
Cohesion: 1.0
Nodes (1): Runs inside the container — quantizes the heretic model to W4A16 compressed-tens

### Community 21 - "Community 21"
Cohesion: 1.0
Nodes (1): Generate EAGLE training data from the model's own code completions.  No external

### Community 22 - "Community 22"
Cohesion: 1.0
Nodes (1): Predict next n_draft tokens from a single hidden state vector.

### Community 23 - "Community 23"
Cohesion: 1.0
Nodes (1): Generate tokens with speculative decoding.

### Community 24 - "Community 24"
Cohesion: 1.0
Nodes (1): Generate n_draft tokens autoregressively.

### Community 25 - "Community 25"
Cohesion: 1.0
Nodes (1): Probe all endpoints, return {endpoint_name: [model_ids]}.

### Community 26 - "Community 26"
Cohesion: 1.0
Nodes (1): Return flat list of Bifrost-routable model names.

### Community 27 - "Community 27"
Cohesion: 1.0
Nodes (1): Start the LLM stack if nothing is running.

### Community 28 - "Community 28"
Cohesion: 1.0
Nodes (1): Stream from backend, buffer silently. Returns (reply, think).

### Community 29 - "Community 29"
Cohesion: 1.0
Nodes (1): List available models.

### Community 30 - "Community 30"
Cohesion: 1.0
Nodes (1): Generate tokens with speculative decoding.

### Community 31 - "Community 31"
Cohesion: 1.0
Nodes (1): Strips think blocks from ALL chat completion responses (streaming + non-streamin

### Community 32 - "Community 32"
Cohesion: 1.0
Nodes (1): Strips think blocks from ALL chat completion responses (streaming + non-streamin

## Knowledge Gaps
- **86 isolated node(s):** `OpenAI-compatible HTTP server on top of test_dflash.      pip install fastapi uv`, `Infer the HuggingFace tokenizer repo from a GGUF target file.      The GGUF file`, `Extract <tool_call>...</tool_call> blocks from generated text.     Handles two f`, `SGLang launcher with qwen35 GGUF patch. Applies the monkey-patch then starts SGL`, `Monkey-patch transformers to support qwen35 GGUF architecture.  Qwen3.5/3.6 uses` (+81 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 17`** (2 nodes): `sglang_launcher.py`, `SGLang launcher with qwen35 GGUF patch. Applies the monkey-patch then starts SGL`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 18`** (2 nodes): `patch_gguf_qwen35.py`, `Monkey-patch transformers to support qwen35 GGUF architecture.  Qwen3.5/3.6 uses`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 20`** (2 nodes): `quantize_inner.py`, `Runs inside the container — quantizes the heretic model to W4A16 compressed-tens`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 21`** (2 nodes): `gen_eagle_data.py`, `Generate EAGLE training data from the model's own code completions.  No external`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 22`** (1 nodes): `Predict next n_draft tokens from a single hidden state vector.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 23`** (1 nodes): `Generate tokens with speculative decoding.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 24`** (1 nodes): `Generate n_draft tokens autoregressively.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 25`** (1 nodes): `Probe all endpoints, return {endpoint_name: [model_ids]}.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 26`** (1 nodes): `Return flat list of Bifrost-routable model names.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 27`** (1 nodes): `Start the LLM stack if nothing is running.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 28`** (1 nodes): `Stream from backend, buffer silently. Returns (reply, think).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 29`** (1 nodes): `List available models.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 30`** (1 nodes): `Generate tokens with speculative decoding.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (1 nodes): `Strips think blocks from ALL chat completion responses (streaming + non-streamin`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (1 nodes): `Strips think blocks from ALL chat completion responses (streaming + non-streamin`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `invoke_swarm()` connect `Community 1` to `Community 12`?**
  _High betweenness centrality (0.011) - this node is a cross-community bridge._
- **Why does `chat_completions()` connect `Community 9` to `Community 4`, `Community 5`?**
  _High betweenness centrality (0.011) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `SQLiteDataLayer` (e.g. with `Chainlit frontend — local-llm stack. Agentic: tool-calling loop with think displ` and `Execute Python code and return stdout/result. Use for calculations, data process`) actually correct?**
  _`SQLiteDataLayer` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `WikiRAG` (e.g. with `Probe all known endpoints and return available models.` and `Send a chat completion to Bifrost (routes to the right backend).`) actually correct?**
  _`WikiRAG` has 5 INFERRED edges - model-reasoned connections that need verification._
- **What connects `OpenAI-compatible HTTP server on top of test_dflash.      pip install fastapi uv`, `Infer the HuggingFace tokenizer repo from a GGUF target file.      The GGUF file`, `Extract <tool_call>...</tool_call> blocks from generated text.     Handles two f` to the rest of the system?**
  _86 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.08 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.09 - nodes in this community are weakly interconnected._