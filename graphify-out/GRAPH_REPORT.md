# Graph Report - local-llm  (2026-05-05)

## Corpus Check
- 46 files · ~31,924 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 530 nodes · 1001 edges · 35 communities detected
- Extraction: 61% EXTRACTED · 39% INFERRED · 0% AMBIGUOUS · INFERRED: 386 edges (avg confidence: 0.53)
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
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]

## God Nodes (most connected - your core abstractions)
1. `ModelEntry` - 71 edges
2. `VramScheduler` - 55 edges
3. `Capability` - 42 edges
4. `Router` - 36 edges
5. `VramBudget` - 34 edges
6. `LoadedModel` - 28 edges
7. `SQLiteDataLayer` - 26 edges
8. `RouteDecision` - 26 edges
9. `LamuMcpServer` - 25 edges
10. `ReasoningMarker` - 20 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `create_app()`  [INFERRED]
  server/serve.py → lamu/api/openai_compat.py
- `ModelEntry` --uses--> `Backend protocol — interface that all model backends implement.`  [INFERRED]
  lamu/core/types.py → lamu/backends/base.py
- `ModelEntry` --uses--> `Abstract base for model backends.      Each backend manages one model process (o`  [INFERRED]
  lamu/core/types.py → lamu/backends/base.py
- `chat_completions()` --calls--> `generate()`  [INFERRED]
  server/gpt2_proxy.py → server/megakernel_server.py
- `WikiRAG` --uses--> `Probe all known endpoints and return available models.`  [INFERRED]
  server/rag.py → server/mcp_qwen.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.07
Nodes (70): ChatRequest, create_app(), Message, OpenAI-compatible HTTP API layer.  Translates /v1/chat/completions → internal ro, Start the OpenAI-compat server., Create the OpenAI-compatible FastAPI app., serve(), Load model onto GPU. Returns PID of the model process.          Args: (+62 more)

### Community 1 - "Community 1"
Cohesion: 0.11
Nodes (34): get_extractor(), NullReasoningExtractor, Reasoning extractor — per-model-family think-block detection and stripping., Handles think-block detection, stripping, and structured extraction.      Regist, For models that don't use think-blocks. Passes through everything., Factory: returns appropriate extractor based on model's marker config., Split full response into (reasoning, content).          Returns ("", text) if no, Strip reasoning, return only content. (+26 more)

### Community 2 - "Community 2"
Cohesion: 0.06
Nodes (40): compare(), load_swebench_tasks(), main(), Agentic benchmark runner — compare Opus solo vs swarm on coding tasks.  Three mo, Load SWE-bench Lite tasks. Requires swebench package., Run a task with just Opus (cloud-only, no local workers)., Run a task with the full swarm (Opus planner + local workers)., Run a benchmark suite with a given config. (+32 more)

### Community 3 - "Community 3"
Cohesion: 0.08
Nodes (20): BaseDataLayer, build_llm(), extract_think(), on_chat_resume(), on_message(), python_repl(), Chainlit frontend — local-llm stack. Agentic: tool-calling loop with think displ, Agentic loop for dflash models:       - Shows thinking as a collapsible cl.Step (+12 more)

### Community 4 - "Community 4"
Cohesion: 0.07
Nodes (10): ABC, Backend, Backend, Backend protocol — interface that all model backends implement., Abstract base for model backends.      Each backend manages one model process (o, LlamaCppBackend, llama.cpp backend — manages llama-server subprocess., Backend for llama-server (llama.cpp HTTP server). (+2 more)

### Community 5 - "Community 5"
Cohesion: 0.09
Nodes (11): Dataset, DS, EagleV2, EAGLE v2 training — standalone script, no datasets library., draft_multi(), EagleV3, HiddenStateDataset, main() (+3 more)

### Community 6 - "Community 6"
Cohesion: 0.12
Nodes (21): BaseModel, AnthropicMessage, AnthropicMessagesRequest, build_app(), ChatMessage, ChatRequest, main(), _parse_tool_calls() (+13 more)

### Community 7 - "Community 7"
Cohesion: 0.13
Nodes (11): write_registry(), _query_gpu_pids(), cmd_scan(), cmd_start(), cmd_status(), main(), LAMU Daemon — main entry point.  Usage:   python -m lamu start       # start dae, Scan ~/models/ and write registry. (+3 more)

### Community 8 - "Community 8"
Cohesion: 0.12
Nodes (14): chat(), get_default(), LocalLLM, models(), Local LLM Python client — import from anywhere.      from server.client import L, Send a multi-turn conversation. Returns the response text., Stream tokens from a chat completion., Check if the LLM backend is reachable. (+6 more)

### Community 9 - "Community 9"
Cohesion: 0.15
Nodes (12): chat(), ChatMessage, ChatRequest, draft(), EagleHead, get_hidden_state(), LeanDecoder, main() (+4 more)

### Community 10 - "Community 10"
Cohesion: 0.13
Nodes (11): call_tool(), _chat(), _discover_models(), MCP server — exposes all local LLMs as tools for Claude Code (or any MCP client), Probe all known endpoints and return available models., Send a chat completion to Bifrost (routes to the right backend)., Simple RAG over the LAMU wiki for the local model.  Loads wiki pages, finds rele, Keyword-based retrieval over the LAMU wiki. (+3 more)

### Community 11 - "Community 11"
Cohesion: 0.17
Nodes (10): chat_completions(), ChatMessage, ChatRequest, EagleHead, generate(), main(), predict_next(), Custom speculative decoding server using the trained EAGLE head.  Loads both the (+2 more)

### Community 12 - "Community 12"
Cohesion: 0.16
Nodes (14): apply_chat_template(), aread_tokens(), chat_completions(), ChatRequest, get_tokenizer(), DFlash server for 24 GB GPUs (RTX 4090) with VRAM park/unpark dance.  The stock, Send command to daemon stdin., Read generated tokens from pipe. (+6 more)

### Community 13 - "Community 13"
Cohesion: 0.17
Nodes (12): create_poincare_plot(), load_graphify_json(), load_networkx_from_codebase(), main(), PoincareBallEmbedding, Poincaré ball embedding for knowledge graphs.  Takes graphify's graph.json and e, Load graphify's graph.json into a NetworkX graph., Build a graph from Python AST. Filters stdlib/third-party noise by default. (+4 more)

### Community 14 - "Community 14"
Cohesion: 0.25
Nodes (14): auto_start(), cmd_models(), cmd_status(), discover_models(), get_available_models(), main(), probe_endpoint(), Start the LLM stack if nothing is running. (+6 more)

### Community 15 - "Community 15"
Cohesion: 0.21
Nodes (12): collect(), export(), main(), prepare(), Training pipeline — collect swarm data + QLoRA fine-tune local models.  Feedback, Run QLoRA fine-tuning with unsloth (optimized for consumer GPUs)., Merge LoRA + export to GGUF (for DFlash) or HF format (for vLLM)., Save a successful (task → implementation) pair for fine-tuning. (+4 more)

### Community 16 - "Community 16"
Cohesion: 0.23
Nodes (7): ctx_for_quant(), find_gguf(), kv_type_for_quant(), main(), Production Qwen3.6 server — think-block stripping via ASGI middleware., Wraps the llama-cpp-python ASGI app and strips think blocks from responses., ThinkStripASGI

### Community 17 - "Community 17"
Cohesion: 0.2
Nodes (7): get_config(), Return a LangChain RunnableConfig dict with active callbacks., chat_node(), ChatState, Minimal single-node LangGraph chat agent.  Usage:     python agents/simple.py "y, SwarmState, TypedDict

### Community 18 - "Community 18"
Cohesion: 0.4
Nodes (3): Convert PyTorch EAGLE head to binary format for llama.cpp., write_tensor(), main()

### Community 19 - "Community 19"
Cohesion: 1.0
Nodes (1): SGLang launcher with qwen35 GGUF patch. Applies the monkey-patch then starts SGL

### Community 20 - "Community 20"
Cohesion: 1.0
Nodes (1): Monkey-patch transformers to support qwen35 GGUF architecture.  Qwen3.5/3.6 uses

### Community 22 - "Community 22"
Cohesion: 1.0
Nodes (1): Runs inside the container — quantizes the heretic model to W4A16 compressed-tens

### Community 23 - "Community 23"
Cohesion: 1.0
Nodes (1): Generate EAGLE training data from the model's own code completions.  No external

### Community 24 - "Community 24"
Cohesion: 1.0
Nodes (1): Allow `python -m lamu` invocation.

### Community 25 - "Community 25"
Cohesion: 1.0
Nodes (1): Configuration constants and paths.

### Community 26 - "Community 26"
Cohesion: 1.0
Nodes (1): Predict next n_draft tokens from a single hidden state vector.

### Community 27 - "Community 27"
Cohesion: 1.0
Nodes (1): Generate tokens with speculative decoding.

### Community 28 - "Community 28"
Cohesion: 1.0
Nodes (1): Generate n_draft tokens autoregressively.

### Community 35 - "Community 35"
Cohesion: 1.0
Nodes (1): Snapshot of VRAM allocation.

### Community 36 - "Community 36"
Cohesion: 1.0
Nodes (1): List available models.

### Community 37 - "Community 37"
Cohesion: 1.0
Nodes (1): Probe all endpoints, return {endpoint_name: [model_ids]}.

### Community 38 - "Community 38"
Cohesion: 1.0
Nodes (1): Return flat list of Bifrost-routable model names.

### Community 39 - "Community 39"
Cohesion: 1.0
Nodes (1): Start the LLM stack if nothing is running.

### Community 40 - "Community 40"
Cohesion: 1.0
Nodes (1): Stream from backend, buffer silently. Returns (reply, think).

### Community 41 - "Community 41"
Cohesion: 1.0
Nodes (1): List available models.

## Knowledge Gaps
- **110 isolated node(s):** `OpenAI-compatible HTTP server on top of test_dflash.      pip install fastapi uv`, `Infer the HuggingFace tokenizer repo from a GGUF target file.      The GGUF file`, `Extract <tool_call>...</tool_call> blocks from generated text.     Handles two f`, `SGLang launcher with qwen35 GGUF patch. Applies the monkey-patch then starts SGL`, `Monkey-patch transformers to support qwen35 GGUF architecture.  Qwen3.5/3.6 uses` (+105 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 19`** (2 nodes): `sglang_launcher.py`, `SGLang launcher with qwen35 GGUF patch. Applies the monkey-patch then starts SGL`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 20`** (2 nodes): `patch_gguf_qwen35.py`, `Monkey-patch transformers to support qwen35 GGUF architecture.  Qwen3.5/3.6 uses`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 22`** (2 nodes): `quantize_inner.py`, `Runs inside the container — quantizes the heretic model to W4A16 compressed-tens`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 23`** (2 nodes): `gen_eagle_data.py`, `Generate EAGLE training data from the model's own code completions.  No external`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 24`** (2 nodes): `__main__.py`, `Allow `python -m lamu` invocation.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 25`** (2 nodes): `Configuration constants and paths.`, `config.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 26`** (1 nodes): `Predict next n_draft tokens from a single hidden state vector.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 27`** (1 nodes): `Generate tokens with speculative decoding.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 28`** (1 nodes): `Generate n_draft tokens autoregressively.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (1 nodes): `Snapshot of VRAM allocation.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 36`** (1 nodes): `List available models.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 37`** (1 nodes): `Probe all endpoints, return {endpoint_name: [model_ids]}.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 38`** (1 nodes): `Return flat list of Bifrost-routable model names.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 39`** (1 nodes): `Start the LLM stack if nothing is running.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 40`** (1 nodes): `Stream from backend, buffer silently. Returns (reply, think).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 41`** (1 nodes): `List available models.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ModelEntry` connect `Community 0` to `Community 1`, `Community 4`, `Community 7`?**
  _High betweenness centrality (0.120) - this node is a cross-community bridge._
- **Why does `Message` connect `Community 0` to `Community 6`?**
  _High betweenness centrality (0.056) - this node is a cross-community bridge._
- **Why does `ChatRequest` connect `Community 0` to `Community 6`?**
  _High betweenness centrality (0.056) - this node is a cross-community bridge._
- **Are the 69 inferred relationships involving `ModelEntry` (e.g. with `Model registry — auto-discovers models on disk, writes/reads YAML config.` and `Read key GGUF metadata without loading full model.`) actually correct?**
  _`ModelEntry` has 69 INFERRED edges - model-reasoned connections that need verification._
- **Are the 39 inferred relationships involving `VramScheduler` (e.g. with `LAMU Daemon — main entry point.  Usage:   python -m lamu start       # start dae` and `Scan ~/models/ and write registry.`) actually correct?**
  _`VramScheduler` has 39 INFERRED edges - model-reasoned connections that need verification._
- **Are the 39 inferred relationships involving `Capability` (e.g. with `Model registry — auto-discovers models on disk, writes/reads YAML config.` and `Read key GGUF metadata without loading full model.`) actually correct?**
  _`Capability` has 39 INFERRED edges - model-reasoned connections that need verification._
- **Are the 27 inferred relationships involving `Router` (e.g. with `VramScheduler` and `Capability`) actually correct?**
  _`Router` has 27 INFERRED edges - model-reasoned connections that need verification._