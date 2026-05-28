//! Single source of truth for the MCP tool catalog.
//!
//! Phase 2.1 design: every tool has one entry in `TOOLS` carrying its
//! name, description, JSON schema, and a dispatch function. The
//! dispatcher in `server::tools_call` looks the entry up by name and
//! invokes the handler; `server::tools_list_response` iterates the
//! same table to build the catalog. The "every tool listed" test
//! guard is now a no-op by construction.
//!
//! Phase 2.2 (deferred): split per-tool dispatch wrappers into one
//! file each under `src/tools/<name>.rs`.

use crate::server::LamuMcpServer;
use serde_json::{json, Value};
use std::future::Future;
use std::pin::Pin;

/// Dispatch handler kind. Distinguishes stateful (needs `&LamuMcpServer`)
/// from free (no state). Sync handlers wrap their result in a ready
/// future so the dispatcher only has two arms instead of four.
///
/// Sealed by design — no `#[non_exhaustive]`. Adding a third variant
/// here forces the dispatcher in `server::tools_call` to recompile
/// with a missing arm, which is the correctness property we want.
pub enum HandlerKind {
    /// Async handler taking `&LamuMcpServer` (or sync-wrapped-as-async).
    Stateful(for<'a> fn(&'a LamuMcpServer, Value) -> Pin<Box<dyn Future<Output = String> + Send + 'a>>),
    /// Async handler with no state (or sync-wrapped-as-async).
    Free(fn(Value) -> Pin<Box<dyn Future<Output = String> + Send>>),
}

pub struct ToolDef {
    pub name: &'static str,
    pub description: &'static str,
    /// Schema is computed lazily so the const table holds plain fn ptrs.
    pub schema_fn: fn() -> Value,
    pub handler: HandlerKind,
    /// True for cloud-LLM tools that routing mode 'local-only' refuses.
    /// The field is required, so adding a tool forces a cloud/local
    /// decision at the call site — a new cloud tool can't silently
    /// bypass the local-only gate (the prior hand-curated match could
    /// drift). NOTE: `parallel_query` is `false` despite reaching cloud:
    /// it mixes local + cloud tasks and self-enforces local-only
    /// per-task in handle_parallel_query, so a blanket dispatcher
    /// refusal would wrongly kill its local tasks too.
    pub cloud: bool,
}

impl ToolDef {
    /// Build the JSON object MCP `tools/list` wants for this tool.
    pub fn to_list_entry(&self) -> Value {
        json!({
            "name": self.name,
            "description": self.description,
            "inputSchema": (self.schema_fn)(),
        })
    }
}

// ── Schema constructors ─────────────────────────────────────────────
// One per tool. Each returns the inputSchema JSON object. Kept as
// functions (not consts) because serde_json::Value isn't const-evaluable.

fn schema_query() -> Value {
    json!({
        "type": "object",
        "properties": {
            "prompt": {"type": "string"},
            "model": {"type": "string"},
            "capabilities": {"type": "array", "items": {"type": "string"}},
            "system": {"type": "string", "default": ""},
            "max_tokens": {"type": "integer", "default": 16384},
            "temperature": {"type": "number", "default": 0.7},
            "include_reasoning": {"type": "boolean", "default": false},
            "enable_thinking": {"type": "boolean", "description": "Toggle Qwen3.6/3.5 <think> reasoning. False = direct answer (4× faster wall on simple queries, ~1.2× on long). Routed via Backend::generate_with_opts → bee chat_template_kwargs.enable_thinking. Default: model's choice (thinking on)."},
            "priority": {"type": "integer", "default": 0, "description": "Higher served first (priority strategy only)"},
            "origin": {"type": "string", "default": "anonymous", "description": "Agent identifier for queue observability"},
        },
        "required": ["prompt"]
    })
}

fn schema_plan_query() -> Value {
    json!({
        "type": "object",
        "properties": {
            "prompt": {"type": "string"},
            "model": {"type": "string"},
            "capabilities": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["prompt"]
    })
}

fn schema_empty_object() -> Value {
    json!({"type": "object", "properties": {}})
}

fn schema_named_only() -> Value {
    json!({
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    })
}

fn schema_cloud_query() -> Value {
    json!({
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "User prompt"},
            "model": {"type": "string", "description": "Cloud model name from cloud-models.yaml (e.g. 'mimo-v2.5', 'mimo-v2.5-pro', 'deepseek-v4-flash', 'claude-haiku-4-5'). Defaults to 'mimo-v2.5'.", "default": "mimo-v2.5"},
            "system": {"type": "string", "description": "System prompt", "default": ""},
            "max_tokens": {"type": "integer", "default": 8192},
            "temperature": {"type": "number", "default": 0.3},
            "include_reasoning": {"type": "boolean", "default": false, "description": "When true, include the model's <think> reasoning_content in the output. Default false (just the answer)."},
            "thinking_enabled": {"type": "boolean", "description": "Engage the model's extended thinking pass. Default: ON for Pro/reasoner/opus model names, OFF for Flash and similar. OFF saves 50-80% wall time on simple tasks. Set explicitly when defaults don't fit."},
            "plan_file": {"type": "string", "description": "Optional path to a plan/spec markdown file to inject as Plan-tier context before the system prompt. Overrides $LAMU_PLAN env and any auto-detected ~/.claude/plans/active.md."},
            "context": {"type": "string", "description": "Optional verbatim Tactical-tier context (file fragments, recent commits, etc.) injected before the system prompt and after Plan tier. Truncated at 200 KiB."},
            "conversation_id": {"type": "string", "description": "Optional conversation identifier. When set, the last 20 turns under this id are prepended to the Tactical tier as prior context, and this turn (user prompt + assistant reply) is appended to the conversation log at ~/.local/share/lamu/conversations.db. Allowed chars: [A-Za-z0-9_-.]"}
        },
        "required": ["prompt"]
    })
}

fn schema_review_commit() -> Value {
    json!({
        "type": "object",
        "properties": {
            "commit": {"type": "string", "description": "Commit SHA or ref (e.g. 'HEAD', 'HEAD~1', 'abc123'). Defaults to HEAD.", "default": "HEAD"},
            "repo": {"type": "string", "description": "Path to the git repo. Defaults to current working directory.", "default": "."},
            "focus": {"type": "string", "description": "Optional review focus (e.g. 'security', 'performance', 'API design'). Defaults to all-around.", "default": ""},
            "plan_file": {"type": "string", "description": "Optional path to a plan/spec markdown file to inject as Plan-tier context. Overrides $LAMU_PLAN and any auto-detected ~/.claude/plans/active.md."},
            "context": {"type": "string", "description": "Optional verbatim Tactical-tier context (file fragments, related commits, etc.) injected before the reviewer system prompt. Truncated at 200 KiB."},
            "auto_context": {"type": "boolean", "default": false, "description": "When true, lamu-mcp builds Tactical-tier context automatically: changed files at HEAD, tree-sitter-extracted added/modified Rust symbols, and ripgrep caller locations across the repo. Concatenated with any caller-supplied `context` arg. Bounded at 200 KiB total. Cost: ~1 git show + 1 ripgrep per added symbol."},
            "preset": {"type": "string", "enum": ["fast", "max"], "default": "max", "description": "Quality/cost preset. 'max' (default, ~$0.005/review hot cache): single Pro pass + critic-role parallel pass (LAMU_CRITIC_PASS) + multi-model ensemble (LAMU_ENSEMBLE_REVIEW — uses claude-opus-4-7 if ANTHROPIC_API_KEY is set, else falls back to v4-flash). 5-7 findings with cross-provider diversity. Use for every-commit reviews and pre-merge gating. 'fast' (~$0.004/review hot cache): single Pro pass with cache discipline only. 0-2 findings. Use only when cost is dominant and false negatives are tolerable. Per-call arg overrides $LAMU_PRESET env."}
        }
    })
}

fn schema_review_diff() -> Value {
    json!({
        "type": "object",
        "properties": {
            "diff": {"type": "string", "description": "Unified diff text or a code chunk to review."},
            "context": {"type": "string", "description": "Optional Tactical-tier context (file paths, related commits, what changed and why). Truncated at 200 KiB.", "default": ""},
            "focus": {"type": "string", "default": ""},
            "plan_file": {"type": "string", "description": "Optional path to a plan/spec markdown file to inject as Plan-tier context. Overrides $LAMU_PLAN and any auto-detected ~/.claude/plans/active.md."},
            "preset": {"type": "string", "enum": ["fast", "max"], "default": "max", "description": "Quality/cost preset. Same semantics as review_commit: 'max' (default) = Pro + critic + ensemble; 'fast' = single Pro pass. Per-call arg overrides $LAMU_PRESET env."}
        },
        "required": ["diff"]
    })
}

fn schema_set_routing_mode() -> Value {
    json!({
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": ["auto", "local-only", "cloud-only"]}
        },
        "required": ["mode"]
    })
}

fn schema_warmup() -> Value {
    json!({
        "type": "object",
        "properties": {
            "plan_file": {"type": "string", "description": "Optional plan file to include in the cached prefix. Match the same `plan_file` you'll pass to review_commit so the prefix bytes line up."},
            "repo": {"type": "string", "default": ".", "description": "Repo path for plan auto-detect. Defaults to cwd."}
        }
    })
}

fn schema_search_repo() -> Value {
    json!({
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search term or regex (ripgrep) / natural-language phrase (semantic)."},
            "mode": {"type": "string", "enum": ["auto", "ripgrep", "semantic"], "default": "auto", "description": "auto = ripgrep first, semantic fallback (requires OPENAI_API_KEY). ripgrep = grep only. semantic = embeddings only."},
            "k": {"type": "integer", "default": 8, "description": "Max hits to return."},
            "repo": {"type": "string", "default": ".", "description": "Path to git repo. Defaults to cwd."}
        },
        "required": ["query"]
    })
}

fn schema_index_repo() -> Value {
    json!({
        "type": "object",
        "properties": {
            "repo": {"type": "string", "default": ".", "description": "Path to git repo to index."},
            "force": {"type": "boolean", "default": false, "description": "Re-embed all files even if mtime is unchanged."}
        }
    })
}

fn schema_train_from_conversations() -> Value {
    json!({
        "type": "object",
        "properties": {
            "output_name": {
                "type": "string",
                "description": "Registry name for the trained model. Must match [A-Za-z0-9_.-]+ with no leading '.' or '-' and no '..'."
            },
            "since": {
                "type": "string",
                "default": "30d",
                "description": "How far back to pull conversations. Humantime duration: '7d', '30d', '12h', etc."
            },
            "base_model": {
                "type": "string",
                "default": "Qwen/Qwen3-7B",
                "description": "HuggingFace base model id (org/name)."
            },
            "method": {
                "type": "string",
                "enum": ["qlora", "lora", "full"],
                "default": "qlora",
                "description": "Fine-tuning method. qlora is the 4090-friendly default."
            },
            "confirm": {
                "type": "boolean",
                "default": false,
                "description": "Must be true to actually start. First call without confirm returns the dataset estimate so the caller can decide."
            }
        },
        "required": ["output_name"]
    })
}

fn schema_recall_conversation() -> Value {
    json!({
        "type": "object",
        "properties": {
            "conversation_id": {"type": "string", "description": "Identifier for the conversation. Allowed chars: [A-Za-z0-9_-.]"},
            "limit": {"type": "integer", "default": 0, "description": "Max number of most-recent turns to return (oldest-first). 0 = no cap (full transcript)."}
        },
        "required": ["conversation_id"]
    })
}

fn schema_write_file() -> Value {
    json!({
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path under cwd. Relative components only — '..' is refused."},
            "content": {"type": "string", "description": "UTF-8 file contents."},
            "session_id": {"type": "string", "description": "Session identifier for rollback. Allowed chars: [A-Za-z0-9_-.]"}
        },
        "required": ["path", "content", "session_id"]
    })
}

fn schema_parallel_query() -> Value {
    json!({
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "description": "Array of task objects. Each can override model/system/max_tokens/temperature/id; missing fields fall back to top-level defaults.",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Optional caller-supplied id for matching results back."},
                        "prompt": {"type": "string"},
                        "model": {"type": "string"},
                        "system": {"type": "string"},
                        "max_tokens": {"type": "integer"},
                        "temperature": {"type": "number"},
                        "include_reasoning": {"type": "boolean"}
                    },
                    "required": ["prompt"]
                }
            },
            "default_model": {"type": "string", "default": "mimo-v2.5"},
            "default_system": {"type": "string", "default": ""},
            "max_concurrency": {"type": "integer", "description": "Optional cap that overrides per-provider defaults (downwards only — never raises an unproven provider above 1)."}
        },
        "required": ["tasks"]
    })
}

// ── Dispatch wrappers ───────────────────────────────────────────────
// One per tool. Async handlers Box::pin their future; sync handlers
// run synchronously and wrap the result in a ready future.

fn dispatch_query<'a>(s: &'a LamuMcpServer, args: Value) -> Pin<Box<dyn Future<Output = String> + Send + 'a>> {
    Box::pin(s.handle_query(args))
}

fn dispatch_plan_query<'a>(s: &'a LamuMcpServer, args: Value) -> Pin<Box<dyn Future<Output = String> + Send + 'a>> {
    let r = s.handle_plan_query(args);
    Box::pin(async move { r })
}

fn dispatch_list_models<'a>(s: &'a LamuMcpServer, _args: Value) -> Pin<Box<dyn Future<Output = String> + Send + 'a>> {
    let r = s.handle_list_models();
    Box::pin(async move { r })
}

fn dispatch_load_model<'a>(s: &'a LamuMcpServer, args: Value) -> Pin<Box<dyn Future<Output = String> + Send + 'a>> {
    Box::pin(s.handle_load_model(args))
}

fn dispatch_unload_model<'a>(s: &'a LamuMcpServer, args: Value) -> Pin<Box<dyn Future<Output = String> + Send + 'a>> {
    Box::pin(s.handle_unload_model(args))
}

fn dispatch_vram_status<'a>(s: &'a LamuMcpServer, _args: Value) -> Pin<Box<dyn Future<Output = String> + Send + 'a>> {
    let r = s.handle_vram_status();
    Box::pin(async move { r })
}

fn dispatch_scan<'a>(s: &'a LamuMcpServer, _args: Value) -> Pin<Box<dyn Future<Output = String> + Send + 'a>> {
    let r = s.handle_scan();
    Box::pin(async move { r })
}

fn dispatch_queue_status<'a>(s: &'a LamuMcpServer, _args: Value) -> Pin<Box<dyn Future<Output = String> + Send + 'a>> {
    Box::pin(s.handle_queue_status())
}

fn dispatch_set_routing_mode<'a>(s: &'a LamuMcpServer, args: Value) -> Pin<Box<dyn Future<Output = String> + Send + 'a>> {
    Box::pin(s.handle_set_routing_mode(args))
}

fn dispatch_routing_status<'a>(s: &'a LamuMcpServer, _args: Value) -> Pin<Box<dyn Future<Output = String> + Send + 'a>> {
    Box::pin(s.handle_routing_status())
}

fn dispatch_parallel_query<'a>(s: &'a LamuMcpServer, args: Value) -> Pin<Box<dyn Future<Output = String> + Send + 'a>> {
    Box::pin(s.handle_parallel_query(args))
}

fn dispatch_cloud_query(args: Value) -> Pin<Box<dyn Future<Output = String> + Send>> {
    Box::pin(crate::cloud::handle_cloud_query(args))
}

fn dispatch_list_cloud_models(_args: Value) -> Pin<Box<dyn Future<Output = String> + Send>> {
    let r = crate::cloud::handle_list_cloud_models();
    Box::pin(async move { r })
}

fn dispatch_review_commit(args: Value) -> Pin<Box<dyn Future<Output = String> + Send>> {
    Box::pin(crate::cloud::handle_review_commit(args))
}

fn dispatch_review_diff(args: Value) -> Pin<Box<dyn Future<Output = String> + Send>> {
    Box::pin(crate::cloud::handle_review_diff(args))
}

fn dispatch_warmup(args: Value) -> Pin<Box<dyn Future<Output = String> + Send>> {
    Box::pin(crate::cloud::handle_warmup(args))
}

fn dispatch_search_repo(args: Value) -> Pin<Box<dyn Future<Output = String> + Send>> {
    Box::pin(crate::cloud::handle_search_repo(args))
}

fn dispatch_index_repo(args: Value) -> Pin<Box<dyn Future<Output = String> + Send>> {
    Box::pin(crate::cloud::handle_index_repo(args))
}

fn dispatch_recall_conversation(args: Value) -> Pin<Box<dyn Future<Output = String> + Send>> {
    let r = crate::cloud::handle_recall_conversation(args);
    Box::pin(async move { r })
}

fn dispatch_train_from_conversations(
    args: Value,
) -> Pin<Box<dyn Future<Output = String> + Send>> {
    Box::pin(crate::train_tool::handle_train_from_conversations(args))
}

fn dispatch_write_file(args: Value) -> Pin<Box<dyn Future<Output = String> + Send>> {
    Box::pin(crate::server::handle_write_file(args))
}

// ── The catalog ─────────────────────────────────────────────────────
// New tool? One entry here. tools_list_response + dispatcher both
// pick it up automatically.

pub static TOOLS: &[ToolDef] = &[
    ToolDef {
        name: "query",
        description: "Send prompt to local LLM. Routes by capabilities or explicit model. Queued per-model (FIFO default) so concurrent agents don't collide. Fast, free, uncensored.",
        schema_fn: schema_query,
        handler: HandlerKind::Stateful(dispatch_query),
        cloud: false,
    },
    ToolDef {
        name: "plan_query",
        description: "Dry-run: see which model WOULD handle a request without generating.",
        schema_fn: schema_plan_query,
        handler: HandlerKind::Stateful(dispatch_plan_query),
        cloud: false,
    },
    ToolDef {
        name: "list_models",
        description: "List all known models with load status and capabilities.",
        schema_fn: schema_empty_object,
        handler: HandlerKind::Stateful(dispatch_list_models),
        cloud: false,
    },
    ToolDef {
        name: "load_model",
        description: "Explicitly load a model onto GPU.",
        schema_fn: schema_named_only,
        handler: HandlerKind::Stateful(dispatch_load_model),
        cloud: false,
    },
    ToolDef {
        name: "unload_model",
        description: "Unload a model from GPU.",
        schema_fn: schema_named_only,
        handler: HandlerKind::Stateful(dispatch_unload_model),
        cloud: false,
    },
    ToolDef {
        name: "vram_status",
        description: "Show current VRAM allocation.",
        schema_fn: schema_empty_object,
        handler: HandlerKind::Stateful(dispatch_vram_status),
        cloud: false,
    },
    ToolDef {
        name: "scan_models",
        description: "Re-scan disk for new models.",
        schema_fn: schema_empty_object,
        handler: HandlerKind::Stateful(dispatch_scan),
        cloud: false,
    },
    ToolDef {
        name: "queue_status",
        description: "Show per-model queue depth and scheduling strategy.",
        schema_fn: schema_empty_object,
        handler: HandlerKind::Stateful(dispatch_queue_status),
        cloud: false,
    },
    ToolDef {
        name: "cloud_query",
        description: "Send prompt to a cloud model (DeepSeek V4, Claude, GLM, Kimi, Qwen-Max, etc.). Use this for tasks that need stronger reasoning than local, OR cheaper inference than the calling agent (e.g. Claude Code → DeepSeek V4 Flash for code generation at ~$0.07/M input, currently 75% off). Auto-routes via OpenAI/Anthropic format detection.",
        schema_fn: schema_cloud_query,
        handler: HandlerKind::Free(dispatch_cloud_query),
        cloud: true,
    },
    ToolDef {
        name: "list_cloud_models",
        description: "List configured cloud models from ~/.config/lamu/cloud-models.yaml. Returns name, provider, context window, and whether the API key env var is set.",
        schema_fn: schema_empty_object,
        handler: HandlerKind::Free(dispatch_list_cloud_models),
        cloud: false,
    },
    ToolDef {
        name: "review_commit",
        description: "PRIMARY REVIEW TOOL — auto-routes to MiMo V2.5 Pro (the project policy reviewer). Takes a commit SHA (or 'HEAD' for the most recent), runs `git show` to get the full diff + commit message, and returns a deep code review covering security, correctness, edge cases, idiom, and architectural fit. NO CODE SHOULD BE CONSIDERED DONE WITHOUT GOING THROUGH THIS TOOL. Use it after every commit you make.\n\nQUALITY/COST PRESETS — `preset` arg controls the review intensity:\n  - 'max' (default, ~$0.005/review hot cache): single Pro pass + critic-role parallel pass + multi-model ensemble (uses claude-opus-4-7 if ANTHROPIC_API_KEY is set, else falls back to deepseek-v4-flash). 5-7 findings, cross-provider diversity. Use for every-commit reviews + pre-merge gating.\n  - 'fast' (~$0.004/review hot cache): single Pro pass only. 0-2 findings. Use when cost is dominant and false negatives are tolerable.\nPer-call arg overrides $LAMU_PRESET env. Individual env knobs (LAMU_CRITIC_PASS, LAMU_ENSEMBLE_REVIEW, LAMU_TEST_PREFLIGHT, LAMU_TWO_STAGE_REVIEW) override the preset's defaults.\n\nMANDATORY: Before applying ANY fix from a review, verify each finding is real, not a hallucination. MiMo V2.5 Pro has ~30% false-positive rate. Open the cited file:line, read the code, confirm the bug exists. Common hallucinations: serde_json indexing claimed to panic (returns Null instead), bwrap claimed to expose paths it doesn't bind (empty namespace by default), GGUF type-5/6 claimed 64-bit (actually 32-bit per spec), env-var race across cargo test binaries (env is process-local). Skip findings that don't reproduce. Note skipped false positives in the follow-up commit message.",
        schema_fn: schema_review_commit,
        handler: HandlerKind::Free(dispatch_review_commit),
        cloud: true,
    },
    ToolDef {
        name: "review_diff",
        description: "Review an arbitrary diff via MiMo V2.5 Pro. Same reviewer policy as review_commit but accepts the diff text directly — useful when reviewing uncommitted changes or a chunk of pasted code.\n\nQUALITY/COST PRESETS — `preset: 'fast' | 'max'` arg, default 'max'. Same semantics as review_commit (max = Pro + critic + ensemble, ANTHROPIC_API_KEY enables cross-provider second reviewer; fast = single Pro pass). Per-call arg overrides $LAMU_PRESET env.\n\nMANDATORY: Before applying ANY fix, verify each finding is real (~30% false-positive rate). Open the cited code, confirm the bug exists. Skip findings that don't reproduce. Common hallucinations: serde_json indexing claimed to panic (returns Null in reality), bwrap claimed to expose paths it doesn't bind (empty namespace by default), GGUF type-5/6 claimed 64-bit (32-bit per spec), env-var race across cargo test binaries (env is process-local).",
        schema_fn: schema_review_diff,
        handler: HandlerKind::Free(dispatch_review_diff),
        cloud: true,
    },
    ToolDef {
        name: "set_routing_mode",
        description: "Control which backends are usable. Modes: 'auto' (default — use local for matching capabilities, cloud for the rest), 'local-only' (refuse cloud requests), 'cloud-only' (kill local llama-server and free VRAM, route everything to cloud). Useful when you want to free GPU for other work but keep DeepSeek/Claude on tap.",
        schema_fn: schema_set_routing_mode,
        handler: HandlerKind::Stateful(dispatch_set_routing_mode),
        cloud: false,
    },
    ToolDef {
        name: "routing_status",
        description: "Report current routing mode + which backends are reachable.",
        schema_fn: schema_empty_object,
        handler: HandlerKind::Stateful(dispatch_routing_status),
        cloud: false,
    },
    ToolDef {
        name: "warmup",
        description: "Prime the cloud reviewer's prompt cache by sending a 1-token completion with the future review_commit's central+plan prefix. Subsequent calls in the session hit cache from byte 0 instead of paying full prefix-prefill on the first call. Cost: ~$0.0001 per warmup. Pass the same `plan_file` you'll use later.",
        schema_fn: schema_warmup,
        handler: HandlerKind::Free(dispatch_warmup),
        cloud: true,
    },
    ToolDef {
        name: "search_repo",
        description: "Find code in the repository. Modes: 'auto' (ripgrep first, semantic fallback when OPENAI_API_KEY is set), 'ripgrep' (instant grep), 'semantic' (cosine-sim against the embedding index — `index_repo` builds it). Returns up to k hits with file:line + snippet. Useful for the orchestrator to populate the Tactical-tier `context` arg of cloud_query / review_commit.",
        schema_fn: schema_search_repo,
        handler: HandlerKind::Free(dispatch_search_repo),
        cloud: false,
    },
    ToolDef {
        name: "index_repo",
        description: "Build / refresh the semantic-search index at ~/.local/share/lamu/embeddings.db. Walks `git ls-files`, chunks each text file at ~1KB boundaries, embeds via OpenAI text-embedding-3-small (~$0.02/M tokens). Skips files whose mtime is unchanged from the previous index unless `force: true`.",
        schema_fn: schema_index_repo,
        handler: HandlerKind::Free(dispatch_index_repo),
        cloud: false,
    },
    ToolDef {
        name: "recall_conversation",
        description: "Read recorded turns from a conversation logged via cloud_query's `conversation_id` arg. Returns oldest-first, optionally capped at `limit` most-recent turns. Storage: ~/.local/share/lamu/conversations.db (SQLite). Use this to inspect what was said in a prior session, or to replay a conversation thread into a fresh cloud_query via the `context` arg.",
        schema_fn: schema_recall_conversation,
        handler: HandlerKind::Free(dispatch_recall_conversation),
        cloud: false,
    },
    ToolDef {
        name: "train_from_conversations",
        description: "Fine-tune a local model on the user's recent conversation history. EXPENSIVE: 30 min – 4 h depending on dataset size; locks the GPU exclusively for the run. First call without `confirm: true` returns the dataset estimate (conversation count + turn count) so the caller can decide. With `confirm: true`, shells out to the `blut` binary (BLUT — Brian Lam's Universal Trainer) in detached background mode and returns immediately — check `blut jobs` for the resulting job id. The MCP server does NOT depend on blut; the binary must be installed separately (cargo install --path lamu-rs/blut) or located via $BLUT_BIN. ($LAMU_TRAIN_BIN remains accepted as a back-compat alias.)",
        schema_fn: schema_train_from_conversations,
        handler: HandlerKind::Free(dispatch_train_from_conversations),
        cloud: false,
    },
    ToolDef {
        name: "write_file",
        description: "Write a file with rollback journaling (Phase 6.1). Records the file's pre-state under session_id; `lamu rollback <session>` restores. Path is required relative to lamu-mcp's cwd; absolute paths and '..' segments are refused so the call cannot escape the working directory. session_id must match [A-Za-z0-9_-.]+ — anything else is rejected up front.",
        schema_fn: schema_write_file,
        handler: HandlerKind::Free(dispatch_write_file),
        cloud: false,
    },
    ToolDef {
        name: "parallel_query",
        description: "Fan out N prompts at once (agent swarm). Provider-aware concurrency: DeepSeek/OpenAI/Anthropic run in parallel up to per-provider caps, untested providers and ALL local models default to sequential (concurrency=1) until proven safe. Tasks are grouped by model so each model gets its own semaphore. Returns results in the original task order, with per-task elapsed time. Use this for batch reviews, parallel code generation, multi-perspective brainstorming.",
        schema_fn: schema_parallel_query,
        handler: HandlerKind::Stateful(dispatch_parallel_query),
        cloud: false, // mixes local+cloud; self-enforces local-only per-task
    },
];

pub fn find(name: &str) -> Option<&'static ToolDef> {
    TOOLS.iter().find(|t| t.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_duplicate_tool_names() {
        let mut seen = std::collections::HashSet::new();
        for t in TOOLS {
            assert!(seen.insert(t.name), "duplicate tool name: {}", t.name);
        }
    }

    #[test]
    fn every_tool_has_nonempty_description() {
        for t in TOOLS {
            assert!(!t.description.is_empty(), "{} has empty description", t.name);
        }
    }

    #[test]
    fn every_tool_schema_is_object() {
        for t in TOOLS {
            let s = (t.schema_fn)();
            assert_eq!(s["type"], "object", "{} schema missing type=object", t.name);
        }
    }

    #[test]
    fn find_resolves_known_tools() {
        assert!(find("query").is_some());
        assert!(find("write_file").is_some());
        assert!(find("nonexistent_tool_xyz").is_none());
    }

    #[test]
    fn catalog_size_floor() {
        // Lower bound, not exact: catches accidental deletions while
        // letting new tools land without a forced test bump. The
        // critical-tools test below pins the named entries that
        // external callers (Claude Code, ultrareview, etc.) depend on.
        assert!(TOOLS.len() >= 20, "catalog shrunk below 20: {}", TOOLS.len());
    }

    #[test]
    fn cloud_flag_matches_routing_policy() {
        // The four cloud-LLM tools must be gated under local-only.
        for name in ["cloud_query", "review_commit", "review_diff", "warmup"] {
            assert!(find(name).unwrap().cloud, "{name} must have cloud:true (local-only gate)");
        }
        // Local / read-only / RAG / mixed tools must NOT be blanket-gated
        // (parallel_query self-enforces per-task; search/index degrade).
        for name in [
            "query", "plan_query", "list_models", "load_model", "unload_model",
            "vram_status", "scan_models", "queue_status", "list_cloud_models",
            "set_routing_mode", "routing_status", "search_repo", "index_repo",
            "recall_conversation", "train_from_conversations", "write_file",
            "parallel_query",
        ] {
            assert!(!find(name).unwrap().cloud, "{name} must have cloud:false");
        }
    }

    #[test]
    fn critical_tools_present() {
        // Tools that external callers rely on by name. Removing one
        // is a load-bearing breakage; this guard surfaces the change
        // before the live MCP integration discovers it.
        for name in [
            "query", "cloud_query", "review_commit", "review_diff",
            "list_models", "list_cloud_models", "write_file",
            "parallel_query", "set_routing_mode", "recall_conversation",
            "search_repo", "index_repo", "warmup",
        ] {
            assert!(find(name).is_some(), "missing critical tool: {name}");
        }
    }
}
