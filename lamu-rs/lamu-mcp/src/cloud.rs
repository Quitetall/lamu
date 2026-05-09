//! Cloud-routing tool implementations: `cloud_query`,
//! `list_cloud_models`, `review_commit`, `review_diff`.
//!
//! Phase 2.2 (partial) extraction from server.rs. Free async functions
//! that consume the unified `lamu_providers::CloudModel` schema and
//! POST against an OpenAI- or Anthropic-shaped endpoint. Reviewer
//! tools layer on top of `handle_cloud_query` with a fixed system
//! prompt; their git-ref / diff-size guards live alongside.
//!
//! Why these four moved together: they all share the cloud transport
//! layer + the per-provider concurrency rules. The TUI swap path,
//! local query queueing, and VRAM scheduling stay in `server.rs`.

use lamu_providers::CloudModel;
use serde_json::{json, Value};
use tracing::warn;

// ── Loaders + concurrency policy ────────────────────────────────────

pub(crate) fn load_cloud_models() -> Vec<CloudModel> {
    lamu_providers::load_or_empty()
}

/// Per-provider concurrency cap. Conservative by default — only
/// providers we've explicitly tested under parallel load get a cap >1.
/// Unknown / lightly-tested providers are sequential until proven safe.
///
/// Override per-provider with env vars:
///   LAMU_PARALLEL_DEEPSEEK / _ANTHROPIC / _OPENAI / etc.
pub(crate) fn provider_concurrency(model_name: &str, cloud: &[CloudModel]) -> usize {
    let provider = cloud.iter()
        .find(|m| m.name == model_name)
        .map(|m| m.provider.as_str())
        .unwrap_or("");

    // Env override takes precedence.
    let env_var = format!("LAMU_PARALLEL_{}", provider.to_uppercase());
    if let Ok(v) = std::env::var(&env_var) {
        if let Ok(n) = v.parse::<usize>() {
            return n.max(1);
        }
    }

    match provider {
        "deepseek" => 8,
        "anthropic" => 4,
        "openai" => 4,
        // Less tested — start at 1 until proven. Bump via env var.
        _ => 1,
    }
}

// ── list_cloud_models ────────────────────────────────────────────────

pub(crate) fn handle_list_cloud_models() -> String {
    let models = load_cloud_models();
    if models.is_empty() {
        return "(no cloud models — edit ~/.config/lamu/cloud-models.yaml or run `lamu` and press 'n' to add)".into();
    }
    let mut out = String::new();
    for m in &models {
        let key_status = match &m.api_key_env {
            None => "(no key needed — gateway-routed)".to_string(),
            Some(env) => if std::env::var(env).is_ok() { format!("${} ✓", env) } else { format!("${} unset ✗", env) },
        };
        let mid = m.model_id.clone().unwrap_or_else(|| format!("{}/{}", m.provider, m.name));
        out.push_str(&format!(
            "{}  ({})  ctx={}  {}  — {}\n",
            m.name, mid, m.context_max, key_status, m.notes
        ));
    }
    out
}

// ── cloud_query ──────────────────────────────────────────────────────

pub(crate) async fn handle_cloud_query(args: Value) -> String {
    let prompt = args["prompt"].as_str().unwrap_or("");
    if prompt.is_empty() { return "error: prompt is required".into(); }
    let model_name = args["model"].as_str().unwrap_or("deepseek-v4-flash");
    let raw_system = args["system"].as_str().unwrap_or("");

    // Phase 6 step 6: context layer for cloud_query. Defaults match
    // backward-compat behavior — when no plan/context args are passed,
    // prefix is empty and `system` stays bit-identical to the caller's
    // raw `system` string. Central is *off* by default for cloud_query
    // because it's reviewer-shaped today; reviewers should opt in via
    // `system="<role>"` then explicitly request central.
    //
    // Rule: if any of plan_file/context/_with_central is present, the
    // context layer engages with the corresponding tiers. Otherwise the
    // pre-step-6 wire format is preserved exactly.
    let plan_arg = args["plan_file"].as_str();
    let conv_id = args["conversation_id"].as_str().unwrap_or("");

    // When conversation_id is set, prepend the rendered prior turns to
    // the Tactical tier. The recall is bounded (last 20 turns) so a
    // long conversation doesn't blow the context window.
    let conv_recall = if !conv_id.is_empty() {
        match crate::memory::shared() {
            Ok(mem) => match mem.recall(conv_id, 20) {
                Ok(turns) if !turns.is_empty() => crate::memory::render_for_context(&turns),
                Ok(_) => String::new(),
                Err(e) => {
                    tracing::warn!("memory recall({}) failed: {}", conv_id, e);
                    String::new()
                }
            },
            Err(e) => {
                tracing::warn!("memory init failed: {}", e);
                String::new()
            }
        }
    } else {
        String::new()
    };

    let raw_tactical_arg = args["context"].as_str().unwrap_or("");
    // Compose tactical = prior-conversation + caller-supplied. Conv
    // first so the most recent turns sit at the back (closer to the
    // user prompt). Both fit under one cap.
    let raw_tactical = if conv_recall.is_empty() {
        raw_tactical_arg.to_string()
    } else if raw_tactical_arg.is_empty() {
        conv_recall
    } else {
        format!("{}\n\n---\n\n{}", conv_recall, raw_tactical_arg)
    };
    let tactical = if raw_tactical.is_empty() {
        String::new()
    } else {
        truncate_with_marker(&raw_tactical, MAX_TACTICAL_CONTEXT_BYTES)
    };
    let want_layer = plan_arg.is_some() || !tactical.is_empty();
    let system = if want_layer {
        let (s, _stats) = crate::context::prepend_to_system(
            crate::context::ContextConfig {
                central: false, // cloud_query is generic; reviewer paths add central themselves
                plan: plan_arg,
                tactical: &tactical,
                repo: None,
            },
            raw_system,
        );
        s
    } else {
        raw_system.to_string()
    };
    let system = system.as_str();
    let max_tokens = args["max_tokens"].as_u64().unwrap_or(8192) as u32;
    let temperature = args["temperature"].as_f64().unwrap_or(0.3) as f32;
    let include_reasoning = args["include_reasoning"].as_bool().unwrap_or(false);
    // thinking_enabled: explicit override beats per-model default.
    // Default rule: Pro models (and reasoner-named models) think;
    // Flash and similarly-named "fast" tiers don't. Saves 50-80%
    // wall time on simple tasks where reasoning isn't needed.
    let thinking_enabled = args["thinking_enabled"].as_bool().unwrap_or_else(|| {
        let n = model_name.to_lowercase();
        n.contains("pro") || n.contains("reasoner") || n.contains("opus")
    });

    let models = load_cloud_models();
    let entry = match models.iter().find(|m| m.name == model_name) {
        Some(m) => m.clone(),
        None => return format!(
            "error: cloud model '{}' not in cloud-models.yaml. Run `list_cloud_models` to see options.",
            model_name
        ),
    };

    // None means gateway-routed (Bifrost or similar handles auth). Skip
    // the auth header entirely in that case — sending a bogus
    // `Bearer no-key-needed` works against permissive gateways but
    // signals a misconfiguration and breaks any gateway that validates.
    let api_key: Option<String> = match entry.api_key_env.as_deref() {
        Some(env) => match std::env::var(env) {
            Ok(k) => Some(k),
            Err(_) => return format!(
                "error: ${} is not set. Add it via `lamu` (press 'a' on the model row) or export it manually.",
                env
            ),
        },
        None => None,
    };

    let base = match entry.base_url.as_deref() {
        Some(b) => b.trim_end_matches('/').to_string(),
        None => return format!(
            "error: cloud model '{}' has no base_url. Edit ~/.config/lamu/cloud-models.yaml.",
            model_name
        ),
    };
    let model_id = entry.model_id.clone().unwrap_or_else(|| entry.name.clone());
    // Use the explicit provider field. Substring-matching the base URL
    // would misroute a non-Anthropic provider whose host happens to
    // contain "anthropic" (e.g. a proxy domain) — every cloud-models.yaml
    // entry already declares its provider, so trust that.
    let is_anthropic = entry.provider == "anthropic";

    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build() {
        Ok(c) => c,
        Err(e) => return format!("error: client init: {e}"),
    };

    let result: String = if is_anthropic {
        let url = format!("{}/v1/messages", base);
        let mut payload = json!({
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": false,
        });
        if !system.is_empty() { payload["system"] = json!(system); }
        // Anthropic native thinking: send `thinking: {type: "enabled",
        // budget_tokens: N}` to engage extended thinking. Skip when
        // disabled — Anthropic's default for non-reasoner models is
        // disabled anyway, but the explicit form is unambiguous.
        if thinking_enabled {
            // Anthropic constraints: budget_tokens MUST be ≥ 1024 AND
            // ≤ max_tokens. If max_tokens ≤ 1024 there is no valid
            // budget; skip the thinking block + log to stderr (the
            // model will run without extended thinking — caller can
            // retry with a larger max_tokens if they really wanted it).
            if max_tokens > 1024 {
                let budget = (max_tokens / 2).max(1024).min(max_tokens - 1);
                payload["thinking"] = json!({
                    "type": "enabled",
                    "budget_tokens": budget,
                });
            } else {
                warn!(
                    "cloud_query: thinking_enabled=true but max_tokens={} ≤ 1024 (Anthropic min budget); thinking block omitted",
                    max_tokens
                );
            }
        }

        let mut req = client.post(&url)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json");
        if let Some(k) = &api_key {
            req = req.header("x-api-key", k);
        }
        // 1M ctx beta: validated via the shared helper so the trim +
        // reject-bad-value rule lives in lamu-providers only.
        if let Some(val) = lamu_providers::anthropic_beta_header() {
            req = req.header("anthropic-beta", val);
        }
        let resp = match req.json(&payload).send().await {
            Ok(r) => r,
            Err(e) => return format!("error: post {url}: {e}"),
        };
        let v: Value = match resp.json().await {
            Ok(v) => v,
            Err(e) => return format!("error: parse: {e}"),
        };
        if let Some(err) = v.get("error") {
            return format!("anthropic error: {}", err);
        }
        // content is an array of {type: "text"|"thinking", text|thinking: "..."}
        let mut out = String::new();
        let mut thinking = String::new();
        if let Some(blocks) = v["content"].as_array() {
            for b in blocks {
                match b["type"].as_str() {
                    Some("text") => out.push_str(b["text"].as_str().unwrap_or("")),
                    Some("thinking") => thinking.push_str(b["thinking"].as_str().unwrap_or("")),
                    _ => {}
                }
            }
        }
        if include_reasoning && !thinking.is_empty() {
            format!("<think>\n{}\n</think>\n{}", thinking, out)
        } else {
            out
        }
    } else {
        let url = format!("{}/chat/completions", base);
        let mut messages: Vec<Value> = Vec::new();
        if !system.is_empty() {
            messages.push(json!({"role": "system", "content": system}));
        }
        messages.push(json!({"role": "user", "content": prompt}));
        let mut payload = json!({
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": false,
        });
        // DeepSeek (and a growing list of OpenAI-compat providers) take
        // a top-level `thinking` field. {type: "enabled"} engages the
        // reasoning trace; {type: "disabled"} or absence skips it.
        // For Flash by default we skip → ~50–80% wall-time savings on
        // simple tasks. For Pro / reasoner / opus tiers we engage.
        payload["thinking"] = json!({
            "type": if thinking_enabled { "enabled" } else { "disabled" }
        });
        let mut req = client.post(&url);
        if let Some(k) = &api_key {
            req = req.bearer_auth(k);
        }
        let resp = match req.json(&payload).send().await {
            Ok(r) => r,
            Err(e) => return format!("error: post {url}: {e}"),
        };
        let v: Value = match resp.json().await {
            Ok(v) => v,
            Err(e) => return format!("error: parse: {e}"),
        };
        if let Some(err) = v.get("error") {
            return format!("provider error: {}", err);
        }
        let msg = &v["choices"][0]["message"];
        let content = msg["content"].as_str().unwrap_or("");
        let reasoning = msg["reasoning_content"].as_str().unwrap_or("");
        if include_reasoning && !reasoning.is_empty() {
            format!("<think>\n{}\n</think>\n{}", reasoning, content)
        } else {
            content.to_string()
        }
    };

    // Persist to conversation memory if conversation_id was set.
    // Best-effort: a memory failure must not fail the query — the
    // model already produced a reply, the user wants it.
    if !conv_id.is_empty() {
        if let Ok(mem) = crate::memory::shared() {
            let meta = format!("model={}", model_name);
            if let Err(e) = mem.append_turns(
                conv_id,
                &[
                    ("user", prompt, None),
                    ("assistant", &result, Some(&meta)),
                ],
            ) {
                tracing::warn!("memory append({}) failed: {}", conv_id, e);
            }
        }
    }

    result
}

// ── Conversation recall (memory tier) ───────────────────────────────

pub(crate) fn handle_recall_conversation(args: Value) -> String {
    let conv_id = args["conversation_id"].as_str().unwrap_or("");
    if conv_id.is_empty() {
        return "error: conversation_id is required".into();
    }
    let limit = args["limit"].as_u64().unwrap_or(0) as usize;

    let mem = match crate::memory::shared() {
        Ok(m) => m,
        Err(e) => return format!("error: memory init: {}", e),
    };
    let turns = match mem.recall(conv_id, limit) {
        Ok(t) => t,
        Err(e) => return format!("error: recall {}: {}", conv_id, e),
    };
    if turns.is_empty() {
        return format!("(no turns recorded for conversation_id='{}')", conv_id);
    }
    let mut out = format!("=== Conversation '{}' — {} turns ===\n\n", conv_id, turns.len());
    for t in &turns {
        let meta = t
            .metadata
            .as_deref()
            .map(|m| format!(" [{}]", m))
            .unwrap_or_default();
        out.push_str(&format!("**{}**{} (idx {}, ts {})\n{}\n\n", t.role, meta, t.idx, t.ts, t.content));
    }
    out
}

// ── DeepSeek V4 Pro reviewer (project policy) ───────────────────────
//
// Every commit goes through this. The system prompt below tells V4 Pro
// to focus on issues that matter — security, correctness, edge cases,
// architecture — and to call out problems even when none exist. The
// model's reasoning_content is included so the human can see HOW the
// review was reached, not just the conclusion.

const REVIEW_SYSTEM_PROMPT: &str = "You are a senior staff engineer doing a code review. Your job is to find real issues, not to pat anyone on the back.\n\nAlways check:\n  1. SECURITY — injection (SQL/shell/XSS/prompt), auth/authz holes, secrets in code, unsafe deserialization, TOCTOU, missing input validation.\n  2. CORRECTNESS — off-by-one, null/empty cases, integer overflow, floating-point traps, race conditions, deadlocks, missing error handling.\n  3. EDGE CASES — what happens at boundaries, with empty inputs, with hostile inputs, under concurrency, on partial failure, on retry.\n  4. ARCHITECTURE — does this fit the existing design? Does it leak abstraction? Does it create coupling that will hurt later? Is there a simpler shape?\n  5. CLARITY — would a stranger understand the intent? Are names accurate? Are comments necessary or noise?\n\nFormat your output:\n  - One-sentence verdict (PASS / PASS WITH NITS / NEEDS CHANGES / REJECT).\n  - Numbered list of findings, each: severity [BUG/SECURITY/STYLE/QUESTION], file:line if knowable, the problem, the suggested fix.\n  - End with a single 'Recommend' line.\n\nBe terse. Be honest. Don't praise unless something is genuinely surprising in a good way. If the code is fine, say so in one line and stop.";

/// Cap on diff size sent to the reviewer. 200 KiB is generous (≈ 4K
/// lines of typical code) but bounded — anything larger gets truncated
/// with a marker so the model knows it's not seeing the whole change.
const MAX_REVIEW_DIFF_BYTES: usize = 200 * 1024;

/// Cap on the Tactical-tier `context` arg that callers can pass to
/// review_commit / review_diff / cloud_query. Same shape as the diff
/// cap above.
pub(crate) const MAX_TACTICAL_CONTEXT_BYTES: usize = 200 * 1024;

/// Validate a git ref / commit. Accepts: hex SHA (7-40 chars), HEAD
/// followed by any sequence of `~N` / `^[N]` suffixes (HEAD^^,
/// HEAD~1^2, HEAD^^^, etc. — all valid git refs), or a plain refname
/// matching git's safe character set (alnum + _ - . /, no leading
/// '-' or '.', no '..').
fn is_safe_git_ref(s: &str) -> bool {
    if s.is_empty() || s.starts_with('-') { return false; }
    // Hex SHA / abbrev.
    if s.len() >= 7 && s.len() <= 40 && s.chars().all(|c| c.is_ascii_hexdigit()) {
        return true;
    }
    // HEAD with any chain of ~N and ^[N] suffixes.
    if let Some(rest) = s.strip_prefix("HEAD") {
        return parse_rev_suffix(rest);
    }
    // General refname.
    if s.contains("..") || s.starts_with('.') { return false; }
    s.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | '/'))
}

/// Walk a sequence of `~N` and `^[N]` suffixes. `^` alone means parent;
/// `^N` means Nth parent. Returns true iff the entire suffix consumes.
fn parse_rev_suffix(mut s: &str) -> bool {
    while !s.is_empty() {
        let first = s.as_bytes()[0];
        if first == b'~' || first == b'^' {
            s = &s[1..];
            // Optional digit run.
            let digit_end = s.bytes().take_while(|b| b.is_ascii_digit()).count();
            s = &s[digit_end..];
        } else {
            return false;
        }
    }
    true
}

/// Truncate `text` to at most `limit` bytes, snapping back to the
/// nearest UTF-8 char boundary so we never split a multi-byte
/// codepoint mid-stream. Appends a marker describing how much was
/// dropped so the reviewer LLM knows it didn't see the full diff.
pub(crate) fn truncate_with_marker(text: &str, limit: usize) -> String {
    if text.len() <= limit { return text.to_string(); }
    // Walk back to the last char boundary at or before `limit`.
    let mut cut = limit;
    while cut > 0 && !text.is_char_boundary(cut) {
        cut -= 1;
    }
    let mut out = text[..cut].to_string();
    out.push_str(&format!(
        "\n\n[…truncated {} more bytes — diff exceeded {} byte review limit]",
        text.len() - cut, limit
    ));
    out
}

pub(crate) async fn handle_review_commit(args: Value) -> String {
    let commit = args["commit"].as_str().unwrap_or("HEAD");
    let repo = args["repo"].as_str().unwrap_or(".");
    let focus = args["focus"].as_str().unwrap_or("");

    if !is_safe_git_ref(commit) {
        return format!(
            "error: commit '{}' rejected — must be a hex SHA, HEAD with ~/^ suffixes, or a safe refname.",
            commit
        );
    }

    // is_safe_git_ref already rejects anything starting with '-', so
    // git can't interpret commit as a flag. No defense-in-depth needed.
    let out = match std::process::Command::new("git")
        .current_dir(repo)
        .args(["show", "--stat", "--patch", commit])
        .output()
    {
        Ok(o) => o,
        Err(e) => return format!("error: spawn git: {}", e),
    };

    if !out.status.success() {
        return format!("error: git show {} failed: {}", commit,
            String::from_utf8_lossy(&out.stderr).trim());
    }
    let diff_text = String::from_utf8_lossy(&out.stdout).to_string();
    if diff_text.trim().is_empty() {
        return format!("error: empty diff for {}", commit);
    }
    let diff_text = truncate_with_marker(&diff_text, MAX_REVIEW_DIFF_BYTES);

    let mut prompt = String::new();
    if !focus.is_empty() {
        prompt.push_str(&format!("Focus the review on: {}\n\n", focus));
    }
    prompt.push_str("Here is the commit to review (full diff):\n\n```\n");
    prompt.push_str(&diff_text);
    prompt.push_str("\n```\n");

    let plan_arg = args["plan_file"].as_str();
    let raw_tactical_arg = args["context"].as_str().unwrap_or("");
    let auto = args["auto_context"].as_bool().unwrap_or(false);

    // When auto_context=true, run the diff-derived context assembler
    // (changed-files + tree-sitter symbols + ripgrep callers) and
    // prepend it to the caller-supplied tactical blob. Caller-supplied
    // sits at the end so it stays closer to the role prompt.
    let auto_blob = if auto {
        match crate::auto_context::assemble_auto_context(commit, std::path::Path::new(repo)) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("auto_context: assemble failed: {}", e);
                String::new()
            }
        }
    } else {
        String::new()
    };
    let raw_tactical = if auto_blob.is_empty() {
        raw_tactical_arg.to_string()
    } else if raw_tactical_arg.is_empty() {
        auto_blob
    } else {
        format!("{}\n\n---\n\n{}", auto_blob, raw_tactical_arg)
    };
    let tactical = truncate_with_marker(&raw_tactical, MAX_TACTICAL_CONTEXT_BYTES);
    let (system, _stats) = crate::context::prepend_to_system(
        crate::context::ContextConfig {
            central: true,
            plan: plan_arg,
            tactical: &tactical,
            repo: Some(std::path::Path::new(repo)),
        },
        REVIEW_SYSTEM_PROMPT,
    );

    let review_args = json!({
        "model": "deepseek-v4-pro",
        "prompt": prompt,
        "system": system,
        "max_tokens": 8192,
        "temperature": 0.2,
        "include_reasoning": false,
    });
    let review = handle_cloud_query(review_args).await;
    format!("=== Review of {} (DeepSeek V4 Pro) ===\n\n{}", commit, review)
}

pub(crate) async fn handle_review_diff(args: Value) -> String {
    let diff = args["diff"].as_str().unwrap_or("");
    if diff.is_empty() {
        return "error: 'diff' is required".into();
    }
    let diff = truncate_with_marker(diff, MAX_REVIEW_DIFF_BYTES);
    let raw_tactical = args["context"].as_str().unwrap_or("");
    let tactical = truncate_with_marker(raw_tactical, MAX_TACTICAL_CONTEXT_BYTES);
    let focus = args["focus"].as_str().unwrap_or("");

    // Step 6: `context` moves from in-prompt body to Tactical-tier
    // prefix on the system prompt. Cache-friendlier (DeepSeek caches
    // the system prefix), and consistent with review_commit + cloud_query.
    let mut prompt = String::new();
    if !focus.is_empty() {
        prompt.push_str(&format!("Focus the review on: {}\n\n", focus));
    }
    prompt.push_str("Diff to review:\n\n```\n");
    prompt.push_str(&diff);
    prompt.push_str("\n```\n");

    let plan_arg = args["plan_file"].as_str();
    let (system, _stats) = crate::context::prepend_to_system(
        crate::context::ContextConfig {
            central: true,
            plan: plan_arg,
            tactical: &tactical,
            repo: None,        // review_diff doesn't carry a repo path
        },
        REVIEW_SYSTEM_PROMPT,
    );

    let review_args = json!({
        "model": "deepseek-v4-pro",
        "prompt": prompt,
        "system": system,
        "max_tokens": 8192,
        "temperature": 0.2,
        "include_reasoning": false,
    });
    let review = handle_cloud_query(review_args).await;
    format!("=== Diff review (DeepSeek V4 Pro) ===\n\n{}", review)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lamu_providers::QuotaState;

    fn dummy_cloud(name: &str, provider: &str) -> CloudModel {
        CloudModel {
            name: name.into(),
            provider: provider.into(),
            model_id: None,
            context_max: 0,
            notes: String::new(),
            quota: QuotaState::Available,
            api_key_env: None,
            base_url: None,
            chat_path: None,
        }
    }

    #[test]
    fn provider_concurrency_known_providers() {
        let cloud = vec![
            dummy_cloud("ds", "deepseek"),
            dummy_cloud("claude", "anthropic"),
            dummy_cloud("gpt", "openai"),
        ];
        assert_eq!(provider_concurrency("ds", &cloud), 8);
        assert_eq!(provider_concurrency("claude", &cloud), 4);
        assert_eq!(provider_concurrency("gpt", &cloud), 4);
    }

    #[test]
    fn provider_concurrency_unknown_defaults_to_1() {
        let cloud = vec![
            dummy_cloud("kimi", "moonshot"),
            dummy_cloud("qwen", "alibaba"),
        ];
        assert_eq!(provider_concurrency("kimi", &cloud), 1);
        assert_eq!(provider_concurrency("qwen", &cloud), 1);
        assert_eq!(provider_concurrency("not-in-yaml", &cloud), 1);
    }

    #[test]
    fn safe_git_ref_accepts_hex_sha() {
        assert!(is_safe_git_ref("abc1234"));
        assert!(is_safe_git_ref("abc1234567890"));
        assert!(is_safe_git_ref(&"a".repeat(40)));
    }

    #[test]
    fn safe_git_ref_accepts_head_variants() {
        assert!(is_safe_git_ref("HEAD"));
        assert!(is_safe_git_ref("HEAD~1"));
        assert!(is_safe_git_ref("HEAD~10"));
        assert!(is_safe_git_ref("HEAD^"));
        assert!(is_safe_git_ref("HEAD^2"));
        // Chained suffixes — all valid git revisions.
        assert!(is_safe_git_ref("HEAD^^"));
        assert!(is_safe_git_ref("HEAD~1^"));
        assert!(is_safe_git_ref("HEAD~1^2"));
        assert!(is_safe_git_ref("HEAD^^^"));
        assert!(is_safe_git_ref("HEAD~3~2"));
    }

    #[test]
    fn safe_git_ref_accepts_branch_names() {
        assert!(is_safe_git_ref("main"));
        assert!(is_safe_git_ref("feature/x-123"));
        assert!(is_safe_git_ref("release-1.0"));
    }

    #[test]
    fn safe_git_ref_rejects_dangerous() {
        assert!(!is_safe_git_ref(""));
        assert!(!is_safe_git_ref("--upload-pack=evil"));
        assert!(!is_safe_git_ref("-v"));
        assert!(!is_safe_git_ref("../escape"));
        assert!(!is_safe_git_ref(".hidden"));
        assert!(!is_safe_git_ref("HEAD; rm -rf /"));
        assert!(!is_safe_git_ref("HEAD~abc"));
        assert!(!is_safe_git_ref("branch with space"));
        assert!(!is_safe_git_ref("name$with#meta"));
    }

    #[test]
    fn truncate_marker_short_string_unchanged() {
        let s = "short";
        assert_eq!(truncate_with_marker(s, 100), s);
    }

    #[test]
    fn truncate_marker_long_string_truncated() {
        let s = "x".repeat(1000);
        let out = truncate_with_marker(&s, 100);
        assert!(out.len() < s.len());
        assert!(out.contains("truncated"));
        assert!(out.contains("900 more bytes"));
    }

    #[test]
    fn truncate_marker_does_not_panic_on_utf8_boundary() {
        // 4-byte UTF-8 codepoint (😀) at position 99 — limit=100 falls
        // mid-codepoint. Naive slicing panics; we must snap back.
        let mut s = "x".repeat(99);
        s.push('😀');
        s.push_str(&"y".repeat(50));
        let out = truncate_with_marker(&s, 100);
        // No panic = test passed. Verify content sane.
        assert!(out.starts_with(&"x".repeat(99)));
        assert!(out.contains("truncated"));
    }
}
