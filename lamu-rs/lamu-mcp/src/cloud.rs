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

    // Env override takes precedence. parse_env_or warns on a set-but-garbage
    // value instead of silently ignoring it. Sentinel 0 = "unset/invalid".
    let env_var = format!("LAMU_PARALLEL_{}", provider.to_uppercase());
    let n = lamu_core::config::parse_env_or::<usize>(&env_var, 0);
    if n > 0 {
        return n;
    }

    match provider {
        "deepseek" => 8,
        "mimo" => 8,
        "anthropic" => 4,
        "openai" => 4,
        // Less tested — start at 1 until proven. Bump via env var.
        _ => 1,
    }
}

/// Process-global per-provider concurrency semaphore. Created once per
/// provider on first use; the `cap` arg only takes effect on creation.
///
/// Lives here (not on LamuMcpServer) so it gates EVERY cloud request —
/// standalone `cloud_query`, the reviewer pipeline, AND `parallel_query`
/// tasks (which call handle_cloud_query) — with one shared limit. The
/// audit found parallel_query's per-invocation semaphores let N
/// concurrent callers each get a full "8 each", and standalone
/// cloud_query had no gate at all; a single global semaphore per provider
/// closes both holes.
fn cloud_provider_sem(provider: &str, cap: usize) -> std::sync::Arc<tokio::sync::Semaphore> {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};
    use tokio::sync::Semaphore;
    static SEMS: OnceLock<Mutex<HashMap<String, Arc<Semaphore>>>> = OnceLock::new();
    let map = SEMS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut m = map.lock().unwrap_or_else(|e| e.into_inner());
    m.entry(provider.to_string())
        .or_insert_with(|| Arc::new(Semaphore::new(cap.max(1))))
        .clone()
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

/// Pure parse of the `LAMU_AUTOCAPTURE` env value: true ONLY for "1" or
/// "true" (case-insensitive), false for everything else (including
/// unset → `None`). Split out so it is unit-testable without touching
/// the process environment.
fn truthy_env_flag(val: Option<&str>) -> bool {
    matches!(
        val.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("1") | Some("true")
    )
}

/// Whether memory autocapture is enabled. DEFAULT FALSE: only "1"/"true"
/// (case-insensitive) in `LAMU_AUTOCAPTURE` turns it on. When unset the
/// autocapture spawn below is unreachable — zero extra MiMo/embedding
/// calls, zero added latency, zero behavior change.
fn autocapture_enabled() -> bool {
    truthy_env_flag(std::env::var("LAMU_AUTOCAPTURE").ok().as_deref())
}

/// Temporary / memoryless ("incognito") chat. When true, the call does
/// NOT recall prior turns, does NOT persist the turn to conversation
/// memory, and is NOT autocaptured — even if `conversation_id` is set.
/// Enabled per-call via the `ephemeral` arg OR globally via
/// `LAMU_EPHEMERAL=1`/`true` (same "1"/"true" parser as autocapture).
fn ephemeral_enabled(arg: Option<bool>) -> bool {
    arg.unwrap_or(false) || truthy_env_flag(std::env::var("LAMU_EPHEMERAL").ok().as_deref())
}

/// Coerce the per-call `ephemeral` arg from EITHER a JSON bool (`true`)
/// OR a "true"/"1" string. `None` when the key is absent (so the env
/// fallback applies). Incognito is privacy-adjacent, so a client that
/// sends `"ephemeral":"true"` must engage it, not silently fall through.
fn coerce_ephemeral_arg(v: &Value) -> Option<bool> {
    v.as_bool()
        .or_else(|| v.as_str().map(|s| truthy_env_flag(Some(s))))
}

pub(crate) async fn handle_cloud_query(args: Value) -> String {
    let prompt = args["prompt"].as_str().unwrap_or("");
    if prompt.is_empty() { return "error: prompt is required".into(); }
    let model_name = args["model"].as_str().unwrap_or("mimo-v2.5");
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
    // Temporary/memoryless chat: when ephemeral (per-call `ephemeral` arg
    // or global LAMU_EPHEMERAL), blank conv_id so the recall / persist /
    // autocapture paths below — all gated on `!conv_id.is_empty()` — skip
    // entirely. Nothing is read from or written to conversation memory.
    // Accept bool OR "true"/"1" string for the per-call arg (privacy-
    // adjacent: a string must not silently fall through to not-ephemeral).
    let ephemeral = ephemeral_enabled(coerce_ephemeral_arg(&args["ephemeral"]));
    let conv_id = if ephemeral {
        ""
    } else {
        args["conversation_id"].as_str().unwrap_or("")
    };

    // V4 Batch 4: when conversation_id is set, recall prior turns via
    // semantic ranking when available (top-K relevant + last 5
    // recent). Falls back to chronological "last 20" when no
    // embeddings key is available or scoring fails. Long
    // conversations no longer bury relevant turns under recency.
    let conv_recall = if !conv_id.is_empty() {
        match crate::memory::shared() {
            Ok(mem) => match crate::memory::recall_ranked(mem, conv_id, prompt, 10, 5).await {
                Ok(turns) if !turns.is_empty() => crate::memory::render_for_context(&turns),
                Ok(_) => String::new(),
                Err(e) => {
                    tracing::warn!("memory recall_ranked({}) failed: {}", conv_id, e);
                    // Fall back to plain chronological.
                    mem.recall(conv_id, 20)
                        .ok()
                        .map(|turns| crate::memory::render_for_context(&turns))
                        .unwrap_or_default()
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

    // Global per-provider concurrency gate. Held for the whole request so
    // the in-flight count across ALL cloud paths can't exceed the
    // provider's cap (parallel_query's per-call sems were per-invocation;
    // standalone cloud_query had none). handle_cloud_query never recurses,
    // so this can't self-deadlock even at cap=1.
    let provider_cap = provider_concurrency(model_name, &models);
    let _permit = {
        let sem = cloud_provider_sem(&entry.provider, provider_cap);
        match sem.acquire_owned().await {
            Ok(p) => p,
            Err(_) => return "error: cloud concurrency semaphore closed".into(),
        }
    };

    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build() {
        Ok(c) => c,
        Err(e) => return format!("error: client init: {e}"),
    };

    let result: String = if is_anthropic {
        // Honor entry.chat_path (proxy/gateway suffix) via the shared
        // CloudModel::chat_url — the hand-built format! dropped it, so a
        // model with a custom chat_path 404'd from MCP while working via
        // the CLI (which already uses chat_url). base is Some here (guarded
        // above), so the fallback arg is never used. (#10)
        let url = entry.chat_url(&base);
        let mut payload = json!({
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": false,
        });
        // V5 J: Anthropic doesn't auto-cache by byte-prefix; it uses
        // explicit `cache_control: {type: "ephemeral"}` on a system-
        // prompt block. Wrap the system prompt in the array form with
        // a cache_control marker on the central+plan portion (the
        // stable prefix). DeepSeek uses byte-prefix auto-cache so this
        // path only kicks in for anthropic.
        if !system.is_empty() {
            payload["system"] = json!([{
                "type": "text",
                "text": system,
                "cache_control": { "type": "ephemeral" }
            }]);
        }
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
                // Anthropic requires temperature == 1 whenever extended
                // thinking is engaged; any other value (default 0.3) is a
                // hard 400. This was the silent failure that knocked the
                // ensemble's anthropic second-reviewer offline. (#11)
                payload["temperature"] = json!(1);
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
        // reqwest does NOT error on 4xx/5xx. Capture the status before
        // consuming the body so a non-2xx whose body lacks an {error:…}
        // key (e.g. a proxy's {"detail":…}/{"message":…}) isn't silently
        // returned as a clean empty reply and persisted to memory. (#29)
        let status = resp.status();
        let v: Value = match resp.json().await {
            Ok(v) => v,
            Err(e) => return format!("error: parse: {e}"),
        };
        if let Some(err) = v.get("error") {
            // Prefix "error:" so tools_call's isError flag fires AND the
            // ensemble/critic gates (which test `starts_with("error:")`)
            // don't feed an API-error blob into merge_unique_findings.
            return format!("error: anthropic API: {}", err);
        }
        if !status.is_success() {
            let body: String = v.to_string().chars().take(300).collect();
            return format!("error: anthropic HTTP {}: {}", status.as_u16(), body);
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
        let url = entry.chat_url(&base); // honor chat_path; see #10 above
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
        // OpenRouter's optional app-attribution headers (shown on their
        // dashboard / leaderboards). Harmless to other OpenAI-compat hosts,
        // but only send for openrouter to avoid surprising strict gateways.
        if entry.provider == "openrouter" {
            req = req
                .header("HTTP-Referer", "https://github.com/Quitetall/lamu")
                .header("X-Title", "LAMU");
        }
        let resp = match req.json(&payload).send().await {
            Ok(r) => r,
            Err(e) => return format!("error: post {url}: {e}"),
        };
        // See anthropic branch — reqwest doesn't error on 4xx/5xx, so check
        // the status so a non-2xx with a non-{error:…} body isn't returned
        // as a clean empty reply. (#29)
        let status = resp.status();
        let v: Value = match resp.json().await {
            Ok(v) => v,
            Err(e) => return format!("error: parse: {e}"),
        };
        if let Some(err) = v.get("error") {
            // See anthropic branch above — "error:" prefix is load-bearing
            // for the isError flag and the review-pipeline error gates.
            return format!("error: provider API: {}", err);
        }
        if !status.is_success() {
            let body: String = v.to_string().chars().take(300).collect();
            return format!("error: provider HTTP {}: {}", status.as_u16(), body);
        }
        if let Some(usage) = v.get("usage") {
            tracing::info!(target: "lamu_bench", "cloud_query usage model={} {}", model_name, usage);
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

    // OPT-IN memory autocapture (default OFF). When LAMU_AUTOCAPTURE is
    // unset this whole block is unreachable — zero extra cost, zero
    // latency, zero behavior change. Gated additionally on a real
    // conversation turn (`conv_id` non-empty — which excludes every
    // internal warmup/review caller, none of which set conversation_id)
    // and on a non-error reply. The work runs on a DETACHED thread that
    // captures only owned data, so `result` is returned IMMEDIATELY below
    // and the user's reply is never blocked or delayed; facts land
    // shortly AFTER the reply (eventual consistency, by design).
    if autocapture_enabled() && !conv_id.is_empty() && !result.starts_with("error:") {
        let user = prompt.to_string();
        let assistant = result.clone();
        let conv = conv_id.to_string();
        spawn_autocapture(user, assistant, conv);
    }

    result
}

/// Run the autocapture pipeline (extract durable facts from this
/// exchange, store the novel ones) detached and best-effort, so the
/// caller's `result` is returned without waiting.
///
/// Runs on a dedicated detached thread with its own current-thread runtime
/// (rather than `tokio::spawn`) so the blocking SQLite + embedding work is
/// isolated from the caller's async worker pool. Only owned `String`s cross
/// the thread boundary; the thread is detached and never joined.
///
/// NOTE: the autocapture future IS `Send` — a previous version of this
/// comment claimed `recall_ranked` holds a `Connection` guard across an
/// `.await`, which is false (`remember_if_novel`/`recall` drop every guard
/// before awaiting). The dedicated thread is for blocking-work isolation,
/// not a Send workaround.
struct AutocaptureJob {
    user: String,
    assistant: String,
    conv: String,
}

/// Bounded queue depth. Beyond this, turns are DROPPED (best-effort) rather
/// than spawning unbounded threads/work under a burst.
const AUTOCAPTURE_QUEUE_CAP: usize = 64;

/// One background worker (own current-thread runtime) draining a bounded
/// channel — replaces the old thread-per-turn spawn, which could fan out
/// unbounded threads under high cloud_query volume. Lazily started on first
/// use. Single worker ⇒ sequential extraction (bounded concurrency = 1),
/// fine for best-effort background capture.
fn autocapture_sender() -> &'static std::sync::mpsc::SyncSender<AutocaptureJob> {
    static SENDER: std::sync::OnceLock<std::sync::mpsc::SyncSender<AutocaptureJob>> =
        std::sync::OnceLock::new();
    SENDER.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::sync_channel::<AutocaptureJob>(AUTOCAPTURE_QUEUE_CAP);
        let spawned = std::thread::Builder::new()
            .name("lamu-autocapture".into())
            .spawn(move || {
                let rt = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(rt) => rt,
                    Err(e) => {
                        tracing::error!("autocapture worker: runtime build failed: {e}");
                        return;
                    }
                };
                while let Ok(job) = rx.recv() {
                    rt.block_on(run_autocapture_job(job));
                }
            });
        if let Err(e) = spawned {
            tracing::error!("autocapture worker: thread spawn failed: {e}");
        }
        tx
    })
}

async fn run_autocapture_job(job: AutocaptureJob) {
    let AutocaptureJob { user, assistant, conv } = job;
    match crate::lifetime_memory::extract_from_exchange(&user, &assistant).await {
        Ok(facts) => {
            let mut stored = 0usize;
            for f in &facts {
                match crate::lifetime_memory::remember_if_novel(f, "fact", &conv).await {
                    Ok(Some(_)) => stored += 1,
                    Ok(None) => {} // skipped as a near-duplicate
                    Err(e) => tracing::debug!("autocapture({conv}): store fact failed: {e}"),
                }
            }
            tracing::info!(
                "autocapture({conv}): extracted {} fact(s), stored {} novel",
                facts.len(),
                stored
            );
        }
        Err(e) => tracing::debug!("autocapture({conv}): extraction failed: {e}"),
    }
}

fn spawn_autocapture(user: String, assistant: String, conv: String) {
    let job = AutocaptureJob { user, assistant, conv };
    // try_send, never block the caller. Full queue or a dead worker → drop
    // the turn (best-effort capture) with a warn, not unbounded growth.
    match autocapture_sender().try_send(job) {
        Ok(()) => {}
        Err(std::sync::mpsc::TrySendError::Full(j)) => tracing::warn!(
            "autocapture({}): queue full (cap {AUTOCAPTURE_QUEUE_CAP}), dropping turn",
            j.conv
        ),
        Err(std::sync::mpsc::TrySendError::Disconnected(j)) => {
            tracing::error!("autocapture({}): worker gone, dropping turn", j.conv)
        }
    }
}

/// V5 improvement H: dedupe overlapping content between auto_context
/// and caller-supplied tactical. Walks both as 6-line shingles, hashes
/// each shingle, drops shingles from the caller-supplied side that
/// already appear in auto_context. Conservative: only drops on exact
/// shingle match — partial overlap stays.
fn dedupe_tactical(auto_blob: &str, manual: &str) -> String {
    use std::collections::HashSet;
    const SHINGLE: usize = 6;
    let auto_lines: Vec<&str> = auto_blob.lines().collect();
    let mut auto_shingles: HashSet<u64> = HashSet::new();
    for w in auto_lines.windows(SHINGLE) {
        auto_shingles.insert(hash_shingle(w));
    }
    if auto_shingles.is_empty() {
        return format!("{}\n\n---\n\n{}", auto_blob, manual);
    }
    let manual_lines: Vec<&str> = manual.lines().collect();
    let mut keep: Vec<&str> = Vec::with_capacity(manual_lines.len());
    let mut skip_until = 0usize;
    for i in 0..manual_lines.len() {
        if i < skip_until {
            continue;
        }
        if i + SHINGLE <= manual_lines.len() {
            let h = hash_shingle(&manual_lines[i..i + SHINGLE]);
            if auto_shingles.contains(&h) {
                // Found an overlap — skip this shingle's worth of
                // lines. Subsequent lines may still uniquely contribute.
                skip_until = i + SHINGLE;
                continue;
            }
        }
        keep.push(manual_lines[i]);
    }
    let pruned = keep.join("\n");
    if pruned.trim().is_empty() {
        // Whole manual blob was redundant.
        auto_blob.to_string()
    } else {
        format!("{}\n\n---\n\n{}", auto_blob, pruned)
    }
}

fn hash_shingle(lines: &[&str]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for l in lines {
        l.trim().hash(&mut h);
    }
    h.finish()
}

/// V5 improvement I: per-process timestamp for the last `review_commit`
/// invocation. When the next call comes in after IDLE_THRESHOLD, we
/// implicitly run a warmup pass first so DeepSeek's prompt-cache TTL
/// (server-side, ~1 hour) has been refreshed. No explicit `warmup`
/// tool call needed.
fn last_review_call_ts() -> &'static parking_lot::Mutex<std::time::Instant> {
    use std::sync::OnceLock;
    static T: OnceLock<parking_lot::Mutex<std::time::Instant>> = OnceLock::new();
    T.get_or_init(|| parking_lot::Mutex::new(std::time::Instant::now() - std::time::Duration::from_secs(3600 * 2)))
}

/// Returns true when the previous review was > IDLE_THRESHOLD ago AND
/// updates the timestamp. Caller can use the boolean to decide
/// whether to fire an implicit warmup before the real review.
fn should_implicit_warmup() -> bool {
    const IDLE_THRESHOLD_SECS: u64 = 30 * 60; // 30 min — DeepSeek cache TTL is ~1h
    let mut ts = last_review_call_ts().lock();
    let now = std::time::Instant::now();
    let elapsed = now.duration_since(*ts).as_secs();
    *ts = now;
    elapsed > IDLE_THRESHOLD_SECS
}

/// V5 improvement F: heuristic for skipping auto_context. Counts
/// added lines (`^+`) and changed files via `git show --shortstat`.
/// Returns true when both counts are below thresholds — the diff is
/// small enough to live entirely in the user prompt, no need to
/// expand to full file bodies + caller scans.
fn is_trivial_diff(commit: &str, repo: &std::path::Path) -> bool {
    const MAX_LINES: usize = 50;
    const MAX_FILES: usize = 3;
    let out = std::process::Command::new("git")
        .current_dir(repo)
        .args(["show", "--shortstat", "--format=", commit])
        .output();
    let Ok(out) = out else { return false };
    if !out.status.success() {
        return false;
    }
    // Output line: " 3 files changed, 12 insertions(+), 4 deletions(-)"
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout.lines().last().unwrap_or("").trim();
    let mut files = 0usize;
    let mut insertions = 0usize;
    for token_pair in line.split(',') {
        let tp = token_pair.trim();
        let mut parts = tp.splitn(2, ' ');
        if let (Some(num), Some(rest)) = (parts.next(), parts.next()) {
            if let Ok(n) = num.parse::<usize>() {
                if rest.starts_with("file") {
                    files = n;
                } else if rest.starts_with("insertion") {
                    insertions = n;
                }
            }
        }
    }
    insertions <= MAX_LINES && files <= MAX_FILES
}

// ── Cache warmup ────────────────────────────────────────────────────

/// Prime DeepSeek's prompt cache with the central + plan tier of a
/// future review_commit call. Runs a 1-token completion (cheapest
/// possible response) so the prefix bytes are cached. Subsequent
/// review_commit calls in this session hit cache from byte 0.
pub(crate) async fn handle_warmup(args: Value) -> String {
    let plan_arg = args["plan_file"].as_str();
    let repo_str = args["repo"].as_str().unwrap_or(".");

    let (system, _stats) = crate::context::prepend_to_system(
        crate::context::ContextConfig {
            central: true,
            plan: plan_arg,
            tactical: "",
            repo: Some(std::path::Path::new(repo_str)),
        },
        REVIEW_SYSTEM_PROMPT,
    );

    let warmup_args = json!({
        "model": "mimo-v2.5-pro",
        "prompt": "ACK only — warmup",
        "system": system,
        "max_tokens": 1,
        "temperature": 0.0,
        "include_reasoning": false,
    });
    let _ = handle_cloud_query(warmup_args).await;
    format!(
        "warmup: cached central+plan prefix ({} bytes) for plan={:?}",
        system.len(),
        plan_arg
    )
}

// ── Repo retrieval (RAG) ────────────────────────────────────────────

pub(crate) async fn handle_search_repo(args: Value) -> String {
    let query = args["query"].as_str().unwrap_or("");
    if query.is_empty() {
        return "error: query is required".into();
    }
    let mode = crate::rag::SearchMode::parse(args["mode"].as_str().unwrap_or("auto"));
    let k = args["k"].as_u64().unwrap_or(8) as usize;
    let repo_str = args["repo"].as_str().unwrap_or(".");
    let repo = std::path::Path::new(repo_str);

    match crate::rag::search(query, mode, k, repo).await {
        Ok(hits) if hits.is_empty() => format!("(no matches for '{}')", query),
        Ok(hits) => {
            let mut out = format!("=== {} hits for '{}' ===\n\n", hits.len(), query);
            for h in &hits {
                let line_part = h.line.map(|l| format!(":{}", l)).unwrap_or_default();
                let score_part = h
                    .score
                    .map(|s| format!(" (score {:.3})", s))
                    .unwrap_or_default();
                out.push_str(&format!(
                    "**{}{}** [{}]{}\n{}\n\n",
                    h.path, line_part, h.source, score_part, h.snippet
                ));
            }
            out
        }
        Err(e) => format!("error: search failed: {}", e),
    }
}

pub(crate) async fn handle_index_repo(args: Value) -> String {
    let repo_str = args["repo"].as_str().unwrap_or(".");
    let repo = std::path::Path::new(repo_str);
    let force = args["force"].as_bool().unwrap_or(false);

    match crate::rag::index_repo(repo, force).await {
        Ok(0) => "(no chunks indexed — repo unchanged or all files already up to date)".into(),
        Ok(n) => format!("indexed {} chunks into ~/.local/share/lamu/embeddings.db", n),
        Err(e) => format!("error: index_repo failed: {}", e),
    }
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

// ── MiMo V2.5 Pro reviewer (project policy) ─────────────────────────
//
// Every commit goes through this. The system prompt below tells the
// reviewer to focus on issues that matter — security, correctness,
// edge cases, architecture — and to call out problems even when none
// exist. Reviews run with include_reasoning:false today (just the
// verdict + findings); flip the per-call flag if the human wants the
// model's reasoning_content surfaced too.

/// Quality/cost preset for review_commit + review_diff.
///
/// Resolution order: per-call `preset` arg > $LAMU_PRESET env > Fast.
/// Individual env flags (LAMU_CRITIC_PASS, LAMU_ENSEMBLE_REVIEW,
/// LAMU_TEST_PREFLIGHT, LAMU_TWO_STAGE_REVIEW) override the preset's
/// defaults — useful for fine-grained tuning without writing a new
/// preset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Preset {
    /// Cache-stable single Pro pass + central FP-list + activity-log
    /// cache lock + structural cross-ref marker. ~$0.003/review.
    /// 2-3 findings typical. Routine commit gating.
    Fast,
    /// Fast + critic-role parallel pass + multi-model ensemble +
    /// test pre-flight. ~$0.013/review (~4× Fast). 5-7 findings.
    /// Pre-merge / pre-release / security-sensitive.
    Max,
}

impl Preset {
    pub(crate) fn resolve(args: &Value) -> Self {
        let from_arg = args["preset"].as_str();
        let from_env = std::env::var("LAMU_PRESET").ok();
        let raw = from_arg.or(from_env.as_deref()).unwrap_or("max");
        match raw.to_ascii_lowercase().as_str() {
            "fast" => Preset::Fast,
            _ => Preset::Max,
        }
    }

    /// Critic-role parallel pass on. Env override wins.
    pub(crate) fn critic_pass_on(self) -> bool {
        env_flag_on("LAMU_CRITIC_PASS").unwrap_or(matches!(self, Preset::Max))
    }

    /// Multi-model ensemble on. Env override wins.
    pub(crate) fn ensemble_on(self) -> bool {
        env_flag_on("LAMU_ENSEMBLE_REVIEW").unwrap_or(matches!(self, Preset::Max))
    }

    /// Test pre-flight on. Env override wins.
    pub(crate) fn test_preflight_on(self) -> bool {
        env_flag_on("LAMU_TEST_PREFLIGHT").unwrap_or(matches!(self, Preset::Max))
    }

    /// Two-stage review (Flash candidates → Pro verify). Off in both
    /// presets — kept available via explicit LAMU_TWO_STAGE_REVIEW=1
    /// for callers who want it (proven cost regression in our setup).
    pub(crate) fn two_stage_on(self) -> bool {
        env_flag_on("LAMU_TWO_STAGE_REVIEW").unwrap_or(false)
    }
}

/// `Some(true/false)` from a truthy env var, `None` when unset — lets the
/// caller fall back to a preset default. Truthiness goes through the one
/// [`truthy_env_flag`] primitive ("1"/"true", case-insensitive, trimmed) so
/// every flag (here + auto_context's LAMU_TEST_PREFLIGHT) parses identically.
pub(crate) fn env_flag_on(name: &str) -> Option<bool> {
    std::env::var(name).ok().map(|v| truthy_env_flag(Some(&v)))
}

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
    let preset = Preset::resolve(&args);
    tracing::debug!(?preset, "review_commit preset resolved");

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

    // V5 improvement I: implicit warmup when this MCP server hasn't
    // seen a review_commit call in > 30 min. Server-side prompt cache
    // (DeepSeek) likely expired. Fire a 1-token warmup call against
    // the same central+plan prefix so the real call hits cache.
    if should_implicit_warmup() {
        tracing::info!(target: "lamu_bench", "auto-warmup triggered (idle > 30 min)");
        let warmup_args = json!({
            "plan_file": plan_arg.unwrap_or(""),
            "repo": repo,
        });
        let _ = handle_warmup(warmup_args).await;
    }

    // When auto_context=true, run the diff-derived context assembler
    // (changed-files + tree-sitter symbols + ripgrep callers) and
    // prepend it to the caller-supplied tactical blob. Caller-supplied
    // sits at the end so it stays closer to the role prompt.
    //
    // V5 improvement F: skip auto_context for trivial commits — small
    // diffs already fit in the diff prompt itself, expanding to full
    // file bodies is pure overhead. Threshold: < 50 added lines + < 3
    // changed files. Above either threshold we engage as normal.
    let auto_blob = if auto {
        let trivial = is_trivial_diff(commit, std::path::Path::new(repo));
        if trivial {
            tracing::debug!("auto_context: skipped (trivial diff under threshold)");
            String::new()
        } else {
            let opts = crate::auto_context::AutoContextOpts {
                test_preflight: preset.test_preflight_on(),
            };
            match crate::auto_context::assemble_auto_context_with_opts(
                commit,
                std::path::Path::new(repo),
                opts,
            ) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!("auto_context: assemble failed: {}", e);
                    String::new()
                }
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
        // V5 H: dedupe overlapping content. When auto_context's
        // changed-files section already includes the same lines that
        // appear in caller-supplied `context`, drop the duplicate
        // chunk to keep the prefix lean. Hash by 6-line shingle.
        dedupe_tactical(&auto_blob, raw_tactical_arg)
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

    // V6 L: two-stage review when auto_context engages and the
    // env knob doesn't disable. Stage 1 = Flash candidate scan,
    // Stage 2 = Pro verify. Cheaper than V5's single-shot Pro on
    // the same input.
    // V6 L is OPT-IN via LAMU_TWO_STAGE_REVIEW=1. Bench showed it's
    // a regression in our setup: full prefix replays in both Flash
    // stage 1 and Pro stage 2, paying ~2× the prefix cost. Kept
    // available for callers who want Flash's tighter candidate
    // shortlist as a focusing aid (e.g. when Pro is the bottleneck
    // not the prefix), but default off.
    let review = if preset.ensemble_on() {
        ensemble_review(&prompt, &system).await
    } else if preset.two_stage_on() {
        two_stage_review(&prompt, &system).await
    } else {
        let review_args = json!({
            "model": "mimo-v2.5-pro",
            "prompt": prompt,
            "system": system,
            "max_tokens": 8192,
            "temperature": 0.2,
            "include_reasoning": false,
        });
        handle_cloud_query(review_args).await
    };

    // V4 Batch 2 + V5 D: two-direction self-reflection.
    //   - When draft has findings, run verify_findings_via_flash to
    //     drop those matching a known-FP pattern. Cuts residual FPs.
    //   - When draft says PASS, run pass_double_check_via_flash to
    //     scan the diff a second time for issues the reviewer might have
    //     skipped. Cuts false negatives (silent merge of broken code).
    // V6 N: skip Flash 2nd-pass on trivial diffs. The double-check
    // primarily catches issues in substantive changes; small commits
    // are too small to hide much. Threshold matches F's auto_context
    // skip — same shape, same call.
    let review = if auto && !review.starts_with("error:") {
        let trivial = is_trivial_diff(commit, std::path::Path::new(repo));
        if trivial {
            tracing::debug!("v6 N: skipping Flash 2nd-pass on trivial diff");
            review
        } else if is_clean_pass(&review) {
            // PASS path — scan for false negatives
            let diff = git_show_diff_or_empty(commit, std::path::Path::new(repo));
            pass_double_check_via_flash(&review, &diff).await
        } else {
            // Has findings — drop FPs
            verify_findings_via_flash(&review).await
        }
    } else {
        review
    };

    // V6 Q: critic pass — second-order regression hunting.
    // Default ON for Max preset, OFF for Fast.
    let review = if preset.critic_pass_on() && !review.starts_with("error:") {
        let extra = critic_pass(&prompt, &system).await;
        if extra.is_empty() {
            review
        } else {
            format!("{}{}", review, extra)
        }
    } else {
        review
    };

    format!("=== Review of {} (MiMo V2.5 Pro) ===\n\n{}", commit, review)
}

/// Self-reflection: feed the reviewer draft + central FP-list to a
/// cheap Flash call asking it to drop findings that match a known
/// false-positive pattern. Returns either a filtered version of the
/// draft, or the draft unchanged if Flash refuses / errors / the
/// caller has no API key.
///
/// Best-effort: any failure path returns the original draft so the
/// reviewer's signal is never lost — only filtered.
/// V6 Q: critic role parallel pass. Same diff fed to a "Critic" Flash
/// call asking "what could this BREAK? what regressions might land?"
/// The reviewer's normal pass finds bugs in the diff itself; the
/// critic looks for second-order effects on callers / contracts /
/// future use. Findings are appended to the main review, marked as
/// CRITIC-source so the reader can weigh them differently.
///
/// Opt-in via LAMU_CRITIC_PASS=1. Cost: ~$0.0003 per review.
async fn critic_pass(prompt: &str, system: &str) -> String {
    let critic_system = format!(
        "You are a CRITIC reviewing code changes for second-order effects. Focus on: regressions to existing callers, contract changes that downstream code may not expect, performance cliffs, edge cases the author probably didn't consider. Don't repeat findings about the diff itself — assume the primary reviewer covered those. Format: numbered list of CONCERN findings, severity tag, file:line, problem, mitigation. If you genuinely see no second-order risk, return ONLY: NO_CRITIC_FINDINGS\n\n{}",
        system
    );
    let critic_prompt = format!(
        "{}\n\n---\n\nWhat could this BREAK?",
        prompt
    );
    let args = json!({
        "model": "mimo-v2.5",
        "prompt": critic_prompt,
        "system": critic_system,
        "max_tokens": 1024,
        "temperature": 0.3,
        "include_reasoning": false,
    });
    let resp = handle_cloud_query(args).await;
    let trim = resp.trim();
    if resp.starts_with("error:") || trim.is_empty() || trim.starts_with("NO_CRITIC_FINDINGS") {
        return String::new();
    }
    format!(
        "\n\n---\n\n## Critic-pass findings (V6 Q — what could this break?)\n\n{}",
        resp
    )
}

/// V6 R: multi-model ensemble review. Pro + Flash run a full review
/// in parallel via tokio::join. Findings are merged: Pro's review
/// drives the verdict, but Flash findings that don't appear in Pro's
/// review (jaccard < 0.4 on line-level similarity) are appended as
/// ENSEMBLE-source. Two independent passes catch what one misses.
///
/// Note: ANTHROPIC_API_KEY is preferred for cross-provider diversity
/// (claude-opus-4-7 second reviewer), but absent that we fall back to
/// V4 Flash. Same wire format, same prompt cache.
///
/// Opt-in via LAMU_ENSEMBLE_REVIEW=1. Cost: ~+$0.0004 per review.
async fn ensemble_review(prompt: &str, system: &str) -> String {
    let pro_args = json!({
        "model": "mimo-v2.5-pro",
        "prompt": prompt, "system": system,
        "max_tokens": 8192, "temperature": 0.2,
        "include_reasoning": false,
    });
    let second_model = if std::env::var("ANTHROPIC_API_KEY").is_ok() {
        "claude-opus-4-7"
    } else {
        "deepseek-v4-flash"
    };
    let second_args = json!({
        "model": second_model,
        "prompt": prompt, "system": system,
        "max_tokens": 4096, "temperature": 0.3,
        "include_reasoning": false,
    });
    let (pro, second) = tokio::join!(
        handle_cloud_query(pro_args),
        handle_cloud_query(second_args),
    );
    if pro.starts_with("error:") {
        return pro;
    }
    if second.starts_with("error:") {
        tracing::debug!("v6 R: second model failed, returning Pro alone");
        return pro;
    }
    let added = merge_unique_findings(&pro, &second);
    if added.is_empty() {
        pro
    } else {
        format!(
            "{}\n\n---\n\n## Ensemble findings (V6 R — second reviewer: {})\n\n{}",
            pro, second_model, added
        )
    }
}

/// Extract numbered findings from a review string. Returns lines that
/// look like `1. **BUG** ...`, `2. STYLE ...`, etc.
fn extract_findings(review: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current: Option<String> = None;
    for line in review.lines() {
        let trimmed = line.trim_start();
        let starts_finding = trimmed.chars().next().is_some_and(|c| c.is_ascii_digit())
            && trimmed.contains('.')
            && trimmed.len() > 3;
        if starts_finding {
            if let Some(prev) = current.take() {
                out.push(prev);
            }
            current = Some(line.to_string());
        } else if let Some(buf) = current.as_mut() {
            buf.push('\n');
            buf.push_str(line);
        }
    }
    if let Some(prev) = current.take() {
        out.push(prev);
    }
    out
}

/// Tokenize a finding into lowercase word set for jaccard similarity.
fn finding_tokens(finding: &str) -> std::collections::HashSet<String> {
    finding
        .split(|c: char| !c.is_alphanumeric() && c != '_' && c != ':')
        .filter(|w| w.len() > 2)
        .map(|w| w.to_ascii_lowercase())
        .collect()
}

fn jaccard(a: &std::collections::HashSet<String>, b: &std::collections::HashSet<String>) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let inter = a.intersection(b).count() as f32;
    let union = a.union(b).count() as f32;
    inter / union
}

/// Return findings from `extra` that don't overlap (jaccard >= 0.4)
/// with any finding from `primary`. Concatenated as a numbered list.
fn merge_unique_findings(primary: &str, extra: &str) -> String {
    let primary_findings = extract_findings(primary);
    let extra_findings = extract_findings(extra);
    let primary_token_sets: Vec<_> = primary_findings.iter().map(|f| finding_tokens(f)).collect();
    let mut new_findings = Vec::new();
    for f in extra_findings {
        let toks = finding_tokens(&f);
        let dup = primary_token_sets.iter().any(|p| jaccard(&toks, p) >= 0.4);
        if !dup {
            new_findings.push(f);
        }
    }
    new_findings.join("\n\n")
}

/// V6 L: two-stage review. Stage 1 (Flash) scans the diff and emits
/// a candidate-issue shortlist. Stage 2 (Pro) reads the same prompt
/// + Flash's shortlist and produces the final review with verdict.
/// Pro's reasoning gets directed at evaluating candidates instead of
/// re-discovering everything from scratch — typically ~40% fewer
/// reasoning tokens.
async fn two_stage_review(prompt: &str, system: &str) -> String {
    // Stage 1: Flash candidate scan. Same system prompt, but the
    // user prompt asks for a shortlist not a verdict.
    let stage1_prompt = format!(
        "{}\n\n---\n\nFor STAGE 1: list candidate issues you'd flag — one bullet each, severity tag + file:line + 1-line description. NO verdict line. NO recommend. Just candidates. If you don't see anything substantive, return: NO_CANDIDATES",
        prompt
    );
    let stage1_args = json!({
        "model": "mimo-v2.5",
        "prompt": stage1_prompt,
        "system": system,
        "max_tokens": 1024,
        "temperature": 0.1,
        "include_reasoning": false,
    });
    let candidates = handle_cloud_query(stage1_args).await;
    let candidates_trim = candidates.trim();
    let candidates_short = candidates_trim.starts_with("NO_CANDIDATES")
        || candidates_trim.starts_with("error:")
        || candidates_trim.is_empty();

    // Stage 2: Pro verifies. If Flash had no candidates, ask Pro to
    // confirm; else ask Pro to grade Flash's shortlist + add anything
    // Flash missed.
    let stage2_prompt = if candidates_short {
        format!(
            "{}\n\n---\n\nA fast first-pass scan found NO candidate issues. Verify this is correct. If you agree, return PASS + a one-line justification. If you disagree, list the missed findings.",
            prompt
        )
    } else {
        format!(
            "{}\n\n---\n\nA fast first-pass scan flagged these candidates:\n\n{}\n\n---\n\nAct as the verifier: for each candidate, KEEP if real (severity, file:line, fix), DROP if FP (cite the FP pattern from the policy). Add any genuine findings the first pass missed. End with the verdict line + recommend.",
            prompt, candidates
        )
    };
    let stage2_args = json!({
        "model": "mimo-v2.5-pro",
        "prompt": stage2_prompt,
        "system": system,
        "max_tokens": 8192,
        "temperature": 0.2,
        "include_reasoning": false,
    });
    handle_cloud_query(stage2_args).await
}

/// V5 D: when the reviewer returns PASS, run a quick Flash pass that
/// re-reads the diff with the central FP-list as system prompt and
/// asks "did the reviewer miss anything obvious?". Returns the
/// original PASS draft unchanged if Flash also says PASS; otherwise
/// upgrades to PASS WITH NITS / NEEDS CHANGES with Flash's findings
/// appended. Cost: ~$0.0002 per review.
async fn pass_double_check_via_flash(draft: &str, diff: &str) -> String {
    if diff.is_empty() {
        return draft.to_string();
    }
    let central = include_str!("../assets/central_review_policy.md");
    let system = format!(
        "You are a second-pass reviewer. The first pass returned PASS. Your job is to scan the diff for issues the first pass might have missed — security, correctness, edge cases, race conditions, missing error handling. Apply the same FP rules below. If you genuinely find no issues, return ONLY the single line: SECOND_PASS_OK\n\n{}",
        central
    );
    let prompt = format!(
        "Diff:\n\n```\n{}\n```\n\nFirst-pass verdict: PASS. Did the first pass miss anything? If yes, list the missed findings (severity tag, file:line, problem, fix). If no genuine missed findings, return SECOND_PASS_OK on a single line.",
        diff
    );
    let args = json!({
        "model": "mimo-v2.5",
        "prompt": prompt,
        "system": system,
        "max_tokens": 2048,
        "temperature": 0.1,
        "include_reasoning": false,
    });
    let second = handle_cloud_query(args).await;
    let second_trim = second.trim();
    if second.starts_with("error:") || second_trim.is_empty() || second_trim.starts_with("SECOND_PASS_OK") {
        return draft.to_string();
    }
    // Flash found something — append to draft, upgrade verdict.
    format!(
        "{}\n\n---\n\n## Second-pass findings (V5 D — Flash, after reviewer PASS)\n\n{}",
        draft, second
    )
}

/// True when the review's verdict is a clean PASS — no findings to
/// filter. Routes the Flash second pass: clean PASS → scan for false
/// NEGATIVES (pass_double_check_via_flash); anything else (PASS WITH
/// NITS / NEEDS CHANGES / REJECT) → FP-drop filter (verify_findings).
///
/// Robust to the documented verdict vocabulary + a "Verdict:" label and
/// markdown "**PASS**" bold, unlike the prior exact "\nPASS\n" substring
/// check which misrouted "PASS WITH NITS" into the false-negative branch.
fn is_clean_pass(review: &str) -> bool {
    review.lines().any(|line| {
        let v = line.trim();
        let v = v.strip_prefix("Verdict:").map(str::trim).unwrap_or(v);
        let v = v.trim_matches('*').trim(); // strip markdown bold **PASS**
        match v.strip_prefix("PASS") {
            // "PASS" / "PASS." / "PASS — …" / "PASS: …" → clean.
            // "PASS WITH NITS" → rest is " WITH …" → not clean.
            // "PASSED" → rest "ED" → not clean.
            Some(rest) => {
                let r = rest.trim_start();
                r.is_empty()
                    || r.starts_with('.')
                    || r.starts_with(':')
                    || r.starts_with('—')
                    || r.starts_with('-')
            }
            None => false,
        }
    })
}

/// Text-only trivial-diff heuristic for review_diff, which (unlike
/// review_commit) has no commit SHA to run `git show --shortstat`
/// against. Counts changed files via `diff --git` headers and added
/// lines (`+` excluding the `+++` file header). Same thresholds as
/// is_trivial_diff — small diffs skip the Flash second pass since the
/// double-check mostly catches issues in substantive changes. A raw
/// pasted chunk without `diff --git` headers has files=0, so only the
/// added-line gate applies.
fn is_trivial_diff_text(diff: &str) -> bool {
    const MAX_LINES: usize = 50;
    const MAX_FILES: usize = 3;
    let mut added = 0usize;
    let mut files = 0usize;
    for line in diff.lines() {
        if line.starts_with("diff --git") {
            files += 1;
        } else if line.starts_with('+') && !line.starts_with("+++") {
            added += 1;
        }
    }
    added <= MAX_LINES && files <= MAX_FILES
}

fn git_show_diff_or_empty(commit: &str, repo: &std::path::Path) -> String {
    let out = std::process::Command::new("git")
        .current_dir(repo)
        .args(["show", "--patch", "--no-color", commit])
        .output();
    match out {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).to_string(),
        _ => String::new(),
    }
}

async fn verify_findings_via_flash(draft: &str) -> String {
    // Skip if draft says PASS — nothing to filter.
    if draft.contains("PASS\n") || draft.starts_with("PASS\n") || draft.contains("PASS\nNo ") {
        return draft.to_string();
    }
    let central = include_str!("../assets/central_review_policy.md");
    let system = format!(
        "You are filtering a code review draft. Drop findings that match a documented false-positive pattern. KEEP all real findings — when in doubt, keep.\n\n{}",
        central
    );
    let prompt = format!(
        "Here is a draft review. For each numbered finding, decide: KEEP if it identifies a real bug, or DROP if it matches a known FP pattern (see system prompt). Return the cleaned-up review verbatim with FP findings removed and the verdict line updated if needed (e.g. NEEDS CHANGES → PASS WITH NITS if all real findings dropped).\n\nDRAFT:\n\n{}",
        draft
    );
    let args = json!({
        "model": "mimo-v2.5",
        "prompt": prompt,
        "system": system,
        "max_tokens": 4096,
        "temperature": 0.0,
        "include_reasoning": false,
    });
    let filtered = handle_cloud_query(args).await;
    if filtered.starts_with("error:") || filtered.is_empty() {
        return draft.to_string();
    }
    filtered
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
    let preset = Preset::resolve(&args);
    tracing::debug!(?preset, "review_diff preset resolved");

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

    let review = if preset.ensemble_on() {
        ensemble_review(&prompt, &system).await
    } else if preset.two_stage_on() {
        two_stage_review(&prompt, &system).await
    } else {
        let review_args = json!({
            "model": "mimo-v2.5-pro",
            "prompt": prompt,
            "system": system,
            "max_tokens": 8192,
            "temperature": 0.2,
            "include_reasoning": false,
        });
        handle_cloud_query(review_args).await
    };

    // Verify-before-fix self-reflection — same two-direction pass as
    // review_commit (audit: review_diff previously had NO FP filter
    // despite tools.rs advertising "same reviewer policy"). review_diff
    // carries the diff text directly, so both directions work: a clean
    // PASS scans for false negatives, findings get the FP-drop filter.
    // Skip on trivial diffs (matches review_commit's cost optimization).
    let review = if !review.starts_with("error:") && !is_trivial_diff_text(&diff) {
        if is_clean_pass(&review) {
            pass_double_check_via_flash(&review, &diff).await
        } else {
            verify_findings_via_flash(&review).await
        }
    } else {
        review
    };

    let review = if preset.critic_pass_on() && !review.starts_with("error:") {
        let extra = critic_pass(&prompt, &system).await;
        if extra.is_empty() {
            review
        } else {
            format!("{}{}", review, extra)
        }
    } else {
        review
    };

    format!("=== Diff review (MiMo V2.5 Pro) ===\n\n{}", review)
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
    fn truthy_env_flag_pure_truth_table() {
        // ON only for "1"/"true" (case-insensitive, trimmed).
        assert!(truthy_env_flag(Some("1")));
        assert!(truthy_env_flag(Some("true")));
        assert!(truthy_env_flag(Some("TRUE")));
        assert!(truthy_env_flag(Some("True")));
        assert!(truthy_env_flag(Some("  true  ")));
        // OFF for unset and any other value.
        assert!(!truthy_env_flag(None));
        assert!(!truthy_env_flag(Some("")));
        assert!(!truthy_env_flag(Some("0")));
        assert!(!truthy_env_flag(Some("false")));
        assert!(!truthy_env_flag(Some("yes")));
        assert!(!truthy_env_flag(Some("2")));
    }

    #[test]
    fn autocapture_enabled_default_off_when_unset() {
        // Rust 2024 makes std::env::set_var unsafe (env mutation races
        // with concurrent test threads), so we don't mutate the env here.
        // The ON/value branches are exhaustively covered by the pure
        // `truthy_env_flag` truth-table test above. This asserts the
        // load-bearing default: the test harness runs WITHOUT
        // LAMU_AUTOCAPTURE set, so the real env-reading wrapper reports
        // OFF — proving the autocapture spawn is unreachable by default.
        if std::env::var("LAMU_AUTOCAPTURE").is_err() {
            assert!(!autocapture_enabled(), "unset env → autocapture OFF");
        }
    }

    #[test]
    fn coerce_ephemeral_arg_accepts_bool_and_string() {
        use serde_json::json;
        // JSON bool both ways.
        assert_eq!(coerce_ephemeral_arg(&json!(true)), Some(true));
        assert_eq!(coerce_ephemeral_arg(&json!(false)), Some(false));
        // String form must engage (the privacy footgun this guards).
        assert_eq!(coerce_ephemeral_arg(&json!("true")), Some(true));
        assert_eq!(coerce_ephemeral_arg(&json!("1")), Some(true));
        assert_eq!(coerce_ephemeral_arg(&json!("false")), Some(false));
        assert_eq!(coerce_ephemeral_arg(&json!("nonsense")), Some(false));
        // Absent / wrong-type → None so the env fallback applies.
        assert_eq!(coerce_ephemeral_arg(&json!(null)), None);
        assert_eq!(coerce_ephemeral_arg(&json!(5)), None);
    }

    #[test]
    fn ephemeral_per_call_arg_truth_table() {
        // The per-call arg drives ephemeral regardless of env (the test
        // harness runs without LAMU_EPHEMERAL, so arg is the only signal).
        if std::env::var("LAMU_EPHEMERAL").is_err() {
            assert!(ephemeral_enabled(Some(true)), "arg true → ephemeral");
            assert!(!ephemeral_enabled(Some(false)), "arg false + no env → not ephemeral");
            assert!(!ephemeral_enabled(None), "unset arg + no env → not ephemeral (default)");
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
    fn is_clean_pass_recognizes_clean_verdicts() {
        assert!(is_clean_pass("PASS\n\nRecommend: ship"));
        assert!(is_clean_pass("**PASS**"));
        assert!(is_clean_pass("Verdict: PASS — looks good"));
        assert!(is_clean_pass("Verdict: PASS"));
        assert!(is_clean_pass("preamble line\nPASS.\nRecommend: ship"));
        assert!(is_clean_pass("PASS: clean, nothing to flag"));
    }

    #[test]
    fn is_clean_pass_rejects_findings_and_non_pass() {
        // Has nits/findings — must route to the FP-drop filter, not the
        // false-negative double-check.
        assert!(!is_clean_pass("PASS WITH NITS\n\n1. STYLE foo"));
        assert!(!is_clean_pass("**PASS WITH NITS**"));
        assert!(!is_clean_pass("Verdict: PASS WITH NITS"));
        assert!(!is_clean_pass("NEEDS CHANGES\n\n1. BUG bar"));
        assert!(!is_clean_pass("REJECT"));
        assert!(!is_clean_pass("PASSED the build"));
        assert!(!is_clean_pass("no verdict at all"));
    }

    #[test]
    fn is_trivial_diff_text_thresholds() {
        // Small raw chunk (no diff --git headers) → trivial.
        assert!(is_trivial_diff_text("+one added line\n-one removed"));
        // > 50 added lines → not trivial.
        let big: String = (0..60).map(|i| format!("+line {i}\n")).collect();
        assert!(!is_trivial_diff_text(&big));
        // 4 changed files → not trivial (even if few lines).
        let many_files = "diff --git a/1 b/1\n+x\ndiff --git a/2 b/2\n+x\n\
                          diff --git a/3 b/3\n+x\ndiff --git a/4 b/4\n+x\n";
        assert!(!is_trivial_diff_text(many_files));
        // +++ header is NOT counted as an added line.
        assert!(is_trivial_diff_text("diff --git a/f b/f\n+++ b/f\n+just one"));
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
