//! `lamu cloud sync` — keep the cloud-model catalog current automatically
//! (ADR 0019). Two sources, merged into `cloud-models.yaml` non-destructively:
//!
//!   1. **OpenRouter** `/api/v1/models` (auth-free) — the maintained
//!      cross-provider catalog (every provider, with context + the latest
//!      models). These become `openrouter`-routed entries, callable via the
//!      existing OpenRouter path (ADR 0007).
//!   2. **Per-provider ping** — each direct provider already configured in
//!      cloud-models.yaml (with a present key) is asked for its own
//!      `/v1/models`, the authoritative list of what *your* key can call.
//!
//! Merge is preservation-first: an existing entry keeps every hand-set field
//! (name, model_id, base_url, api_key_env, chat_path, notes, quota); only an
//! unset `context_max` is filled. New models are appended. Nothing is deleted.

use anyhow::Result;
use lamu_providers::cloud_config::CloudModel;
use serde::Deserialize;
use std::collections::HashSet;

const OPENROUTER_MODELS_URL: &str = "https://openrouter.ai/api/v1/models";

#[derive(Deserialize)]
struct RawModel {
    id: String,
    #[serde(default)]
    context_length: Option<u32>,
}

/// Parse a `{ "data": [ ... ] }` model-list body resiliently: one malformed
/// entry (missing/odd `id`, etc.) is skipped, not fatal to the whole pull.
/// Tolerates the array at `data` (OpenAI/OpenRouter/Anthropic all use it).
fn parse_models(body: &serde_json::Value) -> Vec<RawModel> {
    body.get("data")
        .and_then(|d| d.as_array())
        .map(|arr| arr.iter().filter_map(|v| serde_json::from_value::<RawModel>(v.clone()).ok()).collect())
        .unwrap_or_default()
}

/// Pull the OpenRouter cross-provider catalog (no auth). Each id is
/// `provider/name`; map to an openrouter-routed CloudModel.
pub async fn fetch_openrouter(client: &reqwest::Client) -> Result<Vec<CloudModel>> {
    let resp = client.get(OPENROUTER_MODELS_URL).send().await?;
    if !resp.status().is_success() {
        anyhow::bail!("OpenRouter /models returned {}", resp.status());
    }
    let body: serde_json::Value = resp.json().await?;
    Ok(parse_models(&body)
        .into_iter()
        .map(|m| {
            let (provider, name) = m.id.split_once('/').unwrap_or(("openrouter", m.id.as_str()));
            CloudModel {
                name: name.to_string(),
                provider: provider.to_string(),
                model_id: Some(m.id.clone()), // openrouter wants the full provider/name id
                context_max: m.context_length.unwrap_or(0),
                notes: "via openrouter (auto-synced)".to_string(),
                quota: Default::default(),
                api_key_env: Some("OPENROUTER_API_KEY".to_string()),
                base_url: Some("https://openrouter.ai/api/v1".to_string()),
                chat_path: None,
            }
        })
        .collect())
}

/// Build the `/v1/models` URL for a provider base (tolerates a trailing `/v1`).
fn models_url(base_url: &str) -> String {
    let b = base_url.trim_end_matches('/');
    let b = b.strip_suffix("/v1").unwrap_or(b);
    format!("{b}/v1/models")
}

/// Ask one direct provider for its own model list (authoritative for the
/// configured key). Anthropic uses `x-api-key` + a version header; everything
/// else is OpenAI-compat Bearer. Returns directly-callable CloudModels.
pub async fn ping_provider(
    client: &reqwest::Client,
    provider: &str,
    base_url: &str,
    api_key_env: &str,
    key: &str,
) -> Result<Vec<CloudModel>> {
    let url = models_url(base_url);
    let mut req = client.get(&url);
    if provider == "anthropic" {
        req = req.header("x-api-key", key).header("anthropic-version", "2023-06-01");
    } else {
        req = req.bearer_auth(key);
    }
    let resp = req.send().await?;
    if !resp.status().is_success() {
        anyhow::bail!("{provider} {url} returned {}", resp.status());
    }
    let body: serde_json::Value = resp.json().await?;
    Ok(parse_models(&body)
        .into_iter()
        .map(|m| CloudModel {
            name: m.id.clone(),
            provider: provider.to_string(),
            model_id: Some(m.id.clone()), // direct call uses the bare provider id
            context_max: m.context_length.unwrap_or(0),
            notes: format!("via {provider} /v1/models (auto-synced)"),
            quota: Default::default(),
            api_key_env: Some(api_key_env.to_string()),
            base_url: Some(base_url.to_string()),
            chat_path: None,
        })
        .collect())
}

/// Distinct direct providers to ping: each unique (provider, base_url,
/// api_key_env) already in the catalog, excluding `openrouter` (the pull
/// already covers it) and any without a base_url/key.
pub fn distinct_providers(existing: &[CloudModel]) -> Vec<(String, String, String)> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for m in existing {
        if m.provider == "openrouter" {
            continue;
        }
        let (Some(base), Some(key_env)) = (m.base_url.as_ref(), m.api_key_env.as_ref()) else {
            continue;
        };
        let tuple = (m.provider.clone(), base.clone(), key_env.clone());
        if seen.insert(tuple.clone()) {
            out.push(tuple);
        }
    }
    out
}

/// Merge `discovered` into `existing`, preservation-first. Returns
/// (merged, added, updated). Keyed by `full_id()`.
pub fn merge(existing: Vec<CloudModel>, discovered: Vec<CloudModel>) -> (Vec<CloudModel>, usize, usize) {
    let mut merged = existing;
    let mut index: std::collections::HashMap<String, usize> =
        merged.iter().enumerate().map(|(i, m)| (m.full_id(), i)).collect();
    let (mut added, mut updated) = (0usize, 0usize);
    for d in discovered {
        match index.get(&d.full_id()) {
            Some(&i) => {
                // Preserve every hand-set field; only fill an unset context.
                if merged[i].context_max == 0 && d.context_max > 0 {
                    merged[i].context_max = d.context_max;
                    updated += 1;
                }
            }
            None => {
                index.insert(d.full_id(), merged.len());
                merged.push(d);
                added += 1;
            }
        }
    }
    (merged, added, updated)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cm(name: &str, provider: &str, model_id: Option<&str>, ctx: u32, notes: &str) -> CloudModel {
        CloudModel {
            name: name.into(),
            provider: provider.into(),
            model_id: model_id.map(String::from),
            context_max: ctx,
            notes: notes.into(),
            quota: Default::default(),
            api_key_env: None,
            base_url: None,
            chat_path: None,
        }
    }

    #[test]
    fn merge_preserves_existing_and_adds_new() {
        // Existing hand-set entry (custom notes, ctx set) + a new model.
        let existing = vec![cm("claude-opus-4.8", "anthropic", Some("anthropic/claude-opus-4.8"), 200000, "my notes")];
        let discovered = vec![
            // same id, catalog version with different notes + ctx → must NOT clobber
            cm("claude-opus-4.8", "anthropic", Some("anthropic/claude-opus-4.8"), 999, "via openrouter"),
            // brand-new model → added
            cm("gpt-5.4", "openai", Some("openai/gpt-5.4"), 400000, "via openrouter"),
        ];
        let (merged, added, updated) = merge(existing, discovered);
        assert_eq!(added, 1, "only the new model is added");
        assert_eq!(updated, 0, "existing ctx already set → not touched");
        assert_eq!(merged.len(), 2);
        let opus = merged.iter().find(|m| m.name == "claude-opus-4.8").unwrap();
        assert_eq!(opus.notes, "my notes", "hand-set notes preserved");
        assert_eq!(opus.context_max, 200000, "hand-set context preserved");
    }

    #[test]
    fn merge_fills_unset_context() {
        let existing = vec![cm("m", "p", Some("p/m"), 0, "")]; // ctx unset
        let discovered = vec![cm("m", "p", Some("p/m"), 128000, "catalog")];
        let (merged, added, updated) = merge(existing, discovered);
        assert_eq!(added, 0);
        assert_eq!(updated, 1, "unset context gets filled from the catalog");
        assert_eq!(merged[0].context_max, 128000);
        assert_eq!(merged[0].notes, "", "notes still not clobbered");
    }

    #[test]
    fn models_url_tolerates_v1_suffix() {
        assert_eq!(models_url("https://api.deepseek.com"), "https://api.deepseek.com/v1/models");
        assert_eq!(models_url("https://openrouter.ai/api/v1"), "https://openrouter.ai/api/v1/models");
        assert_eq!(models_url("https://x.com/"), "https://x.com/v1/models");
    }
}
