//! Cloud-model registry — lamu-cli front (seed + wizard + save).
//!
//! The pure data model (`CloudModel`, `QuotaState`, `CloudModelList`)
//! and the read path (`load_or_empty`, `config_path`) live in
//! `lamu-providers::cloud_config` so lamu-mcp can share them without
//! depending on lamu-cli.
//!
//! Path: `$XDG_CONFIG_HOME/lamu/cloud-models.yaml` (defaults to
//! `~/.config/lamu/cloud-models.yaml`). Missing file = a sensible
//! seed list (Anthropic + OpenAI families). Edit the file to add
//! more or to wire your own provider routes.

pub use lamu_providers::{config_path, CloudModel, CloudModelList, QuotaState};

pub fn default_seed() -> Vec<CloudModel> {
    vec![
        // Anthropic — direct API (provider router uses /v1/messages,
        // x-api-key header, native message format).
        CloudModel {
            name: "claude-opus-4-7".into(),
            provider: "anthropic".into(),
            model_id: Some("claude-opus-4-7".into()),
            context_max: 200_000,
            notes: "Anthropic Claude Opus 4.7 (best reasoning)".into(),
            quota: QuotaState::Available,
            api_key_env: Some("ANTHROPIC_API_KEY".into()),
            base_url: Some("https://api.anthropic.com".into()),
            chat_path: None,
        },
        CloudModel {
            name: "claude-sonnet-4-6".into(),
            provider: "anthropic".into(),
            model_id: Some("claude-sonnet-4-6".into()),
            context_max: 200_000,
            notes: "Anthropic Claude Sonnet 4.6 (workhorse)".into(),
            quota: QuotaState::Available,
            api_key_env: Some("ANTHROPIC_API_KEY".into()),
            base_url: Some("https://api.anthropic.com".into()),
            chat_path: None,
        },
        CloudModel {
            name: "claude-haiku-4-5".into(),
            provider: "anthropic".into(),
            model_id: Some("claude-haiku-4-5".into()),
            context_max: 200_000,
            notes: "Anthropic Claude Haiku 4.5 (fast)".into(),
            quota: QuotaState::Available,
            api_key_env: Some("ANTHROPIC_API_KEY".into()),
            base_url: Some("https://api.anthropic.com".into()),
            chat_path: None,
        },
        // GLM 5.1 (Zhipu / Z.AI)
        CloudModel {
            name: "glm-5.1".into(),
            provider: "zhipu".into(),
            model_id: None,
            context_max: 1_000_000,
            notes: "Zhipu GLM 5.1 — open-weights flagship, 1M ctx".into(),
            quota: QuotaState::Available,
            api_key_env: Some("ZHIPU_API_KEY".into()),
            base_url: None,
            chat_path: None,
        },
        // Kimi K2.6 (Moonshot)
        CloudModel {
            name: "kimi-k2.6".into(),
            provider: "moonshot".into(),
            model_id: None,
            context_max: 1_000_000,
            notes: "Moonshot Kimi K2.6 — long-context agentic, ~1M ctx".into(),
            quota: QuotaState::Available,
            api_key_env: Some("MOONSHOT_API_KEY".into()),
            base_url: None,
            chat_path: None,
        },
        // Qwen3 large via Alibaba DashScope (>100B params).
        CloudModel {
            name: "qwen3-235b-a22b-thinking-2507".into(),
            provider: "alibaba".into(),
            model_id: None,
            context_max: 1_000_000,
            notes: "Alibaba Qwen3 235B-A22B Thinking (DashScope), 1M ctx".into(),
            quota: QuotaState::Available,
            api_key_env: Some("DASHSCOPE_API_KEY".into()),
            base_url: None,
            chat_path: None,
        },
        CloudModel {
            name: "qwen3-max".into(),
            provider: "alibaba".into(),
            model_id: None,
            context_max: 1_000_000,
            notes: "Alibaba Qwen3-Max (DashScope, >1T params)".into(),
            quota: QuotaState::Available,
            api_key_env: Some("DASHSCOPE_API_KEY".into()),
            base_url: None,
            chat_path: None,
        },
        // DeepSeek V4 — direct API (no Bifrost needed)
        // Both models: 128K ctx, chat+code+reasoning. Flash = fast/cheap
        // (~$0.27/$1.10 per M tok in/out); Pro = higher quality thinking.
        CloudModel {
            name: "deepseek-v4-flash".into(),
            provider: "deepseek".into(),
            model_id: Some("deepseek-v4-flash".into()),
            context_max: 1_000_000,
            notes: "DeepSeek V4 Flash — 1M ctx. 75% promo: cache-hit ~$0.014/M in (71M/$), cache-miss ~$0.07/M in, ~$0.42/M out (2.4M/$).".into(),
            quota: QuotaState::Available,
            api_key_env: Some("DEEPSEEK_API_KEY".into()),
            base_url: Some("https://api.deepseek.com".into()),
            chat_path: None,
        },
        CloudModel {
            name: "deepseek-v4-pro".into(),
            provider: "deepseek".into(),
            model_id: Some("deepseek-v4-pro".into()),
            context_max: 1_000_000,
            notes: "DeepSeek V4 Pro — 1M ctx, deeper thinking. 75% promo: cache-hit ~$0.14/M in, cache-miss ~$0.55/M in, ~$1.03/M out.".into(),
            quota: QuotaState::Available,
            api_key_env: Some("DEEPSEEK_API_KEY".into()),
            base_url: Some("https://api.deepseek.com".into()),
            chat_path: None,
        },
        // Xiaomi MiMo V2.5 — OpenAI-compat endpoint at
        // https://token-plan-sgp.xiaomimimo.com/v1. Anthropic-compat
        // available at /anthropic on the same host.
        CloudModel {
            name: "mimo-v2.5-pro".into(),
            provider: "mimo".into(),
            model_id: Some("mimo-v2.5-pro".into()),
            context_max: 256_000,
            notes: "Xiaomi MiMo V2.5 Pro — flagship reasoning.".into(),
            quota: QuotaState::Available,
            api_key_env: Some("MIMO_API_KEY".into()),
            base_url: Some("https://token-plan-sgp.xiaomimimo.com/v1".into()),
            chat_path: None,
        },
        CloudModel {
            name: "mimo-v2.5".into(),
            provider: "mimo".into(),
            model_id: Some("mimo-v2.5".into()),
            context_max: 256_000,
            notes: "Xiaomi MiMo V2.5 — workhorse chat.".into(),
            quota: QuotaState::Available,
            api_key_env: Some("MIMO_API_KEY".into()),
            base_url: Some("https://token-plan-sgp.xiaomimimo.com/v1".into()),
            chat_path: None,
        },
        CloudModel {
            name: "mimo-v2-pro".into(),
            provider: "mimo".into(),
            model_id: Some("mimo-v2-pro".into()),
            context_max: 256_000,
            notes: "Xiaomi MiMo V2 Pro — prior gen flagship.".into(),
            quota: QuotaState::Available,
            api_key_env: Some("MIMO_API_KEY".into()),
            base_url: Some("https://token-plan-sgp.xiaomimimo.com/v1".into()),
            chat_path: None,
        },
        CloudModel {
            name: "mimo-v2-omni".into(),
            provider: "mimo".into(),
            model_id: Some("mimo-v2-omni".into()),
            context_max: 256_000,
            notes: "Xiaomi MiMo V2 Omni — multimodal.".into(),
            quota: QuotaState::Available,
            api_key_env: Some("MIMO_API_KEY".into()),
            base_url: Some("https://token-plan-sgp.xiaomimimo.com/v1".into()),
            chat_path: None,
        },
        // Anthropic-shaped MiMo mirror — same model, /anthropic
        // endpoint. Useful for routing claude-code (and any other
        // Claude-shaped harness) through MiMo via ANTHROPIC_BASE_URL
        // without the Anthropic SDK noticing the difference.
        CloudModel {
            name: "mimo-v2.5-pro-anthropic".into(),
            provider: "anthropic".into(),
            model_id: Some("mimo-v2.5-pro".into()),
            context_max: 256_000,
            notes: "Xiaomi MiMo V2.5 Pro via Anthropic-compat endpoint.".into(),
            quota: QuotaState::Available,
            api_key_env: Some("MIMO_API_KEY".into()),
            base_url: Some("https://token-plan-sgp.xiaomimimo.com/anthropic".into()),
            chat_path: None,
        },
        CloudModel {
            name: "mimo-v2.5-anthropic".into(),
            provider: "anthropic".into(),
            model_id: Some("mimo-v2.5".into()),
            context_max: 256_000,
            notes: "Xiaomi MiMo V2.5 via Anthropic-compat endpoint.".into(),
            quota: QuotaState::Available,
            api_key_env: Some("MIMO_API_KEY".into()),
            base_url: Some("https://token-plan-sgp.xiaomimimo.com/anthropic".into()),
            chat_path: None,
        },
    ]
}

/// Sensible defaults for a known provider. The wizard pre-fills these
/// so adding a model is mostly hitting Enter to accept defaults.
/// Returns None for unknown / "custom" provider — caller fills every
/// field by hand.
pub fn provider_template(provider: &str) -> Option<CloudModel> {
    let make = |provider: &str, env: &str, ctx: u32, notes: &str| CloudModel {
        name: String::new(),
        provider: provider.to_string(),
        model_id: None,
        context_max: ctx,
        notes: notes.to_string(),
        quota: QuotaState::Available,
        api_key_env: Some(env.to_string()),
        base_url: None,
        chat_path: None,
    };
    match provider {
        "anthropic" => Some(make("anthropic", "ANTHROPIC_API_KEY", 200_000, "Anthropic Claude")),
        "openai" => Some(make("openai", "OPENAI_API_KEY", 200_000, "OpenAI")),
        "zhipu" => Some(make("zhipu", "ZHIPU_API_KEY", 200_000, "Zhipu / Z.AI GLM")),
        "moonshot" => Some(make("moonshot", "MOONSHOT_API_KEY", 256_000, "Moonshot Kimi")),
        "alibaba" => Some(make("alibaba", "DASHSCOPE_API_KEY", 262_144, "Alibaba Qwen via DashScope")),
        "deepseek" => Some(make("deepseek", "DEEPSEEK_API_KEY", 128_000, "DeepSeek")),
        "mistral" => Some(make("mistral", "MISTRAL_API_KEY", 128_000, "Mistral")),
        "openrouter" => Some(make("openrouter", "OPENROUTER_API_KEY", 128_000, "OpenRouter aggregator")),
        "google" => Some(make("google", "GOOGLE_API_KEY", 1_000_000, "Google Gemini")),
        "xai" => Some(make("xai", "XAI_API_KEY", 256_000, "xAI Grok")),
        "mimo" => {
            let mut m = make("mimo", "MIMO_API_KEY", 256_000, "Xiaomi MiMo");
            m.base_url = Some("https://token-plan-sgp.xiaomimimo.com/v1".into());
            Some(m)
        }
        _ => None,
    }
}

pub const KNOWN_PROVIDERS: &[&str] = &[
    "anthropic", "openai", "zhipu", "moonshot", "alibaba",
    "deepseek", "mistral", "openrouter", "google", "xai", "mimo",
];

/// Run an interactive add-cloud-model wizard on plain stdin/stdout.
/// Caller must have torn down the alt-screen first. Returns the new
/// model on confirm, or `None` on cancel / failure.
pub fn add_via_wizard() -> Option<CloudModel> {
    use std::io::{self, Write};

    fn ask(prompt: &str) -> String {
        eprint!("{}", prompt);
        let _ = io::stderr().flush();
        let mut buf = String::new();
        let _ = io::stdin().read_line(&mut buf);
        buf.trim().to_string()
    }

    println!();
    println!("──────────────────────────────────────────────");
    println!(" Add a cloud model");
    println!("──────────────────────────────────────────────");
    println!();
    println!("Known provider presets (autofills env var, ctx, notes):");
    for p in KNOWN_PROVIDERS {
        println!("  • {p}");
    }
    println!("  • custom  (enter all fields by hand)");
    println!();

    let provider_input = ask("Provider [anthropic]: ");
    let provider = if provider_input.is_empty() { "anthropic".to_string() } else { provider_input };

    let mut entry = if provider == "custom" {
        CloudModel {
            name: String::new(),
            provider: ask("Provider key (e.g. some-new-vendor): "),
            model_id: None,
            context_max: 128_000,
            notes: String::new(),
            quota: QuotaState::Available,
            api_key_env: None,
            base_url: None,
            chat_path: None,
        }
    } else {
        match provider_template(&provider) {
            Some(t) => t,
            None => {
                eprintln!("Unknown provider '{provider}'. Use one of the listed presets or 'custom'.");
                return None;
            }
        }
    };

    let name = ask("Display name (e.g. claude-opus-4-7) [required]: ");
    if name.is_empty() {
        eprintln!("Cancelled (empty name).");
        return None;
    }
    entry.name = name;

    let model_id = ask(&format!("Model ID for request body [enter = {}/{} default]: ", entry.provider, entry.name));
    if !model_id.is_empty() {
        entry.model_id = Some(model_id);
    }

    let ctx_input = ask(&format!("Context window in tokens [{}]: ", entry.context_max));
    if let Ok(n) = ctx_input.parse::<u32>() {
        entry.context_max = n;
    }

    let env_default = entry.api_key_env.clone().unwrap_or_default();
    let env_input = ask(&format!("API key env var [{}]: ", if env_default.is_empty() { "(none — Bifrost handles auth)" } else { &env_default }));
    if !env_input.is_empty() {
        entry.api_key_env = Some(env_input);
    } else if env_default.is_empty() {
        entry.api_key_env = None;
    }

    let notes = ask(&format!("Notes [{}]: ", entry.notes));
    if !notes.is_empty() {
        entry.notes = notes;
    }

    println!();
    println!("Will add:");
    println!("  name:        {}", entry.name);
    println!("  provider:    {}", entry.provider);
    println!("  full_id:     {}", entry.full_id());
    println!("  context_max: {}", entry.context_max);
    if let Some(env) = &entry.api_key_env {
        let set = std::env::var(env).is_ok();
        println!("  api_key_env: {} ({})", env, if set { "set ✓" } else { "unset ✗" });
    } else {
        println!("  api_key_env: (none — routed via Bifrost)");
    }
    println!("  notes:       {}", entry.notes);
    println!();

    let confirm = ask("Confirm? [Y/n]: ");
    if confirm.to_lowercase().starts_with('n') {
        eprintln!("Cancelled.");
        return None;
    }

    Some(entry)
}

/// Persist the full model list to disk, preserving the user's existing
/// list shape. Errors swallowed → a write failure leaves the in-memory
/// state untouched and the next `load()` returns the old file.
pub fn save(models: &[CloudModel]) -> std::io::Result<()> {
    let path = config_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let list = CloudModelList { models: models.to_vec() };
    let buf = serde_yaml::to_string(&list)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(&path, buf)
}

/// Load cloud models. Missing file → seed list written to disk for
/// users to edit, then returned. Existing file → consume via the
/// shared `lamu_providers::load_or_empty` parser, then top up with
/// any seed entries the user is missing.
///
/// Phase 5.3 — the seed list is the canonical Rust-side definition.
/// When a new model lands in `default_seed()` (e.g. a new release),
/// existing users get the entry added on next launch. User
/// customizations (notes, quota, model_id overrides) are preserved
/// because we never overwrite an entry whose `name` already lives in
/// the YAML — we only append the seed entries that are absent.
pub fn load() -> Vec<CloudModel> {
    let path = config_path();
    if !path.exists() {
        let seed = default_seed();
        let list = CloudModelList { models: seed.clone() };
        if let Ok(buf) = serde_yaml::to_string(&list) {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let _ = std::fs::write(&path, buf);
        }
        return seed;
    }
    let mut existing = lamu_providers::load_or_empty();
    let merged = merge_seed_into(&mut existing, default_seed());
    if merged {
        // Persist the topped-up list so subsequent reads (TUI + MCP)
        // see the same entries. A failure here just means the seed
        // re-applies on the next launch, which is harmless.
        let _ = save(&existing);
    }
    existing
}

/// Append seed entries whose `name` is missing from `existing`.
/// Returns true when at least one entry was appended (so the caller
/// can persist).
fn merge_seed_into(existing: &mut Vec<CloudModel>, seed: Vec<CloudModel>) -> bool {
    let mut changed = false;
    for s in seed {
        if !existing.iter().any(|e| e.name == s.name) {
            existing.push(s);
            changed = true;
        }
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    // Pure CloudModel/QuotaState behavior tested in lamu-providers.
    // Tests here cover seed + provider_template + on-disk save/load
    // — the lamu-cli-specific surface.

    #[test]
    fn seed_has_well_formed_entries() {
        for m in default_seed() {
            assert!(!m.name.is_empty());
            assert!(!m.provider.is_empty());
            assert!(m.context_max >= 100_000);
        }
    }

    #[test]
    fn seed_drops_old_gpt_entries() {
        for m in default_seed() {
            assert!(!m.name.starts_with("gpt-"), "stale entry survived: {}", m.name);
        }
    }

    #[test]
    fn seed_includes_glm_kimi_qwen_large() {
        let names: Vec<String> = default_seed().into_iter().map(|m| m.name).collect();
        assert!(names.iter().any(|n| n.contains("glm-5.1")));
        assert!(names.iter().any(|n| n.contains("kimi-k2.6")));
        assert!(names.iter().any(|n| n.contains("235b")));
        assert!(names.iter().any(|n| n.contains("qwen3-max")));
        assert!(names.iter().any(|n| n.contains("deepseek-v4")));
    }

    #[test]
    fn yaml_round_trip() {
        let list = CloudModelList { models: default_seed() };
        let s = serde_yaml::to_string(&list).unwrap();
        let back: CloudModelList = serde_yaml::from_str(&s).unwrap();
        assert_eq!(back.models.len(), default_seed().len());
    }

    #[test]
    fn known_providers_all_have_templates() {
        for p in KNOWN_PROVIDERS {
            assert!(provider_template(p).is_some(), "missing template: {p}");
        }
    }

    #[test]
    fn provider_template_anthropic_sane() {
        let t = provider_template("anthropic").unwrap();
        assert_eq!(t.provider, "anthropic");
        assert_eq!(t.api_key_env.as_deref(), Some("ANTHROPIC_API_KEY"));
        assert!(t.context_max >= 100_000);
    }

    #[test]
    fn provider_template_unknown_returns_none() {
        assert!(provider_template("definitely-fake-provider").is_none());
    }

    fn dummy(name: &str) -> CloudModel {
        CloudModel {
            name: name.into(),
            provider: "anthropic".into(),
            model_id: None,
            context_max: 200_000,
            notes: String::new(),
            quota: QuotaState::Available,
            api_key_env: None,
            base_url: None,
            chat_path: None,
        }
    }

    #[test]
    fn merge_seed_appends_missing_entries() {
        let mut existing = vec![dummy("kept-1"), dummy("kept-2")];
        let seed = vec![dummy("kept-1"), dummy("new-1"), dummy("new-2")];
        let changed = merge_seed_into(&mut existing, seed);
        assert!(changed);
        let names: Vec<&str> = existing.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, vec!["kept-1", "kept-2", "new-1", "new-2"]);
    }

    #[test]
    fn merge_seed_preserves_user_customizations() {
        // User's existing entry has custom notes + quota; seed has the
        // same name with the default values. The merge must NOT clobber.
        let mut existing = vec![CloudModel {
            name: "claude-opus-4-7".into(),
            provider: "anthropic".into(),
            model_id: None,
            context_max: 200_000,
            notes: "user-edited note".into(),
            quota: QuotaState::Low,
            api_key_env: Some("MY_OWN_KEY_VAR".into()),
            base_url: None,
            chat_path: None,
        }];
        let seed = default_seed();
        let _ = merge_seed_into(&mut existing, seed);
        let opus = existing.iter().find(|m| m.name == "claude-opus-4-7").unwrap();
        assert_eq!(opus.notes, "user-edited note");
        assert_eq!(opus.quota, QuotaState::Low);
        assert_eq!(opus.api_key_env.as_deref(), Some("MY_OWN_KEY_VAR"));
    }

    #[test]
    fn merge_seed_no_change_returns_false() {
        let mut existing = default_seed();
        let changed = merge_seed_into(&mut existing, default_seed());
        assert!(!changed, "merging seed into seed shouldn't add entries");
    }

    #[test]
    fn merge_seed_keeps_user_only_entries() {
        let mut existing = vec![dummy("user-only")];
        let seed = vec![dummy("seed-1")];
        let _ = merge_seed_into(&mut existing, seed);
        let names: Vec<&str> = existing.iter().map(|m| m.name.as_str()).collect();
        assert!(names.contains(&"user-only"));
        assert!(names.contains(&"seed-1"));
    }
}
