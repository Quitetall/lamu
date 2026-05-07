//! Cloud-model registry.
//!
//! Local models live in `~/local-llm/config/models.yaml` (the lamu-core
//! registry — actual GGUFs on disk). Cloud models are everything else:
//! Anthropic, OpenAI, Pi, etc. They're served through Bifrost
//! (`http://localhost:8080/v1/chat/completions`) which dispatches by
//! `provider/model_id`.
//!
//! Path: `$XDG_CONFIG_HOME/lamu/cloud-models.yaml` (defaults to
//! `~/.config/lamu/cloud-models.yaml`). Missing file = a sensible
//! seed list (Anthropic + OpenAI families). Edit the file to add
//! more or to wire your own provider routes.
//!
//! Cloud models DON'T require API keys here — Bifrost handles auth
//! from its own key store. lamu just sends the request.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Cloud model availability state. Drives the row's color in the TUI:
///   Available → blue (use it)
///   Low       → amber  (running out of budget, slow it down)
///   Exhausted → red    (don't try)
///
/// We default everything to Available. Real quota probing (provider API
/// usage endpoints, Bifrost key-usage stats) is a follow-up — for now
/// this is a manual signal you can edit in cloud-models.yaml.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuotaState {
    Available,
    Low,
    Exhausted,
}

impl Default for QuotaState {
    fn default() -> Self {
        QuotaState::Available
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudModel {
    /// Display name used in the TUI (e.g. "claude-opus-4-7").
    pub name: String,
    /// Provider key Bifrost knows about ("anthropic", "openai", ...).
    pub provider: String,
    /// Bifrost-shaped id for the OpenAI request body. Defaults to
    /// `<provider>/<name>` if unset.
    #[serde(default)]
    pub model_id: Option<String>,
    /// Headline context window in tokens.
    #[serde(default)]
    pub context_max: u32,
    /// Free-form notes shown next to the row.
    #[serde(default)]
    pub notes: String,
    /// Quota status — colors the row.
    #[serde(default)]
    pub quota: QuotaState,
    /// Name of the environment variable holding the API key.
    /// chat_tui reads this and passes it as the Bearer token so
    /// direct-to-provider requests (no Bifrost needed) work.
    #[serde(default)]
    pub api_key_env: Option<String>,
    /// Full OpenAI-compat base URL for the provider's API.
    /// e.g. "https://api.deepseek.com" — the `/v1/chat/completions`
    /// suffix is appended automatically. Omit for Bifrost-routed models.
    #[serde(default)]
    pub base_url: Option<String>,
}

impl CloudModel {
    pub fn full_id(&self) -> String {
        self.model_id
            .clone()
            .unwrap_or_else(|| format!("{}/{}", self.provider, self.name))
    }

    /// True iff `api_key_env` is unset OR the named env var is present.
    pub fn key_present(&self) -> bool {
        match &self.api_key_env {
            None => true,
            Some(var) => std::env::var(var).is_ok(),
        }
    }

    /// Resolve the API key from the named env var. Returns None when
    /// no env var is configured or the var is unset.
    pub fn resolved_api_key(&self) -> Option<String> {
        self.api_key_env.as_deref().and_then(|v| std::env::var(v).ok())
    }

    /// Full chat-completions URL. Uses base_url when set, otherwise
    /// falls back to the Bifrost gateway or a default.
    pub fn chat_url(&self, gateway_fallback: &str) -> String {
        match &self.base_url {
            Some(base) => format!("{}/chat/completions", base.trim_end_matches('/')),
            None => gateway_fallback.to_string(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CloudModelList {
    #[serde(default)]
    pub models: Vec<CloudModel>,
}

pub fn config_path() -> PathBuf {
    if let Some(dir) = dirs::config_dir() {
        return dir.join("lamu").join("cloud-models.yaml");
    }
    if let Some(home) = dirs::home_dir() {
        return home.join(".lamu-cloud-models.yaml");
    }
    PathBuf::from("./cloud-models.yaml")
}

pub fn default_seed() -> Vec<CloudModel> {
    vec![
        // Anthropic — kept. Bifrost route handles auth.
        CloudModel {
            name: "claude-opus-4-7".into(),
            provider: "anthropic".into(),
            model_id: None,
            context_max: 200_000,
            notes: "Anthropic Claude Opus 4.7 (best reasoning)".into(),
            quota: QuotaState::Available,
            api_key_env: Some("ANTHROPIC_API_KEY".into()),
            base_url: None,
        },
        CloudModel {
            name: "claude-sonnet-4-6".into(),
            provider: "anthropic".into(),
            model_id: None,
            context_max: 200_000,
            notes: "Anthropic Claude Sonnet 4.6 (workhorse)".into(),
            quota: QuotaState::Available,
            api_key_env: Some("ANTHROPIC_API_KEY".into()),
            base_url: None,
        },
        CloudModel {
            name: "claude-haiku-4-5".into(),
            provider: "anthropic".into(),
            model_id: None,
            context_max: 200_000,
            notes: "Anthropic Claude Haiku 4.5 (fast)".into(),
            quota: QuotaState::Available,
            api_key_env: Some("ANTHROPIC_API_KEY".into()),
            base_url: None,
        },
        // GLM 5.1 (Zhipu / Z.AI)
        CloudModel {
            name: "glm-5.1".into(),
            provider: "zhipu".into(),
            model_id: None,
            context_max: 200_000,
            notes: "Zhipu GLM 5.1 (open-weights flagship)".into(),
            quota: QuotaState::Available,
            api_key_env: Some("ZHIPU_API_KEY".into()),
            base_url: None,
        },
        // Kimi K2.6 (Moonshot)
        CloudModel {
            name: "kimi-k2.6".into(),
            provider: "moonshot".into(),
            model_id: None,
            context_max: 256_000,
            notes: "Moonshot Kimi K2.6 (long-context agentic)".into(),
            quota: QuotaState::Available,
            api_key_env: Some("MOONSHOT_API_KEY".into()),
            base_url: None,
        },
        // Qwen3 large via Alibaba DashScope (>100B params).
        CloudModel {
            name: "qwen3-235b-a22b-thinking-2507".into(),
            provider: "alibaba".into(),
            model_id: None,
            context_max: 262_144,
            notes: "Alibaba Qwen3 235B-A22B Thinking (DashScope)".into(),
            quota: QuotaState::Available,
            api_key_env: Some("DASHSCOPE_API_KEY".into()),
            base_url: None,
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
        },
        // DeepSeek V4 — direct API (no Bifrost needed)
        CloudModel {
            name: "deepseek-v4-flash".into(),
            provider: "deepseek".into(),
            model_id: Some("deepseek-v4-flash".into()),
            context_max: 128_000,
            notes: "DeepSeek V4 Flash — fast, non-thinking (direct API)".into(),
            quota: QuotaState::Available,
            api_key_env: Some("DEEPSEEK_API_KEY".into()),
            base_url: Some("https://api.deepseek.com".into()),
        },
        CloudModel {
            name: "deepseek-v4-pro".into(),
            provider: "deepseek".into(),
            model_id: Some("deepseek-v4-pro".into()),
            context_max: 128_000,
            notes: "DeepSeek V4 Pro — thinking mode (direct API)".into(),
            quota: QuotaState::Available,
            api_key_env: Some("DEEPSEEK_API_KEY".into()),
            base_url: Some("https://api.deepseek.com".into()),
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
        _ => None,
    }
}

pub const KNOWN_PROVIDERS: &[&str] = &[
    "anthropic", "openai", "zhipu", "moonshot", "alibaba",
    "deepseek", "mistral", "openrouter", "google", "xai",
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
/// users to edit. Bad YAML → empty list + a stderr warning.
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
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(_) => return Vec::new(),
    };
    match serde_yaml::from_slice::<CloudModelList>(&bytes) {
        Ok(list) => list.models,
        Err(e) => {
            eprintln!(
                "lamu: cloud-models.yaml is corrupt at {} ({}); using empty list.",
                path.display(),
                e
            );
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy(name: &str, provider: &str) -> CloudModel {
        CloudModel {
            name: name.into(),
            provider: provider.into(),
            model_id: None,
            context_max: 200_000,
            notes: String::new(),
            quota: QuotaState::Available,
            api_key_env: None,
            base_url: None,
        }
    }

    #[test]
    fn full_id_defaults_to_provider_slash_name() {
        assert_eq!(
            dummy("claude-opus-4-7", "anthropic").full_id(),
            "anthropic/claude-opus-4-7"
        );
    }

    #[test]
    fn full_id_honours_explicit_model_id() {
        let mut m = dummy("name", "p");
        m.model_id = Some("custom/path".into());
        assert_eq!(m.full_id(), "custom/path");
    }

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
    fn key_present_when_no_env_required() {
        let m = dummy("x", "y");
        assert!(m.key_present());
    }

    #[test]
    fn key_absent_when_env_unset() {
        let mut m = dummy("x", "y");
        m.api_key_env = Some("LAMU_TEST_NEVER_SET_VAR_xyz123".into());
        assert!(!m.key_present());
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
}
