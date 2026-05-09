//! Cloud-model registry — single source of truth for both the
//! lamu-cli TUI and the lamu-mcp server.
//!
//! Path: `$XDG_CONFIG_HOME/lamu/cloud-models.yaml` (defaults to
//! `~/.config/lamu/cloud-models.yaml`).

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Cloud model availability state. Drives the row's color in the TUI.
/// Real quota probing (provider API usage endpoints, Bifrost key-usage
/// stats) is a follow-up — for now this is a manual signal you can edit
/// in cloud-models.yaml.
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
    /// Defaults to empty string for forward-compat with older YAMLs
    /// where the field could be omitted; the loader treats an empty
    /// provider as "use Bifrost / catch-all routing".
    #[serde(default)]
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
    /// Name of the environment variable holding the API key. chat_tui
    /// reads this and passes it as the Bearer token so direct-to-provider
    /// requests (no Bifrost needed) work.
    #[serde(default)]
    pub api_key_env: Option<String>,
    /// Full OpenAI-compat base URL for the provider's API. e.g.
    /// "https://api.deepseek.com" — the `/v1/chat/completions` suffix is
    /// appended automatically. Omit for Bifrost-routed models.
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

    /// Full chat URL. Provider-aware: Anthropic-shaped providers use
    /// `/v1/messages`; everything else (OpenAI, DeepSeek, Moonshot,
    /// Alibaba, Zhipu, etc.) uses `/chat/completions`. base_url is the
    /// root (no path) — the provider router decides the path.
    pub fn chat_url(&self, gateway_fallback: &str) -> String {
        let base = match &self.base_url {
            Some(b) => b.trim_end_matches('/').to_string(),
            None => return gateway_fallback.to_string(),
        };
        if self.provider == "anthropic" {
            return format!("{}/v1/messages", base);
        }
        format!("{}/chat/completions", base)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CloudModelList {
    #[serde(default)]
    pub models: Vec<CloudModel>,
}

/// Resolve the on-disk path for cloud-models.yaml. Falls back through
/// `$XDG_CONFIG_HOME` → `$HOME/.lamu-cloud-models.yaml` → CWD.
pub fn config_path() -> PathBuf {
    if let Some(dir) = dirs::config_dir() {
        return dir.join("lamu").join("cloud-models.yaml");
    }
    if let Some(home) = dirs::home_dir() {
        return home.join(".lamu-cloud-models.yaml");
    }
    PathBuf::from("./cloud-models.yaml")
}

/// Best-effort load: missing file → empty Vec, parse error → empty Vec
/// + a stderr warning so a broken YAML isn't invisible.
///
/// Use this from MCP / non-interactive paths where seeding to disk on
/// missing-file would be surprising. The lamu-cli TUI uses `load()` (in
/// lamu-cli) which seeds the file when missing.
pub fn load_or_empty() -> Vec<CloudModel> {
    let path = config_path();
    let body = match std::fs::read_to_string(&path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Vec::new(),
        Err(e) => {
            eprintln!(
                "lamu: cloud-models.yaml read failed at {} ({}); using empty list.",
                path.display(),
                e
            );
            return Vec::new();
        }
    };
    match serde_yaml::from_str::<CloudModelList>(&body) {
        Ok(l) => l.models,
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
    fn chat_url_anthropic_uses_v1_messages() {
        let mut m = dummy("claude-opus", "anthropic");
        m.base_url = Some("https://api.anthropic.com".into());
        assert_eq!(m.chat_url("fallback"), "https://api.anthropic.com/v1/messages");
    }

    #[test]
    fn chat_url_openai_uses_chat_completions() {
        let mut m = dummy("ds-flash", "deepseek");
        m.base_url = Some("https://api.deepseek.com".into());
        assert_eq!(m.chat_url("fallback"), "https://api.deepseek.com/chat/completions");
    }

    #[test]
    fn chat_url_falls_back_when_no_base() {
        let m = dummy("x", "y");
        assert_eq!(
            m.chat_url("http://localhost:8080/v1/chat/completions"),
            "http://localhost:8080/v1/chat/completions"
        );
    }

    #[test]
    fn chat_url_strips_trailing_slash() {
        let mut m = dummy("ds-flash", "deepseek");
        m.base_url = Some("https://api.deepseek.com/".into());
        assert_eq!(m.chat_url("fallback"), "https://api.deepseek.com/chat/completions");
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
}
