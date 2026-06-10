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
    /// Optional explicit path appended to base_url. When unset, the
    /// default per provider applies (`/v1/messages` for anthropic,
    /// `/chat/completions` otherwise). Set this when a provider uses
    /// a non-standard endpoint shape (e.g. Anthropic-format proxies
    /// served at `/anthropic/v1/messages`, or a vendor with its own
    /// route layout). The value is appended verbatim — include the
    /// leading slash.
    #[serde(default)]
    pub chat_path: Option<String>,
}

impl CloudModel {
    pub fn full_id(&self) -> String {
        self.model_id
            .clone()
            .unwrap_or_else(|| format!("{}/{}", self.provider, self.name))
    }

    /// True iff `api_key_env` is `None` (no key required) OR the named env
    /// var is present and non-empty after trimming. An exported-but-empty
    /// or whitespace-only var (`export VAR=`) counts as MISSING — otherwise
    /// the row renders valid and the key flows in as `""`, producing a
    /// confusing 401 instead of a clear missing-key message. (#32)
    pub fn key_present(&self) -> bool {
        match &self.api_key_env {
            None => true,
            Some(var) => std::env::var(var).map(|v| !v.trim().is_empty()).unwrap_or(false),
        }
    }

    /// Resolve the API key from the named env var. Returns None when no env
    /// var is configured, the var is unset, or it is present-but-empty. (#32)
    pub fn resolved_api_key(&self) -> Option<String> {
        self.api_key_env
            .as_deref()
            .and_then(|v| std::env::var(v).ok())
            .filter(|k| !k.trim().is_empty())
    }

    /// Full chat URL. Resolution order:
    /// 1. `chat_path` (caller-supplied verbatim suffix), if set.
    /// 2. Default per provider: anthropic → `/v1/messages`, else
    ///    `/chat/completions`.
    /// 3. `gateway_fallback` when `base_url` is absent.
    pub fn chat_url(&self, gateway_fallback: &str) -> String {
        let base = match &self.base_url {
            Some(b) => b.trim_end_matches('/').to_string(),
            None => return gateway_fallback.to_string(),
        };
        if let Some(path) = &self.chat_path {
            return format!("{}{}", base, path);
        }
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

/// Append-or-replace `export VAR=val` in `~/.config/lamu/api-keys.env` —
/// the file `lamu`/serve/MCP source into the process env at startup.
/// Returns the path written. Tightens perms to 0o600 (it holds secrets).
pub fn save_api_key_env(var: &str, val: &str) -> std::io::Result<PathBuf> {
    let dir = dirs::config_dir()
        .map(|d| d.join("lamu"))
        .unwrap_or_else(|| PathBuf::from("."));
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("api-keys.env");
    let existing = std::fs::read_to_string(&path).unwrap_or_default();
    let prefix = format!("export {}=", var);
    let new_line = format!("export {}={}", var, val);
    let updated = if existing.lines().any(|l| l.trim_start().starts_with(&prefix)) {
        let mut s = existing
            .lines()
            .map(|l| {
                if l.trim_start().starts_with(&prefix) {
                    new_line.clone()
                } else {
                    l.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        s.push('\n');
        s
    } else {
        let mut s = existing;
        if !s.is_empty() && !s.ends_with('\n') {
            s.push('\n');
        }
        s.push_str(&new_line);
        s.push('\n');
        s
    };
    std::fs::write(&path, updated)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
    }
    Ok(path)
}

/// Best-effort load: missing file → empty Vec, parse error → empty Vec
/// + a stderr warning so a broken YAML isn't invisible.
///
/// Use this from MCP / non-interactive paths where seeding to disk on
/// missing-file would be surprising. The lamu-cli TUI uses `load()` (in
/// lamu-cli) which seeds the file when missing.
pub fn load_or_empty() -> Vec<CloudModel> {
    use std::sync::{Mutex, OnceLock};
    use std::time::SystemTime;
    // Cache the parsed list, invalidated on file mtime — this is called on every
    // cloud_query / council member / routing_status, and re-reading + YAML-
    // parsing the (potentially 300+-row) catalog per call is blocking IO on a
    // tokio worker. The mtime check preserves the only reason it re-read: picking
    // up edits made by `lamu cloud sync` / the TUI without a restart.
    static CACHE: OnceLock<Mutex<Option<(SystemTime, Vec<CloudModel>)>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));

    // Recover from a poisoned lock rather than panicking — a re-parse is cheaper
    // than bricking every later cloud call.
    let lock = || cache.lock().unwrap_or_else(|e| e.into_inner());

    let path = config_path();
    let mtime = std::fs::metadata(&path).and_then(|m| m.modified()).ok();
    if let Some(mt) = mtime {
        if let Some((cmt, cached)) = lock().as_ref() {
            if *cmt == mt {
                return cached.clone();
            }
        }
    }

    // On a read/parse failure return an empty list, but still cache it (against
    // the file's mtime) so a corrupt cloud-models.yaml doesn't re-parse +
    // eprintln on every cloud call until it's fixed.
    let models = match std::fs::read_to_string(&path) {
        Ok(body) => match serde_yaml::from_str::<CloudModelList>(&body) {
            Ok(l) => l.models,
            Err(e) => {
                eprintln!(
                    "lamu: cloud-models.yaml is corrupt at {} ({}); using empty list.",
                    path.display(),
                    e
                );
                Vec::new()
            }
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(e) => {
            eprintln!(
                "lamu: cloud-models.yaml read failed at {} ({}); using empty list.",
                path.display(),
                e
            );
            Vec::new()
        }
    };
    if let Some(mt) = mtime {
        *lock() = Some((mt, models.clone()));
    }
    models
}

/// Write the cloud-model list to `config_path()` as YAML, creating the parent
/// dir. Used by `lamu cloud sync` to persist the merged catalog. Returns the
/// path written.
pub fn save_models(models: &[CloudModel]) -> std::io::Result<PathBuf> {
    let path = config_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let list = CloudModelList { models: models.to_vec() };
    let yaml = serde_yaml::to_string(&list)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    // Atomic: write a sibling temp then rename over the target (atomic on
    // POSIX, same dir). A crash mid-write can't leave a partial/empty catalog —
    // which a later sync would otherwise read as "no existing models" and drop
    // every hand-set entry.
    let tmp = path.with_extension("yaml.tmp");
    std::fs::write(&tmp, yaml)?;
    std::fs::rename(&tmp, &path)?;
    Ok(path)
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
            chat_path: None,
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
    fn chat_url_explicit_path_overrides_provider_default() {
        // Anthropic-shaped proxy living at /anthropic/v1/messages.
        let mut m = dummy("claude-via-proxy", "anthropic");
        m.base_url = Some("https://gateway.example.com".into());
        m.chat_path = Some("/anthropic/v1/messages".into());
        assert_eq!(
            m.chat_url("fallback"),
            "https://gateway.example.com/anthropic/v1/messages"
        );
    }

    #[test]
    fn chat_url_explicit_path_overrides_openai_default() {
        // Vendor with non-standard route, OpenAI shape.
        let mut m = dummy("vendor", "vendor-x");
        m.base_url = Some("https://api.vendor.example".into());
        m.chat_path = Some("/v2/chat".into());
        assert_eq!(m.chat_url("fallback"), "https://api.vendor.example/v2/chat");
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
