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
}

impl CloudModel {
    pub fn full_id(&self) -> String {
        self.model_id
            .clone()
            .unwrap_or_else(|| format!("{}/{}", self.provider, self.name))
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
        CloudModel {
            name: "claude-opus-4-7".into(),
            provider: "anthropic".into(),
            model_id: None,
            context_max: 200_000,
            notes: "Anthropic Claude Opus 4.7 (best reasoning)".into(),
            quota: QuotaState::Available,
        },
        CloudModel {
            name: "claude-sonnet-4-6".into(),
            provider: "anthropic".into(),
            model_id: None,
            context_max: 200_000,
            notes: "Anthropic Claude Sonnet 4.6 (workhorse)".into(),
            quota: QuotaState::Available,
        },
        CloudModel {
            name: "claude-haiku-4-5".into(),
            provider: "anthropic".into(),
            model_id: None,
            context_max: 200_000,
            notes: "Anthropic Claude Haiku 4.5 (fast)".into(),
            quota: QuotaState::Available,
        },
        CloudModel {
            name: "gpt-4.1".into(),
            provider: "openai".into(),
            model_id: None,
            context_max: 1_000_000,
            notes: "OpenAI GPT-4.1".into(),
            quota: QuotaState::Available,
        },
        CloudModel {
            name: "gpt-4.1-mini".into(),
            provider: "openai".into(),
            model_id: None,
            context_max: 1_000_000,
            notes: "OpenAI GPT-4.1 mini".into(),
            quota: QuotaState::Available,
        },
        CloudModel {
            name: "gpt-4.5-preview".into(),
            provider: "openai".into(),
            model_id: None,
            context_max: 128_000,
            notes: "OpenAI GPT-4.5 preview".into(),
            quota: QuotaState::Available,
        },
    ]
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

    #[test]
    fn full_id_defaults_to_provider_slash_name() {
        let m = CloudModel {
            name: "claude-opus-4-7".into(),
            provider: "anthropic".into(),
            model_id: None,
            context_max: 200_000,
            notes: "x".into(),
            quota: QuotaState::Available,
        };
        assert_eq!(m.full_id(), "anthropic/claude-opus-4-7");
    }

    #[test]
    fn full_id_honours_explicit_model_id() {
        let m = CloudModel {
            name: "name".into(),
            provider: "p".into(),
            model_id: Some("custom/path".into()),
            context_max: 0,
            notes: "".into(),
            quota: QuotaState::Available,
        };
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
    fn yaml_round_trip() {
        let list = CloudModelList { models: default_seed() };
        let s = serde_yaml::to_string(&list).unwrap();
        let back: CloudModelList = serde_yaml::from_str(&s).unwrap();
        assert_eq!(back.models.len(), default_seed().len());
    }
}
