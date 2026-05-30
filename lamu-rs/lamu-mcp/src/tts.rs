//! Text-to-speech tool — routes LOCAL (managed fish-speech S2-Pro) vs
//! CLOUD (Fish Audio api.fish.audio) by the model's registry modality.
//!
//! - If `model` names a registry entry with `modality: tts`, the request is
//!   served LOCALLY: ensure the fish-speech server is loaded (spawns it +
//!   evicts LLMs via the scheduler), then POST to `localhost:<port>/v1/tts`.
//! - Otherwise it's a CLOUD request to Fish Audio (key from
//!   `FISH_AUDIO_API_KEY`, `model:` header = `s2-pro`/`s1`).
//!
//! Both paths write audio to a CONFINED dir (`<data_dir>/lamu/tts`) — an
//! MCP caller (LLM) never gets an arbitrary file-write primitive.
//!
//! NOTE: pass already-VERBALIZED prose. Fish reads input literally — raw
//! LaTeX/markup ("\\iint_D", "$x^2$") is spoken character-by-character.

use crate::server::LamuMcpServer;
use lamu_core::types::Modality;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Component, PathBuf};

const DEFAULT_BASE: &str = "https://api.fish.audio";
const KEY_ENV: &str = "FISH_AUDIO_API_KEY";
const MAX_AUDIO_BYTES: u64 = 100 * 1024 * 1024; // 100 MiB safety cap

/// Resolve where to write the audio, CONFINED to `<data_dir>/lamu/tts`. A
/// caller-supplied `output_path` may only be a relative name with no `..`;
/// absolute paths and parent traversal are rejected so an MCP caller can't
/// turn this into an arbitrary file-write primitive. Shared by both the
/// local and cloud paths. `Err` is a ready-to-return error string.
fn resolve_tts_output_path(output_path: Option<&str>, format: &str) -> Result<PathBuf, String> {
    let dir = dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("lamu")
        .join("tts");
    std::fs::create_dir_all(&dir)
        .map_err(|e| format!("error: create tts dir {}: {e}", dir.display()))?;
    match output_path {
        Some(p) if !p.is_empty() => {
            let pb = PathBuf::from(p);
            if pb.is_absolute() || pb.components().any(|c| matches!(c, Component::ParentDir)) {
                return Err(format!(
                    "error: output_path must be a relative name with no '..' (writes are confined to {}): got '{p}'",
                    dir.display()
                ));
            }
            Ok(dir.join(pb))
        }
        _ => {
            // nanos, not secs — two calls in the same second would otherwise
            // overwrite each other.
            let stamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            Ok(dir.join(format!("tts-{stamp}.{format}")))
        }
    }
}

/// True iff `model` is a registry entry with `modality: tts` (→ serve
/// locally). Pure over the entries map so it's unit-testable without a
/// running server.
fn is_local_tts(entries: &HashMap<String, lamu_core::types::ModelEntry>, model: &str) -> bool {
    entries
        .get(model)
        .map(|e| e.modality == Modality::Tts)
        .unwrap_or(false)
}

/// Tool entrypoint. Branches local-vs-cloud BEFORE the cloud `s2-pro|s1`
/// model validator (which would otherwise reject a local registry name).
pub async fn handle_text_to_speech_stateful(server: &LamuMcpServer, args: Value) -> String {
    let model = args["model"].as_str().unwrap_or("s2-pro").to_string();
    let local = {
        let st = server.state.lock();
        is_local_tts(&st.entries, &model)
    };
    if local {
        handle_text_to_speech_local(server, model, args).await
    } else {
        handle_text_to_speech_cloud(args).await
    }
}

/// LOCAL path: ensure the fish-speech server is loaded, then proxy to it.
async fn handle_text_to_speech_local(server: &LamuMcpServer, model: String, args: Value) -> String {
    let text = args["text"].as_str().unwrap_or("").trim().to_string();
    if text.is_empty() {
        return "error: text_to_speech requires a non-empty `text`".into();
    }
    let format = args["format"].as_str().unwrap_or("mp3").to_string();
    if !matches!(format.as_str(), "mp3" | "wav" | "pcm" | "opus") {
        return format!("error: unsupported format '{format}' (mp3|wav|pcm|opus)");
    }

    // Ensure the local server is up. handle_load_model does the atomic
    // plan/evict/spawn/confirm (evicting LLMs per the modality-tiered
    // scheduler) and is idempotent — "already loaded" if it's up. A pinned
    // LLM blocking the eviction surfaces here as a clear "insufficient
    // space" error.
    let status = server.handle_load_model(json!({ "name": model })).await;
    if status.starts_with("error") {
        return status;
    }
    let port = {
        let st = server.state.lock();
        match st.scheduler.get_loaded(&model) {
            Some(m) if m.port != 0 => m.port,
            _ => return format!("error: TTS '{model}' not loaded after attempt: {status}"),
        }
    };

    // ServeTTSRequest. chunk_length=200 bounds the per-batch codec decode
    // (the lever that keeps VRAM bounded regardless of total text length);
    // max_new_tokens capped at the server default.
    let mut body = json!({
        "text": text,
        "format": format,
        "streaming": false,
        "normalize": true,
        "chunk_length": 200,
        "max_new_tokens": 1024,
    });
    if let Some(rid) = args["reference_id"].as_str() {
        if !rid.is_empty() {
            body["reference_id"] = Value::String(rid.to_string());
        }
    }
    if let Some(seed) = args["seed"].as_u64() {
        body["seed"] = json!(seed);
    }
    if let Some(t) = args["temperature"].as_f64() {
        body["temperature"] = json!(t.clamp(0.1, 1.0)); // fish bounds 0.1-1.0
    }
    if let Some(tp) = args["top_p"].as_f64() {
        body["top_p"] = json!(tp.clamp(0.1, 1.0));
    }

    let out_path = match resolve_tts_output_path(args["output_path"].as_str(), &format) {
        Ok(p) => p,
        Err(e) => return e,
    };

    let url = format!("http://127.0.0.1:{port}/v1/tts");
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()
    {
        Ok(c) => c,
        Err(e) => return format!("error: client init: {e}"),
    };
    let resp = match client.post(&url).json(&body).send().await {
        Ok(r) => r,
        Err(e) => return format!("error: post {url}: {e}"),
    };
    let st = resp.status();
    let bytes = match resp.bytes().await {
        Ok(b) => b,
        Err(e) => return format!("error: read audio bytes: {e}"),
    };
    if !st.is_success() {
        let snippet: String = String::from_utf8_lossy(&bytes).chars().take(300).collect();
        return format!("error: fish-speech HTTP {}: {}", st.as_u16(), snippet);
    }
    if bytes.is_empty() {
        return "error: fish-speech returned empty audio (no bytes)".into();
    }
    if let Err(e) = std::fs::write(&out_path, &bytes) {
        return format!("error: write {}: {e}", out_path.display());
    }
    format!(
        "ok: wrote {} bytes to {} (local {model}, format={format})",
        bytes.len(),
        out_path.display()
    )
}

/// CLOUD path: Fish Audio api.fish.audio. `model:` is a request HEADER
/// (verified against docs.fish.audio + the python SDK; the body has no
/// `model`), default `s2-pro`.
async fn handle_text_to_speech_cloud(args: Value) -> String {
    let text = args["text"].as_str().unwrap_or("").trim().to_string();
    if text.is_empty() {
        return "error: text_to_speech requires a non-empty `text`".into();
    }
    let model = args["model"].as_str().unwrap_or("s2-pro").to_string();
    if !matches!(model.as_str(), "s2-pro" | "s1") {
        return format!("error: unknown cloud model '{model}' (expected 's2-pro' or 's1'); for a local model declare modality: tts in the registry");
    }
    let format = args["format"].as_str().unwrap_or("mp3").to_string();
    if !matches!(format.as_str(), "mp3" | "wav" | "pcm" | "opus") {
        return format!("error: unsupported format '{format}' (mp3|wav|pcm|opus)");
    }

    let key = match std::env::var(KEY_ENV) {
        // Trim: a key pasted/exported with a trailing newline would 401.
        Ok(k) if !k.trim().is_empty() => k.trim().to_string(),
        _ => {
            return format!(
                "error: ${KEY_ENV} is not set. Export your Fish Audio API key, e.g. `export {KEY_ENV}=<key>`, then retry — or use a local model (registry modality: tts)."
            )
        }
    };
    let base = std::env::var("FISH_AUDIO_BASE_URL")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_BASE.to_string());
    let url = format!("{}/v1/tts", base.trim_end_matches('/'));

    let out_path = match resolve_tts_output_path(args["output_path"].as_str(), &format) {
        Ok(p) => p,
        Err(e) => return e,
    };

    let mut body = json!({ "text": text, "format": format });
    if let Some(rid) = args["reference_id"].as_str() {
        if !rid.is_empty() {
            body["reference_id"] = Value::String(rid.to_string());
        }
    }
    if let Some(t) = args["temperature"].as_f64() {
        body["temperature"] = json!(t.clamp(0.0, 1.0));
    }
    if let Some(tp) = args["top_p"].as_f64() {
        body["top_p"] = json!(tp.clamp(0.0, 1.0));
    }
    if let Some(br) = args["mp3_bitrate"].as_u64() {
        if matches!(br, 64 | 128 | 192) {
            body["mp3_bitrate"] = json!(br);
        }
    }

    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()
    {
        Ok(c) => c,
        Err(e) => return format!("error: client init: {e}"),
    };
    let resp = match client
        .post(&url)
        .bearer_auth(&key)
        .header("model", &model)
        .json(&body)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => return format!("error: post {url}: {e}"),
    };
    let status = resp.status();
    if let Some(len) = resp.content_length() {
        if len > MAX_AUDIO_BYTES {
            return format!("error: fish.audio response too large ({len} bytes)");
        }
    }
    let bytes = match resp.bytes().await {
        Ok(b) => b,
        Err(e) => return format!("error: read audio bytes: {e}"),
    };
    if !status.is_success() {
        let snippet: String = String::from_utf8_lossy(&bytes).chars().take(300).collect();
        return format!("error: fish.audio HTTP {}: {}", status.as_u16(), snippet);
    }
    if bytes.is_empty() {
        return "error: fish.audio returned empty audio (no bytes)".into();
    }
    if bytes.len() as u64 > MAX_AUDIO_BYTES {
        return format!("error: fish.audio audio exceeds cap ({} bytes)", bytes.len());
    }
    if let Err(e) = std::fs::write(&out_path, &bytes) {
        return format!("error: write {}: {e}", out_path.display());
    }
    format!(
        "ok: wrote {} bytes to {} (cloud {model}, format={format})",
        bytes.len(),
        out_path.display()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use lamu_core::types::{
        BackendType, ModelEntry, ModelFormat, ModelStatus,
    };
    use std::path::PathBuf;

    fn entry(name: &str, modality: Modality) -> ModelEntry {
        ModelEntry {
            name: name.to_string(),
            path: PathBuf::from("/tmp/x"),
            format: ModelFormat::Gguf,
            backend: BackendType::LlamaCpp,
            arch: "t".into(),
            params_b: 1.0,
            quant: "Q4".into(),
            vram_mb: 1,
            context_max: 0,
            capabilities: vec![],
            reasoning_marker: None,
            speculative: None,
            sampling: None,
            pinned: false,
            main: false,
            notes: String::new(),
            status: ModelStatus::Unspecified,
            modality,
        }
    }

    #[test]
    fn is_local_tts_true_for_tts_entry() {
        let mut m = HashMap::new();
        m.insert("local-fish-s2pro".to_string(), entry("local-fish-s2pro", Modality::Tts));
        assert!(is_local_tts(&m, "local-fish-s2pro"));
    }

    #[test]
    fn is_local_tts_false_for_llm_or_unknown() {
        let mut m = HashMap::new();
        m.insert("chat".to_string(), entry("chat", Modality::Llm));
        assert!(!is_local_tts(&m, "chat")); // an LLM entry
        assert!(!is_local_tts(&m, "s2-pro")); // unknown → cloud
    }

    #[test]
    fn output_path_rejects_traversal_and_absolute() {
        assert!(resolve_tts_output_path(Some("../escape.mp3"), "mp3").is_err());
        assert!(resolve_tts_output_path(Some("/etc/passwd"), "mp3").is_err());
        assert!(resolve_tts_output_path(Some("ok.mp3"), "mp3").is_ok());
        assert!(resolve_tts_output_path(None, "wav").is_ok());
    }
}
