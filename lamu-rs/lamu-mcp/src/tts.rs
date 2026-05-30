//! Cloud text-to-speech via the Fish Audio API (api.fish.audio).
//!
//! LAMU's first non-LLM cloud modality. Same shape as `cloud_query`: an
//! HTTP POST to a hosted provider, API key from the environment, no local
//! GPU (so it never contends with the LLM VRAM scheduler). The `model` arg
//! maps to Fish Audio's `model:` request HEADER — default `s2-pro`. (Per
//! docs.fish.audio + the official python SDK, model selection is a header,
//! NOT a body field; the TTSRequest body has no `model`.)
//!
//! Request (per https://docs.fish.audio/.../text-to-speech):
//!   POST {base}/v1/tts
//!   Authorization: Bearer <key>     (FISH_AUDIO_API_KEY)
//!   model: s2-pro | s1
//!   content-type: application/json  (set by reqwest .json())
//!   { "text": "...", "format": "mp3"|"wav"|"pcm"|"opus", "reference_id"?: "..." }
//! Response: raw audio bytes (chunked). Errors: JSON { status, message }.
//!
//! NOTE: pass already-VERBALIZED prose. Fish reads input literally — raw
//! LaTeX/markup ("\\iint_D", "$x^2$") is spoken character-by-character.

use serde_json::{json, Value};
use std::path::{Component, PathBuf};

const DEFAULT_BASE: &str = "https://api.fish.audio";
const KEY_ENV: &str = "FISH_AUDIO_API_KEY";
const MAX_AUDIO_BYTES: u64 = 100 * 1024 * 1024; // 100 MiB safety cap

pub async fn handle_text_to_speech(args: Value) -> String {
    let text = args["text"].as_str().unwrap_or("").trim().to_string();
    if text.is_empty() {
        return "error: text_to_speech requires a non-empty `text`".into();
    }
    let model = args["model"].as_str().unwrap_or("s2-pro").to_string();
    if !matches!(model.as_str(), "s2-pro" | "s1") {
        return format!("error: unknown model '{model}' (expected 's2-pro' or 's1')");
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
                "error: ${KEY_ENV} is not set. Export your Fish Audio API key, e.g. `export {KEY_ENV}=<key>`, then retry."
            )
        }
    };
    let base = std::env::var("FISH_AUDIO_BASE_URL")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_BASE.to_string());
    let url = format!("{}/v1/tts", base.trim_end_matches('/'));

    // All writes are CONFINED to <data_dir>/lamu/tts — an MCP caller (LLM)
    // must not get an arbitrary file-write primitive. A caller-supplied
    // `output_path` may only be a relative name with no `..`; absolute paths
    // and parent traversal are rejected.
    let dir = dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("lamu")
        .join("tts");
    if let Err(e) = std::fs::create_dir_all(&dir) {
        return format!("error: create tts dir {}: {e}", dir.display());
    }
    let out_path = match args["output_path"].as_str() {
        Some(p) if !p.is_empty() => {
            let pb = PathBuf::from(p);
            if pb.is_absolute()
                || pb.components().any(|c| matches!(c, Component::ParentDir))
            {
                return format!(
                    "error: output_path must be a relative name with no '..' (writes are confined to {}): got '{p}'",
                    dir.display()
                );
            }
            dir.join(pb)
        }
        _ => {
            // nanos, not secs — two calls in the same second would otherwise
            // overwrite each other.
            let stamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            dir.join(format!("tts-{stamp}.{format}"))
        }
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

    // `model` is a header (verified against the API docs); `.json()` sets
    // content-type + serializes the body.
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
    // Capture status before consuming the body (reqwest doesn't error on
    // 4xx/5xx — same gate as cloud.rs #29).
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
        "ok: wrote {} bytes to {} (model={model}, format={format})",
        bytes.len(),
        out_path.display()
    )
}
