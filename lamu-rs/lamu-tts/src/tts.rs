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
//!
//! ADR 0023: this tool lives in the lamu-tts MODULE and runs against a
//! `&dyn lamu_core::tools_ext::ToolCtx` (it needs only the model's modality, a
//! load trigger, and the loaded port) — no lamu-mcp dependency. The cloud path
//! ignores the ctx entirely.

use lamu_core::media_paths::resolve_confined_output_path;
use lamu_core::tools_ext::ToolCtx;
use lamu_core::types::Modality;
use serde_json::{json, Value};
use std::future::Future;
use std::pin::Pin;

const DEFAULT_BASE: &str = "https://api.fish.audio";
const KEY_ENV: &str = "FISH_AUDIO_API_KEY";
const MAX_AUDIO_BYTES: u64 = 100 * 1024 * 1024; // 100 MiB safety cap

/// JSON schema for the `text_to_speech` MCP tool.
pub fn schema_text_to_speech() -> Value {
    json!({
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to synthesize. Pass already-VERBALIZED prose — spell math/symbols out ('the integral of x squared'); raw LaTeX/markup is read literally."},
            "model": {"type": "string", "default": "s2-pro", "description": "Fish Audio model header: 's2-pro' (default) or 's1'. A registry model with modality: tts is served locally instead."},
            "format": {"type": "string", "enum": ["mp3", "wav", "pcm", "opus"], "default": "mp3"},
            "reference_id": {"type": "string", "description": "Optional Fish Audio voice model id (cloned/preset voice)."},
            "temperature": {"type": "number", "description": "Expressiveness 0-1 (Fish default 0.7)."},
            "top_p": {"type": "number", "description": "Nucleus sampling 0-1 (Fish default 0.7)."},
            "mp3_bitrate": {"type": "integer", "enum": [64, 128, 192], "description": "mp3 only."},
            "output_path": {"type": "string", "description": "Where to write the audio file. Default: <data_dir>/lamu/tts/tts-<unixsecs>.<format>."}
        },
        "required": ["text"]
    })
}

/// The `ModuleToolHandler` wrapper registered into lamu-core's tool registry.
pub fn dispatch_text_to_speech<'a>(
    ctx: &'a dyn ToolCtx,
    args: Value,
) -> Pin<Box<dyn Future<Output = String> + Send + 'a>> {
    Box::pin(handle_text_to_speech(ctx, args))
}

/// Tool entrypoint. Branches local-vs-cloud BEFORE the cloud `s2-pro|s1`
/// model validator (which would otherwise reject a local registry name). A
/// `model` whose registry entry is `modality: tts` is served locally; anything
/// else (including the default `s2-pro`) goes to Fish Audio cloud.
pub async fn handle_text_to_speech(ctx: &dyn ToolCtx, args: Value) -> String {
    let model = args["model"].as_str().unwrap_or("s2-pro").to_string();
    if ctx.model_modality(&model) == Some(Modality::Tts) {
        handle_text_to_speech_local(ctx, model, args).await
    } else {
        handle_text_to_speech_cloud(args).await
    }
}

/// LOCAL path: ensure the fish-speech server is loaded, then proxy to it.
async fn handle_text_to_speech_local(ctx: &dyn ToolCtx, model: String, args: Value) -> String {
    let text = args["text"].as_str().unwrap_or("").trim().to_string();
    if text.is_empty() {
        return "error: text_to_speech requires a non-empty `text`".into();
    }
    let format = args["format"].as_str().unwrap_or("mp3").to_string();
    if !matches!(format.as_str(), "mp3" | "wav" | "pcm" | "opus") {
        return format!("error: unsupported format '{format}' (mp3|wav|pcm|opus)");
    }

    // Ensure the local server is up. ensure_loaded does the atomic
    // plan/evict/spawn/confirm (evicting LLMs per the modality-tiered
    // scheduler) and is idempotent — "already loaded" if it's up. A pinned
    // LLM blocking the eviction surfaces here as a clear "insufficient
    // space" error.
    let status = ctx.ensure_loaded(&model).await;
    if status.starts_with("error") {
        return status;
    }
    let port = match ctx.loaded_port(&model) {
        Some(p) if p != 0 => p,
        _ => return format!("error: TTS '{model}' not loaded after attempt: {status}"),
    };

    // Long input: the server caps a single request at --max-text-length
    // (4000). Sentence-split into <=TTS_CHUNK_MAX-char pieces, synthesize
    // each, and concatenate the WAVs. (The request's chunk_length=200 still
    // bounds the per-batch codec decode VRAM, independent of total length.)
    const TTS_CHUNK_MAX: usize = 1800;
    let needs_chunking = text.chars().count() > TTS_CHUNK_MAX;
    // PCM WAV is the only byte-concatenable format → force wav when chunking.
    let req_format = if needs_chunking { "wav".to_string() } else { format.clone() };
    let chunks: Vec<String> = if needs_chunking {
        split_for_tts(&text, TTS_CHUNK_MAX)
    } else {
        vec![text.clone()]
    };

    // Per-chunk request params (text + format set per call below).
    let mut base = json!({
        "streaming": false,
        "normalize": true,
        "chunk_length": 200,
        "max_new_tokens": 1024,
    });
    if let Some(rid) = args["reference_id"].as_str() {
        if !rid.is_empty() {
            base["reference_id"] = Value::String(rid.to_string());
        }
    }
    if let Some(seed) = args["seed"].as_u64() {
        base["seed"] = json!(seed);
    }
    if let Some(t) = args["temperature"].as_f64() {
        base["temperature"] = json!(t.clamp(0.1, 1.0)); // fish bounds 0.1-1.0
    }
    if let Some(tp) = args["top_p"].as_f64() {
        base["top_p"] = json!(tp.clamp(0.1, 1.0));
    }

    let out_path = match resolve_confined_output_path("tts", "tts", &req_format, args["output_path"].as_str()) {
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

    let mut parts: Vec<Vec<u8>> = Vec::with_capacity(chunks.len());
    for (i, chunk) in chunks.iter().enumerate() {
        let mut body = base.clone();
        body["text"] = Value::String(chunk.clone());
        body["format"] = Value::String(req_format.clone());
        match tts_post_one(&client, &url, &body).await {
            Ok(b) => parts.push(b),
            Err(e) => return format!("error: chunk {}/{}: {e}", i + 1, chunks.len()),
        }
    }
    let audio: Vec<u8> = if parts.len() == 1 {
        parts.pop().unwrap()
    } else {
        match concat_wav(&parts) {
            Ok(w) => w,
            Err(e) => return format!("error: concat wav: {e}"),
        }
    };
    if audio.is_empty() {
        return "error: fish-speech returned empty audio".into();
    }
    if let Err(e) = std::fs::write(&out_path, &audio) {
        return format!("error: write {}: {e}", out_path.display());
    }
    format!(
        "ok: wrote {} bytes to {} (local {model}, format={req_format}, {} chunk(s))",
        audio.len(),
        out_path.display(),
        chunks.len()
    )
}

/// POST one ServeTTSRequest body, returning the audio bytes (or a ready
/// error string). Mirrors the cloud path's status + size guards.
async fn tts_post_one(client: &reqwest::Client, url: &str, body: &Value) -> Result<Vec<u8>, String> {
    let resp = client
        .post(url)
        .json(body)
        .send()
        .await
        .map_err(|e| format!("post {url}: {e}"))?;
    let st = resp.status();
    if let Some(len) = resp.content_length() {
        if len > MAX_AUDIO_BYTES {
            return Err(format!("response too large ({len} bytes)"));
        }
    }
    let bytes = resp.bytes().await.map_err(|e| format!("read audio bytes: {e}"))?;
    if !st.is_success() {
        let snippet: String = String::from_utf8_lossy(&bytes).chars().take(300).collect();
        return Err(format!("fish-speech HTTP {}: {}", st.as_u16(), snippet));
    }
    if bytes.is_empty() {
        return Err("empty audio (no bytes)".into());
    }
    if bytes.len() as u64 > MAX_AUDIO_BYTES {
        return Err(format!("audio exceeds cap ({} bytes)", bytes.len()));
    }
    Ok(bytes.to_vec())
}

/// Split into sentence-ish units (break after . ! ? or newline, keeping
/// trailing whitespace with the sentence).
fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        cur.push(c);
        if matches!(c, '.' | '!' | '?' | '\n') {
            while matches!(chars.peek(), Some(' ') | Some('\t')) {
                cur.push(chars.next().unwrap());
            }
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Hard char-split (for a single sentence longer than `max`).
fn hard_split(s: &str, max: usize) -> Vec<String> {
    s.chars()
        .collect::<Vec<char>>()
        .chunks(max.max(1))
        .map(|c| c.iter().collect())
        .collect()
}

/// Sentence-aware chunking: pack sentences into <=`max`-char chunks; a
/// single oversized sentence is hard-split. Empty pieces dropped.
fn split_for_tts(text: &str, max: usize) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    for sentence in split_sentences(text) {
        let slen = sentence.chars().count();
        if slen > max {
            if !cur.trim().is_empty() {
                out.push(std::mem::take(&mut cur));
            } else {
                cur.clear();
            }
            out.extend(hard_split(&sentence, max));
            continue;
        }
        if cur.chars().count() + slen > max && !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
        cur.push_str(&sentence);
    }
    if !cur.trim().is_empty() {
        out.push(cur);
    }
    out.into_iter().filter(|s| !s.trim().is_empty()).collect()
}

/// Concatenate canonical 44-byte-header PCM WAVs into one. Keeps the first
/// chunk's header (rate/channels/bits are identical across chunks — same
/// model+format), appends every chunk's PCM data, and patches the RIFF
/// chunk size (offset 4) + data subchunk size (offset 40).
fn concat_wav(parts: &[Vec<u8>]) -> Result<Vec<u8>, String> {
    const HDR: usize = 44;
    let first = parts
        .iter()
        .find(|p| p.len() >= HDR)
        .ok_or("no chunk has a valid WAV header")?;
    let mut out = first[..HDR].to_vec();
    let mut data_len: usize = 0;
    for p in parts {
        if p.len() > HDR {
            out.extend_from_slice(&p[HDR..]);
            data_len += p.len() - HDR;
        }
    }
    // The RIFF + data size fields are 32-bit; a >4 GiB concatenation would
    // silently truncate them (corrupt WAV). Reject instead. (~6.7h of audio
    // at 44.1kHz/16-bit mono — well past any real TTS request.)
    if out.len() > u32::MAX as usize {
        return Err(format!(
            "concatenated WAV exceeds 4 GiB ({} bytes) — RIFF size is 32-bit",
            out.len()
        ));
    }
    let riff = (out.len() as u32).saturating_sub(8);
    out[4..8].copy_from_slice(&riff.to_le_bytes());
    out[40..44].copy_from_slice(&(data_len as u32).to_le_bytes());
    Ok(out)
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

    let out_path = match resolve_confined_output_path("tts", "tts", &format, args["output_path"].as_str()) {
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

    #[test]
    fn split_for_tts_packs_and_hard_splits() {
        // Short → single chunk.
        assert_eq!(split_for_tts("One. Two.", 100).len(), 1);

        // Long → multiple chunks, each ≤ max, content preserved verbatim.
        let long = "Sentence number one. ".repeat(20); // 420 chars
        let parts = split_for_tts(&long, 100);
        assert!(parts.len() > 1);
        assert!(parts.iter().all(|p| p.chars().count() <= 100));
        assert_eq!(parts.join(""), long);

        // A single oversized sentence (no terminator) → hard char-split.
        let mono = "x".repeat(250);
        let hp = split_for_tts(&mono, 100);
        assert_eq!(hp.len(), 3); // 100 + 100 + 50
        assert!(hp.iter().all(|p| p.chars().count() <= 100));
        assert_eq!(hp.join(""), mono);
    }

    #[test]
    fn concat_wav_merges_and_patches_sizes() {
        let mk = |data: &[u8]| {
            let mut v = vec![0u8; 44];
            v[0..4].copy_from_slice(b"RIFF");
            v[8..12].copy_from_slice(b"WAVE");
            v.extend_from_slice(data);
            v
        };
        let out = concat_wav(&[mk(&[1, 2, 3, 4]), mk(&[5, 6])]).unwrap();
        assert_eq!(out.len(), 44 + 6);
        assert_eq!(u32::from_le_bytes(out[40..44].try_into().unwrap()), 6); // data size
        assert_eq!(u32::from_le_bytes(out[4..8].try_into().unwrap()), (44 + 6 - 8)); // RIFF size
        assert_eq!(&out[44..], &[1, 2, 3, 4, 5, 6]);
    }
}
