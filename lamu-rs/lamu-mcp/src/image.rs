//! Image generation — local ComfyUI (managed subprocess), modality routing
//! mirrors `tts`. A `model` whose registry entry is `modality: image` is
//! served by spawning ComfyUI (evicting LLMs via the tiered scheduler) and
//! proxying a txt2img workflow: POST /prompt → poll /history → GET /view.
//!
//! One ComfyUI serves many checkpoints; the per-request `checkpoint`
//! (a file under <comfy>/models/checkpoints/) is selected in the workflow
//! graph, not at spawn. Output PNGs are written to a CONFINED dir
//! (<data_dir>/lamu/images) — an MCP caller never gets an arbitrary write.

use crate::server::LamuMcpServer;
use lamu_core::types::Modality;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Component, PathBuf};
use std::time::Duration;

const MAX_IMAGE_BYTES: u64 = 64 * 1024 * 1024; // 64 MiB/image safety cap

/// True iff `model` is a registry entry with `modality: image`.
fn is_local_image(entries: &HashMap<String, lamu_core::types::ModelEntry>, model: &str) -> bool {
    entries
        .get(model)
        .map(|e| e.modality == Modality::Image)
        .unwrap_or(false)
}

/// Confined output path under <data_dir>/lamu/images (reject absolute + `..`).
fn resolve_image_output_path(output_path: Option<&str>, ext: &str) -> Result<PathBuf, String> {
    let dir = dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("lamu")
        .join("images");
    std::fs::create_dir_all(&dir)
        .map_err(|e| format!("error: create images dir {}: {e}", dir.display()))?;
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
            let stamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            Ok(dir.join(format!("img-{stamp}.{ext}")))
        }
    }
}

/// Canonical SDXL/SD txt2img workflow in ComfyUI API format.
fn build_workflow(
    checkpoint: &str,
    prompt: &str,
    negative: &str,
    width: u64,
    height: u64,
    steps: u64,
    cfg: f64,
    seed: u64,
    sampler: &str,
) -> Value {
    json!({
        "3": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": steps, "cfg": cfg,
            "sampler_name": sampler, "scheduler": "normal", "denoise": 1.0,
            "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0],
            "latent_image": ["5", 0]
        }},
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["4", 1]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "lamu", "images": ["8", 0]}}
    })
}

/// Tool entrypoint. Local ComfyUI only (no cloud image provider in M1).
pub async fn handle_generate_image(server: &LamuMcpServer, args: Value) -> String {
    let model = args["model"].as_str().unwrap_or("comfy-image").to_string();
    let local = {
        let st = server.state.lock();
        is_local_image(&st.entries, &model)
    };
    if !local {
        return format!(
            "error: '{model}' is not a registry image model (need modality: image). No cloud image provider is wired."
        );
    }

    let prompt = args["prompt"].as_str().unwrap_or("").trim().to_string();
    if prompt.is_empty() {
        return "error: generate_image requires a non-empty `prompt`".into();
    }
    let checkpoint = match args["checkpoint"].as_str() {
        Some(c) if !c.is_empty() => c.to_string(),
        _ => return "error: generate_image requires `checkpoint` (a file under ComfyUI models/checkpoints/, e.g. 'sd_xl_base_1.0.safetensors')".into(),
    };
    // Confine the checkpoint to models/checkpoints/ — block absolute paths
    // and `..` so a crafted name can't load a .safetensors from elsewhere.
    // (Subfolders like 'sdxl/foo.safetensors' are allowed.)
    {
        let cpb = std::path::Path::new(&checkpoint);
        if cpb.is_absolute() || cpb.components().any(|c| matches!(c, Component::ParentDir)) {
            return format!(
                "error: checkpoint must be a name under models/checkpoints/ with no '..' or absolute path: got '{checkpoint}'"
            );
        }
    }
    let negative = args["negative"].as_str().unwrap_or("").to_string();
    let width = args["width"].as_u64().unwrap_or(1024).clamp(64, 2048);
    let height = args["height"].as_u64().unwrap_or(1024).clamp(64, 2048);
    let steps = args["steps"].as_u64().unwrap_or(25).clamp(1, 150);
    let cfg = args["cfg"].as_f64().unwrap_or(7.0).clamp(0.0, 30.0);
    let sampler = args["sampler"].as_str().unwrap_or("euler").to_string();
    let seed = args["seed"].as_u64().unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0)
    });

    // Ensure ComfyUI is up (spawns + evicts LLMs per the scheduler).
    let status = server.handle_load_model(json!({ "name": model })).await;
    if status.starts_with("error") {
        return status;
    }
    let port = {
        let st = server.state.lock();
        match st.scheduler.get_loaded(&model) {
            Some(m) if m.port != 0 => m.port,
            _ => return format!("error: image model '{model}' not loaded after attempt: {status}"),
        }
    };

    let out_path = match resolve_image_output_path(args["output_path"].as_str(), "png") {
        Ok(p) => p,
        Err(e) => return e,
    };

    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()
    {
        Ok(c) => c,
        Err(e) => return format!("error: client init: {e}"),
    };
    let base = format!("http://127.0.0.1:{port}");
    let workflow = build_workflow(
        &checkpoint, &prompt, &negative, width, height, steps, cfg, seed, &sampler,
    );

    // Queue the prompt.
    let queue = match client
        .post(format!("{base}/prompt"))
        .json(&json!({ "prompt": workflow, "client_id": "lamu-mcp" }))
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => return format!("error: post /prompt: {e}"),
    };
    if !queue.status().is_success() {
        let body: String = queue
            .text()
            .await
            .unwrap_or_default()
            .chars()
            .take(400)
            .collect();
        return format!("error: ComfyUI rejected workflow (checkpoint '{checkpoint}' present?): {body}");
    }
    let qv: Value = match queue.json().await {
        Ok(v) => v,
        Err(e) => return format!("error: parse /prompt response: {e}"),
    };
    let prompt_id = match qv["prompt_id"].as_str() {
        Some(id) => id.to_string(),
        None => return format!("error: /prompt returned no prompt_id: {qv}"),
    };

    // Poll /history/<id> until the prompt's outputs appear (image gen +
    // first-time checkpoint load can take a while → up to ~5 min).
    let hist_url = format!("{base}/history/{prompt_id}");
    let mut images: Vec<Value> = Vec::new();
    for i in 0..150 {
        if i > 0 {
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
        // Short per-poll timeout (separate from the client's 600s): if
        // ComfyUI's port is bound but the server hung, the default timeout
        // would block each poll for 600s → 150×600s, not the intended ~5min.
        let h = match client.get(&hist_url).timeout(Duration::from_secs(5)).send().await {
            Ok(r) => r,
            Err(_) => continue,
        };
        let hv: Value = match h.json().await {
            Ok(v) => v,
            Err(_) => continue,
        };
        if let Some(entry) = hv.get(&prompt_id) {
            // SaveImage node "9" → outputs.images[]
            if let Some(imgs) = entry["outputs"]["9"]["images"].as_array() {
                images = imgs.clone();
                break;
            }
            // Some graphs key outputs by a different SaveImage id; scan all.
            if let Some(outputs) = entry["outputs"].as_object() {
                for node in outputs.values() {
                    if let Some(imgs) = node["images"].as_array() {
                        if !imgs.is_empty() {
                            images = imgs.clone();
                            break;
                        }
                    }
                }
                if !images.is_empty() {
                    break;
                }
            }
        }
    }
    if images.is_empty() {
        return format!("error: ComfyUI produced no image within timeout (prompt_id {prompt_id}) — check the server log");
    }

    // Fetch the first image via /view.
    let img = &images[0];
    let filename = img["filename"].as_str().unwrap_or("");
    let subfolder = img["subfolder"].as_str().unwrap_or("");
    let typ = img["type"].as_str().unwrap_or("output");
    let view = match client
        .get(format!("{base}/view"))
        .query(&[("filename", filename), ("subfolder", subfolder), ("type", typ)])
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => return format!("error: get /view: {e}"),
    };
    if !view.status().is_success() {
        return format!("error: /view HTTP {}", view.status().as_u16());
    }
    if let Some(len) = view.content_length() {
        if len > MAX_IMAGE_BYTES {
            return format!("error: image too large ({len} bytes)");
        }
    }
    let bytes = match view.bytes().await {
        Ok(b) => b,
        Err(e) => return format!("error: read image bytes: {e}"),
    };
    if bytes.is_empty() {
        return "error: ComfyUI returned an empty image".into();
    }
    if bytes.len() as u64 > MAX_IMAGE_BYTES {
        // Guards the no-Content-Length case the pre-check above can't catch.
        return format!("error: image exceeds cap ({} bytes)", bytes.len());
    }
    if let Err(e) = std::fs::write(&out_path, &bytes) {
        return format!("error: write {}: {e}", out_path.display());
    }
    format!(
        "ok: wrote {} bytes to {} ({}x{}, {} steps, cfg {cfg}, seed {seed}, checkpoint {checkpoint})",
        bytes.len(),
        out_path.display(),
        width,
        height,
        steps
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use lamu_core::types::{BackendType, ModelEntry, ModelFormat, ModelStatus};

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
    fn is_local_image_true_only_for_image_entry() {
        let mut m = HashMap::new();
        m.insert("comfy-image".to_string(), entry("comfy-image", Modality::Image));
        m.insert("chat".to_string(), entry("chat", Modality::Llm));
        assert!(is_local_image(&m, "comfy-image"));
        assert!(!is_local_image(&m, "chat"));
        assert!(!is_local_image(&m, "unknown"));
    }

    #[test]
    fn output_path_confined() {
        assert!(resolve_image_output_path(Some("../x.png"), "png").is_err());
        assert!(resolve_image_output_path(Some("/etc/x"), "png").is_err());
        assert!(resolve_image_output_path(Some("ok.png"), "png").is_ok());
        assert!(resolve_image_output_path(None, "png").is_ok());
    }

    #[test]
    fn workflow_has_required_nodes() {
        let w = build_workflow("ck.safetensors", "a cat", "blurry", 1024, 1024, 25, 7.0, 42, "euler");
        assert_eq!(w["4"]["inputs"]["ckpt_name"], "ck.safetensors");
        assert_eq!(w["6"]["inputs"]["text"], "a cat");
        assert_eq!(w["7"]["inputs"]["text"], "blurry");
        assert_eq!(w["3"]["inputs"]["seed"], 42);
        assert_eq!(w["5"]["inputs"]["width"], 1024);
        assert_eq!(w["9"]["class_type"], "SaveImage");
    }
}
