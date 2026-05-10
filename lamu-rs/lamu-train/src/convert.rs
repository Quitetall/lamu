//! HuggingFace checkpoint → GGUF → quantized GGUF.
//!
//! Wraps llama.cpp's `convert_hf_to_gguf.py` and `llama-quantize`
//! binaries (located via `lamu_core::config::llama_tool`). Two-step
//! pipeline:
//!
//!   1. `python3 convert_hf_to_gguf.py <checkpoint_dir> --outfile <name.f16.gguf>`
//!   2. `llama-quantize <name.f16.gguf> <name.<quant>.gguf> <quant>`
//!
//! The intermediate f16 file is removed after a successful quantize
//! so disk pressure stays bounded — a 7B f16 is ~14 GB; the Q4_K_M
//! is ~4 GB. Failed conversions leave the f16 in place so the user
//! can inspect / re-run the quantize step manually.
//!
//! `quant == "f16"` skips the quantize step and returns the f16
//! file as the final artifact.

use std::path::{Path, PathBuf};

use tokio::process::Command;

use crate::error::{Result, TrainError};

/// Run the HF → GGUF pipeline. Returns the final on-disk path
/// (quantized GGUF if `quant != "f16"`, otherwise the f16 GGUF).
///
/// The output files land next to `checkpoint_dir`'s parent — i.e.
/// alongside the HF checkpoint folder, not inside it. This keeps
/// the HF and GGUF artifacts at the same directory level so the
/// registry can reference one cleanly while leaving the other on
/// disk for re-quantization.
pub async fn convert_to_gguf(
    checkpoint_dir: &Path,
    name: &str,
    quant: &str,
) -> Result<PathBuf> {
    if !checkpoint_dir.exists() {
        return Err(TrainError::Convert(format!(
            "checkpoint_dir does not exist: {}",
            checkpoint_dir.display()
        )));
    }
    let parent = checkpoint_dir.parent().ok_or_else(|| {
        TrainError::Convert(format!(
            "checkpoint_dir '{}' has no parent",
            checkpoint_dir.display()
        ))
    })?;
    let f16_path = parent.join(format!("{name}.f16.gguf"));

    let convert = lamu_core::config::llama_tool("convert_hf_to_gguf.py")
        .map_err(|e| TrainError::Convert(format!("locate convert_hf_to_gguf.py: {e}")))?;

    let convert_status = Command::new("python3")
        .arg(&convert)
        .arg(checkpoint_dir)
        .arg("--outfile")
        .arg(&f16_path)
        .status()
        .await
        .map_err(|e| {
            TrainError::Convert(format!(
                "spawn python3 {}: {e}",
                convert.display()
            ))
        })?;
    if !convert_status.success() {
        return Err(TrainError::Convert(format!(
            "convert_hf_to_gguf.py exited with {convert_status}"
        )));
    }
    if !f16_path.exists() {
        return Err(TrainError::Convert(format!(
            "convert succeeded but produced no file at {}",
            f16_path.display()
        )));
    }

    if quant == "f16" {
        return Ok(f16_path);
    }

    let quantize = lamu_core::config::llama_tool("llama-quantize")
        .map_err(|e| TrainError::Convert(format!("locate llama-quantize: {e}")))?;
    let q_path = parent.join(format!("{name}.{quant}.gguf"));
    let q_status = Command::new(&quantize)
        .arg(&f16_path)
        .arg(&q_path)
        .arg(quant)
        .status()
        .await
        .map_err(|e| {
            TrainError::Convert(format!("spawn {}: {e}", quantize.display()))
        })?;
    if !q_status.success() {
        return Err(TrainError::Convert(format!(
            "llama-quantize exited with {q_status}; \
             intermediate f16 left at {} for manual retry",
            f16_path.display()
        )));
    }
    if !q_path.exists() {
        return Err(TrainError::Convert(format!(
            "quantize succeeded but produced no file at {}",
            q_path.display()
        )));
    }

    // Quant succeeded; reclaim disk by removing the f16 intermediate.
    if let Err(e) = std::fs::remove_file(&f16_path) {
        tracing::warn!(
            "failed to remove intermediate f16 {}: {} (artifact remains usable)",
            f16_path.display(),
            e
        );
    }

    Ok(q_path)
}
