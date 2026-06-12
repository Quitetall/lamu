//! ONNX embedding engine — ort Session + HF tokenizer (ADR 0034).
//!
//! Load: ort `Session` from the `.onnx` file (CPU execution provider —
//! ort's default features carry no EP, every GPU provider is opt-in) plus
//! the HuggingFace `tokenizer.json` sidecar that MUST sit next to the
//! model file. Embed: batch-encode (truncating to the tokenizer's
//! declared max, else 512), build `input_ids`/`attention_mask` (and
//! `token_type_ids` only when the session's inputs declare it), run,
//! attention-masked mean-pool the first output, L2-normalize. A model
//! that exports an output literally named `sentence_embedding` (some
//! sentence-transformers exports bake the pooling into the graph) skips
//! the pooling step.
//!
//! Dimensionality is discovered at load via a probe encode of `"x"` —
//! reading it off the real output beats trusting graph metadata, which
//! frequently declares dynamic dims.

use anyhow::{anyhow, bail, Context, Result};
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;
use std::sync::Mutex;
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

/// Which graph output to read and how to turn it into one vector per text.
enum OutputPlan {
    /// `[batch, seq, hidden]` token states — attention-masked mean-pool.
    MeanPool(String),
    /// `[batch, hidden]` already pooled by the graph — take rows as-is.
    Pooled(String),
}

pub struct OnnxEmbedEngine {
    id: String,
    /// ort `Session::run` takes `&mut self`; `EmbedEngine::embed` is `&self`
    /// (the server holds the engine behind an `Arc`). A std Mutex is correct
    /// here — embed already runs inside `spawn_blocking`.
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    /// The session's inputs include `token_type_ids` (BERT-style). Models
    /// without it (e.g. distil exports) must not receive the extra input —
    /// ONNX Runtime rejects unknown input names.
    needs_token_type_ids: bool,
    output: OutputPlan,
    dims: usize,
}

impl OnnxEmbedEngine {
    /// Load the model at `model_path` (a `.onnx` file) and its
    /// `tokenizer.json` sidecar from the same directory.
    pub fn load(model_path: &Path) -> Result<Self> {
        if !model_path.is_file() {
            bail!("onnx model not found: {}", model_path.display());
        }
        let dir = model_path
            .parent()
            .ok_or_else(|| anyhow!("onnx model path has no parent dir: {}", model_path.display()))?;
        let tokenizer_path = dir.join("tokenizer.json");
        if !tokenizer_path.is_file() {
            bail!(
                "missing tokenizer sidecar {} — the onnx backend needs the model's HuggingFace \
                 tokenizer.json next to the .onnx file (export it with the model, or copy it \
                 from the source repo)",
                tokenizer_path.display()
            );
        }
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("load {}: {e}", tokenizer_path.display()))?;
        // Truncate to the tokenizer's declared max when it carries one;
        // otherwise the BERT-family default 512 — an over-long input must
        // become a truncated embedding, never a runtime shape error.
        if tokenizer.get_truncation().is_none() {
            tokenizer
                .with_truncation(Some(TruncationParams { max_length: 512, ..Default::default() }))
                .map_err(|e| anyhow!("set tokenizer truncation: {e}"))?;
        }
        // Batch inputs must be rectangular; pad to the longest in batch.
        // pad_id 0 is safe regardless of the model's real pad token — the
        // attention mask zeroes padded positions out of the mean-pool.
        if tokenizer.get_padding().is_none() {
            tokenizer.with_padding(Some(PaddingParams::default()));
        }

        let session = Session::builder()
            .context("ort session builder")?
            .commit_from_file(model_path)
            .with_context(|| format!("ort load {}", model_path.display()))?;

        // Introspect the graph's input names: require the BERT-family
        // contract (input_ids + attention_mask), feed token_type_ids only
        // if declared. Anything else (CLIP image towers, seq2seq encoders
        // with named branches, …) is out of v1 scope — name what we saw.
        let input_names: Vec<String> =
            session.inputs().iter().map(|o| o.name().to_string()).collect();
        for required in ["input_ids", "attention_mask"] {
            if !input_names.iter().any(|n| n == required) {
                bail!(
                    "onnx model {} does not take '{required}' — v1 supports BERT-family \
                     embedding exports only (session inputs: {input_names:?})",
                    model_path.display()
                );
            }
        }
        let needs_token_type_ids = input_names.iter().any(|n| n == "token_type_ids");

        // Output plan: a graph output literally named `sentence_embedding`
        // is already pooled; otherwise the FIRST output is treated as
        // last_hidden_state-shaped token states and mean-pooled.
        let output_names: Vec<String> =
            session.outputs().iter().map(|o| o.name().to_string()).collect();
        let output = if output_names.iter().any(|n| n == "sentence_embedding") {
            OutputPlan::Pooled("sentence_embedding".to_string())
        } else {
            let first = output_names
                .first()
                .cloned()
                .ok_or_else(|| anyhow!("onnx model declares no outputs"))?;
            OutputPlan::MeanPool(first)
        };

        let id = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("onnx-embed")
            .to_string();

        let mut engine = Self {
            id,
            session: Mutex::new(session),
            tokenizer,
            needs_token_type_ids,
            output,
            dims: 0,
        };
        // Probe: dims comes off a real output, not graph metadata (which
        // usually declares the hidden dim dynamic).
        let probe = engine
            .embed_batch(&["x".to_string()])
            .context("probe embed of \"x\" at load")?;
        engine.dims = probe
            .first()
            .map(|v| v.len())
            .filter(|d| *d > 0)
            .ok_or_else(|| anyhow!("probe embed returned an empty vector"))?;
        Ok(engine)
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Tokenize → run → pool → L2-normalize. One vector per input text.
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow!("tokenize: {e}"))?;
        let batch = encodings.len();
        // with_padding(BatchLongest) makes every encoding the same length.
        let seq = encodings.iter().map(|e| e.len()).max().unwrap_or(0);
        if seq == 0 {
            bail!("tokenizer produced empty encodings");
        }

        let mut input_ids: Vec<i64> = Vec::with_capacity(batch * seq);
        let mut attention_mask: Vec<i64> = Vec::with_capacity(batch * seq);
        let mut token_type_ids: Vec<i64> = Vec::with_capacity(batch * seq);
        // Keep the mask as plain counts too — the pooling below needs it
        // after `input_ids`/`attention_mask` move into the ort tensors.
        let mut mask_rows: Vec<Vec<u32>> = Vec::with_capacity(batch);
        for enc in &encodings {
            if enc.len() != seq {
                bail!(
                    "ragged batch after padding ({} vs {seq}) — tokenizer padding misconfigured",
                    enc.len()
                );
            }
            input_ids.extend(enc.get_ids().iter().map(|&v| v as i64));
            attention_mask.extend(enc.get_attention_mask().iter().map(|&v| v as i64));
            token_type_ids.extend(enc.get_type_ids().iter().map(|&v| v as i64));
            mask_rows.push(enc.get_attention_mask().to_vec());
        }

        let shape = vec![batch as i64, seq as i64];
        let mut inputs = ort::inputs![
            "input_ids" => Tensor::from_array((shape.clone(), input_ids))?,
            "attention_mask" => Tensor::from_array((shape.clone(), attention_mask))?,
        ];
        if self.needs_token_type_ids {
            inputs.push((
                "token_type_ids".into(),
                Tensor::from_array((shape, token_type_ids))?.into(),
            ));
        }

        let mut session = self
            .session
            .lock()
            .map_err(|_| anyhow!("onnx session mutex poisoned"))?;
        let outputs = session.run(inputs).context("ort run")?;

        let mut vectors = match &self.output {
            OutputPlan::Pooled(name) => {
                let value = outputs
                    .get(name.as_str())
                    .ok_or_else(|| anyhow!("output '{name}' missing from run results"))?;
                let arr = value
                    .try_extract_array::<f32>()
                    .with_context(|| format!("extract f32 output '{name}'"))?;
                let arr2 = arr
                    .into_dimensionality::<ndarray::Ix2>()
                    .map_err(|e| anyhow!("output '{name}' is not [batch, hidden]: {e}"))?;
                arr2.rows().into_iter().map(|r| r.to_vec()).collect::<Vec<_>>()
            }
            OutputPlan::MeanPool(name) => {
                let value = outputs
                    .get(name.as_str())
                    .ok_or_else(|| anyhow!("output '{name}' missing from run results"))?;
                let arr = value
                    .try_extract_array::<f32>()
                    .with_context(|| format!("extract f32 output '{name}'"))?;
                let arr3 = arr.into_dimensionality::<ndarray::Ix3>().map_err(|e| {
                    anyhow!("output '{name}' is not [batch, seq, hidden] token states: {e}")
                })?;
                let hidden = arr3.shape()[2];
                let mut pooled = Vec::with_capacity(batch);
                for (b, mask) in mask_rows.iter().enumerate() {
                    let mut acc = vec![0f32; hidden];
                    let mut count = 0f32;
                    for (t, &m) in mask.iter().enumerate() {
                        if m == 0 {
                            continue;
                        }
                        count += 1.0;
                        let row = arr3.slice(ndarray::s![b, t, ..]);
                        for (a, &v) in acc.iter_mut().zip(row.iter()) {
                            *a += v;
                        }
                    }
                    // count can't be 0 in practice: every encoding has at
                    // least one unmasked token and seq==0 bailed above —
                    // max(1.0) just keeps a hostile mask from dividing by 0.
                    let denom = count.max(1.0);
                    acc.iter_mut().for_each(|a| *a /= denom);
                    pooled.push(acc);
                }
                pooled
            }
        };

        for v in &mut vectors {
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                v.iter_mut().for_each(|x| *x /= norm);
            }
        }
        Ok(vectors)
    }
}

impl lamu_inproc::EmbedEngine for OnnxEmbedEngine {
    fn id(&self) -> &str {
        self.id()
    }

    fn dims(&self) -> usize {
        self.dims()
    }

    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.embed_batch(texts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real-model engine test, fixture-gated: there is no .onnx in the
    /// repo, so this skips cleanly unless `LAMU_TEST_ONNX_MODEL` points at
    /// a directory containing `model.onnx` + `tokenizer.json`, e.g.:
    ///   LAMU_TEST_ONNX_MODEL=~/models/bge-small-en-v1.5 cargo test -p lamu-onnx
    /// (Fetch one with: `huggingface-cli download BAAI/bge-small-en-v1.5
    /// onnx/model.onnx tokenizer.json`.)
    #[test]
    fn engine_embeds_with_real_fixture() {
        let Ok(dir) = std::env::var("LAMU_TEST_ONNX_MODEL") else {
            eprintln!(
                "SKIP engine_embeds_with_real_fixture: set LAMU_TEST_ONNX_MODEL=<dir with model.onnx + tokenizer.json>"
            );
            return;
        };
        let model = Path::new(&dir).join("model.onnx");
        let engine = OnnxEmbedEngine::load(&model).expect("engine load");
        assert!(engine.dims() > 0, "dims discovered from probe");

        let out = engine
            .embed_batch(&[
                "hello world".to_string(),
                "an entirely different sentence".to_string(),
            ])
            .expect("embed");
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), engine.dims());
        for v in &out {
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-3, "L2-normalized, got norm {norm}");
        }
        // Identical input → identical embedding; different input → not.
        let again = engine.embed_batch(&["hello world".to_string()]).unwrap();
        assert_eq!(out[0], again[0]);
        assert_ne!(out[0], out[1]);
    }

    #[test]
    fn load_missing_model_is_a_clear_error() {
        let Err(e) = OnnxEmbedEngine::load(Path::new("/nonexistent/model.onnx")) else {
            panic!("load of a nonexistent model must fail");
        };
        assert!(format!("{e}").contains("not found"), "got: {e}");
    }

    #[test]
    fn load_missing_tokenizer_sidecar_names_the_sidecar() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("model.onnx");
        std::fs::write(&model, b"not a real onnx").unwrap();
        let Err(e) = OnnxEmbedEngine::load(&model) else {
            panic!("load without tokenizer.json must fail");
        };
        let msg = format!("{e}");
        assert!(msg.contains("tokenizer.json"), "must name the missing sidecar: {msg}");
    }
}
