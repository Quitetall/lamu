//! Candle text-generation engine (ADR 0035).
//!
//! Loads a HuggingFace safetensors checkpoint directory (config.json +
//! *.safetensors shards + tokenizer.json [+ tokenizer_config.json +
//! generation_config.json]) and serves generation for the Llama / Mistral
//! / Qwen2 families via candle. Weights are MMAPPED, never copied through
//! the heap; on CUDA the VarBuilder uploads straight from the mapping.
//!
//! Concurrency: ONE generation at a time (`Mutex<ArchModel>`), matching
//! lamu-api's per-model RequestQueue (concurrency = 1). The mutex also
//! protects the families' model-internal KV caches (mistral/qwen2 store
//! KV inside the model struct; llama gets a fresh `Cache` per call).
//!
//! 5c (BERT-family embeddings through candle) and the `hf_py` python
//! escape hatch are deliberately OUT of this milestone — embeddings stay
//! on lamu-onnx (ADR 0034), and exotic archs stay unserved until a
//! follow-up ADR. See ADR 0035 for the cut lines.

use anyhow::{anyhow, bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::{llama, mistral, qwen2};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;

use lamu_inproc::ChatRequestIn;

/// The model families this engine serves (config.json `model_type`).
pub const SUPPORTED_MODEL_TYPES: [&str; 3] = ["llama", "mistral", "qwen2"];

enum ArchModel {
    /// Stateless model + per-generation `Cache` (created fresh each call,
    /// which both resets KV and keeps `forward(&self, ...)` re-entrant).
    Llama(llama::Llama),
    /// KV cache lives inside the model — cleared at the top of each call.
    Mistral(mistral::Model),
    Qwen2(qwen2::ModelForCausalLM),
}

impl ArchModel {
    fn name(&self) -> &'static str {
        match self {
            ArchModel::Llama(_) => "llama",
            ArchModel::Mistral(_) => "mistral",
            ArchModel::Qwen2(_) => "qwen2",
        }
    }
}

struct EngineCore {
    model_name: String,
    arch: &'static str,
    model: Mutex<ArchModel>,
    /// Kept for per-generation `llama::Cache` construction; `None` for the
    /// families that own their KV cache.
    llama_cfg: Option<llama::Config>,
    tokenizer: Tokenizer,
    device: Device,
    dtype: DType,
    /// tokenizer_config.json `chat_template` (Jinja source), when present.
    chat_template: Option<String>,
    bos_token: Option<String>,
    eos_token: Option<String>,
    /// All eos candidates from config.json + generation_config.json +
    /// tokenizer_config.json — generation stops on ANY of them.
    eos_ids: Vec<u32>,
    /// `max_position_embeddings` — hard ceiling for prompt + generation.
    context_max: usize,
}

/// Cheaply-cloneable handle (the `ChatEngine` impl moves a clone into
/// `spawn_blocking` for each generation).
#[derive(Clone)]
pub struct CandleEngine {
    core: Arc<EngineCore>,
}

impl CandleEngine {
    /// Load the checkpoint in `model_dir` onto `device`.
    ///
    /// `model_dir` may also be a path to one of the `.safetensors` files
    /// (the registry catalogs the first shard's path) — the parent
    /// directory is used in that case.
    pub fn load(model_dir: &Path, device: Device) -> Result<Self> {
        let dir: PathBuf = if model_dir.is_file() {
            model_dir
                .parent()
                .ok_or_else(|| anyhow!("model path has no parent dir: {}", model_dir.display()))?
                .to_path_buf()
        } else {
            model_dir.to_path_buf()
        };
        if !dir.is_dir() {
            bail!("model directory not found: {}", dir.display());
        }

        let config_path = dir.join("config.json");
        let config: Value = serde_json::from_str(
            &std::fs::read_to_string(&config_path)
                .with_context(|| format!("read {}", config_path.display()))?,
        )
        .with_context(|| format!("parse {}", config_path.display()))?;
        let model_type = config
            .get("model_type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("{} has no model_type", config_path.display()))?
            .to_string();
        // Reject unsupported archs BEFORE touching the (potentially huge)
        // shards — the error should be instant and name the arch.
        if !SUPPORTED_MODEL_TYPES.contains(&model_type.as_str()) {
            bail!(
                "model_type '{model_type}' is not served by lamu-hf (ADR 0035 5a/5b covers \
                 {SUPPORTED_MODEL_TYPES:?}; BERT-embed archs are 5c, everything else awaits the \
                 hf_py escape hatch)"
            );
        }

        // All shards, sorted by name — HF shard names
        // (model-0000N-of-0000M.safetensors) sort into load order, and the
        // VarBuilder resolves tensors across the whole set anyway.
        let mut shards: Vec<PathBuf> = std::fs::read_dir(&dir)
            .with_context(|| format!("read dir {}", dir.display()))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("safetensors"))
            .collect();
        shards.sort();
        if shards.is_empty() {
            bail!("no .safetensors shards in {}", dir.display());
        }

        let tokenizer_path = dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("load {}: {e}", tokenizer_path.display()))?;

        // Optional sidecars.
        let tokenizer_config: Option<Value> = read_json_if_present(&dir.join("tokenizer_config.json"));
        let generation_config: Option<Value> = read_json_if_present(&dir.join("generation_config.json"));

        let chat_template = tokenizer_config
            .as_ref()
            .and_then(|tc| tc.get("chat_template"))
            .and_then(|t| t.as_str())
            .map(String::from);
        let bos_token = tokenizer_config.as_ref().and_then(|tc| token_string(tc.get("bos_token")));
        let eos_token = tokenizer_config.as_ref().and_then(|tc| token_string(tc.get("eos_token")));

        // EOS candidates: config.json, generation_config.json, and the
        // tokenizer_config's eos_token string mapped through the
        // tokenizer. Stop on ANY — model families disagree about which
        // file is authoritative (Llama-3 puts <|eot_id|> only in
        // generation_config; Qwen puts <|im_end|> in all three).
        let mut eos_ids: Vec<u32> = Vec::new();
        push_eos_ids(config.get("eos_token_id"), &mut eos_ids);
        if let Some(gc) = &generation_config {
            push_eos_ids(gc.get("eos_token_id"), &mut eos_ids);
        }
        if let Some(tok) = &eos_token {
            if let Some(id) = tokenizer.token_to_id(tok) {
                eos_ids.push(id);
            }
        }
        eos_ids.sort_unstable();
        eos_ids.dedup();
        if eos_ids.is_empty() {
            tracing::warn!(
                model_dir = %dir.display(),
                "no eos token id found in config/generation_config/tokenizer_config — \
                 generation will only stop on max_tokens"
            );
        }

        let context_max = config
            .get("max_position_embeddings")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(4096);

        // CPU runs f32 (bf16 on CPU is simulated and slow); CUDA honors
        // the checkpoint's torch_dtype (bf16/f16), defaulting to f16.
        let dtype = if device.is_cuda() {
            match config.get("torch_dtype").and_then(|v| v.as_str()) {
                Some("bfloat16") => DType::BF16,
                Some("float32") => DType::F32,
                _ => DType::F16,
            }
        } else {
            DType::F32
        };

        // SAFETY: mmaps the shard files. Sound as long as the files are
        // not truncated/mutated while mapped — model dirs are treated as
        // immutable by the whole loading architecture.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&shards, dtype, &device)? };

        let (model, llama_cfg) = match model_type.as_str() {
            "llama" => {
                let cfg: llama::LlamaConfig = serde_json::from_value(config.clone())
                    .with_context(|| format!("llama config from {}", config_path.display()))?;
                let cfg = cfg.into_config(false); // no flash-attn dep in v1
                let model = llama::Llama::load(vb, &cfg).context("build llama")?;
                (ArchModel::Llama(model), Some(cfg))
            }
            "mistral" => {
                let cfg: mistral::Config = serde_json::from_value(config.clone())
                    .with_context(|| format!("mistral config from {}", config_path.display()))?;
                let model = mistral::Model::new(&cfg, vb).context("build mistral")?;
                (ArchModel::Mistral(model), None)
            }
            "qwen2" => {
                let cfg: qwen2::Config = serde_json::from_value(normalize_qwen2_config(config.clone())?)
                    .with_context(|| format!("qwen2 config from {}", config_path.display()))?;
                let model = qwen2::ModelForCausalLM::new(&cfg, vb).context("build qwen2")?;
                (ArchModel::Qwen2(model), None)
            }
            other => unreachable!("model_type '{other}' passed the SUPPORTED_MODEL_TYPES gate"),
        };

        let model_name = dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("hf-candle")
            .to_string();
        let arch = model.name();

        tracing::info!(
            model = %model_name,
            arch,
            shards = shards.len(),
            context_max,
            ?dtype,
            eos = ?eos_ids,
            chat_template = chat_template.is_some(),
            "candle engine loaded"
        );

        Ok(Self {
            core: Arc::new(EngineCore {
                model_name,
                arch,
                model: Mutex::new(model),
                llama_cfg,
                tokenizer,
                device,
                dtype,
                chat_template,
                bos_token,
                eos_token,
                eos_ids,
                context_max,
            }),
        })
    }

    pub fn model_name(&self) -> &str {
        &self.core.model_name
    }

    pub fn arch(&self) -> &'static str {
        self.core.arch
    }

    pub fn context_max(&self) -> usize {
        self.core.context_max
    }

    /// Engine-tokenizer token count (ADR 0021 engine-truth). Mirrors
    /// llama-server's `/tokenize` default: no special tokens added.
    pub fn tokenize_count(&self, text: &str) -> Result<usize> {
        let enc = self
            .core
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow!("tokenize: {e}"))?;
        Ok(enc.len())
    }

    /// Render the chat prompt: the model's own chat_template when present
    /// (minijinja, `add_generation_prompt: true`), else hardcoded ChatML.
    /// A template that fails to render falls back to ChatML with a warning
    /// rather than failing the request — a renderable-but-wrong prompt
    /// degrades quality, an error kills the request; templates only fail
    /// here on Jinja features minijinja lacks, which is rare and loud.
    pub fn render_prompt(&self, messages: &[(String, String)]) -> String {
        if let Some(tpl) = &self.core.chat_template {
            match render_chat_template(
                tpl,
                messages,
                self.core.bos_token.as_deref(),
                self.core.eos_token.as_deref(),
            ) {
                Ok(p) => return p,
                Err(e) => tracing::warn!(
                    model = %self.core.model_name,
                    "chat_template render failed ({e:#}) — falling back to ChatML"
                ),
            }
        }
        chatml_prompt(messages)
    }

    /// Blocking generation — run inside `spawn_blocking`. Sends each text
    /// fragment to `tx`; returns `(prompt_tokens, completion_tokens,
    /// finish_reason)`.
    pub fn generate_sync(
        &self,
        req: ChatRequestIn,
        tx: tokio::sync::mpsc::Sender<String>,
    ) -> Result<(usize, usize, String)> {
        let core = &*self.core;
        let prompt = self.render_prompt(&req.messages);

        // `add_special_tokens: false` — chat templates already spell out
        // their special tokens (bos/<|im_start|>/…) literally; letting the
        // tokenizer add another bos would double it.
        let encoding = core
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow!("tokenize prompt: {e}"))?;
        let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
        let prompt_tokens = prompt_ids.len();
        if prompt_tokens == 0 {
            bail!("prompt tokenized to zero tokens");
        }
        if prompt_tokens >= core.context_max {
            bail!(
                "prompt is {prompt_tokens} tokens but the model's maximum context is {} — \
                 request exceeds the context window",
                core.context_max
            );
        }
        let max_new = (req.max_tokens as usize).min(core.context_max - prompt_tokens);

        let mut lp = build_logits_processor(req.temperature, req.top_p, req.top_k);

        // One generation at a time; also guards the in-model KV caches.
        let mut model = core
            .model
            .lock()
            .map_err(|_| anyhow!("candle model mutex poisoned"))?;

        // Fresh KV state per call: llama gets a new external Cache;
        // mistral/qwen2 clear the cache embedded in the model.
        let mut llama_cache = match &mut *model {
            ArchModel::Llama(_) => {
                let cfg = core
                    .llama_cfg
                    .as_ref()
                    .ok_or_else(|| anyhow!("llama model without retained config (bug)"))?;
                Some(llama::Cache::new(true, core.dtype, cfg, &core.device).context("llama kv cache")?)
            }
            ArchModel::Mistral(m) => {
                m.clear_kv_cache();
                None
            }
            ArchModel::Qwen2(m) => {
                m.clear_kv_cache();
                None
            }
        };

        // Single forward closure over the arch dispatch: feed `ids` at
        // `pos`, get back f32 logits for the LAST position as a 1-D tensor.
        let mut forward = |ids: &[u32], pos: usize| -> Result<Tensor> {
            let input = Tensor::new(ids, &core.device)?.unsqueeze(0)?;
            let logits = match &mut *model {
                ArchModel::Llama(m) => {
                    let cache = llama_cache.as_mut().expect("constructed above");
                    // llama's forward already returns (b, vocab) in f32.
                    m.forward(&input, pos, cache)?.squeeze(0)?
                }
                ArchModel::Mistral(m) => {
                    // (b, 1, vocab) in model dtype.
                    m.forward(&input, pos)?.squeeze(0)?.squeeze(0)?
                }
                ArchModel::Qwen2(m) => m.forward(&input, pos)?.squeeze(0)?.squeeze(0)?,
            };
            Ok(logits.to_dtype(DType::F32)?)
        };

        // Incremental detokenizer: tokenizers' DecodeStream holds the
        // undecodable tail (a UTF-8 char split across tokens) and only
        // releases complete text — no replacement-char fragments on the
        // wire. skip_special_tokens so a non-eos special never leaks.
        let mut decode_stream = core.tokenizer.decode_stream(true);

        let mut logits = forward(&prompt_ids, 0)?;
        let mut pos = prompt_tokens;
        let mut completion_tokens = 0usize;
        let mut finish_reason = "length".to_string();

        for _ in 0..max_new {
            let next = lp.sample(&logits)?;
            if core.eos_ids.contains(&next) {
                finish_reason = "stop".to_string();
                break;
            }
            completion_tokens += 1;
            if let Some(fragment) = decode_stream
                .step(next)
                .map_err(|e| anyhow!("detokenize: {e}"))?
            {
                if !fragment.is_empty() && tx.blocking_send(fragment).is_err() {
                    // Receiver gone — the HTTP client disconnected. Stop
                    // burning compute; the result goes nowhere anyway.
                    finish_reason = "stop".to_string();
                    break;
                }
            }
            if completion_tokens == max_new {
                break; // skip the final (wasted) forward
            }
            logits = forward(&[next], pos)?;
            pos += 1;
        }

        Ok((prompt_tokens, completion_tokens, finish_reason))
    }
}

#[async_trait::async_trait]
impl lamu_inproc::ChatEngine for CandleEngine {
    fn model(&self) -> &str {
        self.model_name()
    }

    fn tokenize_count(&self, text: &str) -> Result<usize> {
        CandleEngine::tokenize_count(self, text)
    }

    async fn generate(
        &self,
        req: ChatRequestIn,
        tx: tokio::sync::mpsc::Sender<String>,
    ) -> Result<(usize, usize, String)> {
        let engine = self.clone();
        tokio::task::spawn_blocking(move || engine.generate_sync(req, tx))
            .await
            .map_err(|e| anyhow!("generation task panicked: {e}"))?
    }
}

/// temperature ≤ 0 → greedy; top_k/top_p compose via candle's `Sampling`.
fn build_logits_processor(temperature: f32, top_p: Option<f32>, top_k: Option<u32>) -> LogitsProcessor {
    // Seeded from the clock: lamu has no per-request seed plumbing (yet),
    // and a fixed seed would make retries deterministic-identical.
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(42);
    let temperature = temperature as f64;
    let sampling = if temperature <= 0.0 {
        Sampling::ArgMax
    } else {
        match (top_k, top_p) {
            (Some(k), Some(p)) => Sampling::TopKThenTopP { k: k as usize, p: p as f64, temperature },
            (Some(k), None) => Sampling::TopK { k: k as usize, temperature },
            (None, Some(p)) => Sampling::TopP { p: p as f64, temperature },
            (None, None) => Sampling::All { temperature },
        }
    };
    LogitsProcessor::from_sampling(seed, sampling)
}

/// Render a HF `chat_template` (Jinja) with minijinja. The environment
/// carries minijinja-contrib's pycompat layer (`.strip()`, `.split()`, …)
/// and HF's `raise_exception` — the two extensions templates in the wild
/// actually use.
pub fn render_chat_template(
    template: &str,
    messages: &[(String, String)],
    bos_token: Option<&str>,
    eos_token: Option<&str>,
) -> Result<String> {
    let mut env = minijinja::Environment::new();
    minijinja_contrib::add_to_environment(&mut env);
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_function("raise_exception", |msg: String| -> std::result::Result<String, minijinja::Error> {
        Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, msg))
    });
    env.add_template("chat", template)
        .context("parse chat_template")?;
    let tmpl = env.get_template("chat").expect("just added");

    let messages_json: Vec<Value> = messages
        .iter()
        .map(|(role, content)| serde_json::json!({"role": role, "content": content}))
        .collect();
    let rendered = tmpl
        .render(minijinja::context! {
            messages => minijinja::Value::from_serialize(&messages_json),
            add_generation_prompt => true,
            bos_token => bos_token.unwrap_or(""),
            eos_token => eos_token.unwrap_or(""),
        })
        .context("render chat_template")?;
    Ok(rendered)
}

/// Hardcoded ChatML — the fallback when a checkpoint ships no template.
/// Qwen-style: every message fenced, generation prompt opened for the
/// assistant.
pub fn chatml_prompt(messages: &[(String, String)]) -> String {
    let mut out = String::new();
    for (role, content) in messages {
        out.push_str("<|im_start|>");
        out.push_str(role);
        out.push('\n');
        out.push_str(content);
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>assistant\n");
    out
}

/// candle's `qwen2::Config` derives plain serde with several mandatory
/// fields that real Qwen2/2.5 config.json files omit or null out. Fill
/// the gaps from the values transformers itself defaults to.
fn normalize_qwen2_config(mut config: Value) -> Result<Value> {
    let obj = config
        .as_object_mut()
        .ok_or_else(|| anyhow!("config.json is not an object"))?;
    let max_pos = obj
        .get("max_position_embeddings")
        .and_then(|v| v.as_u64())
        .unwrap_or(32768);
    let n_layers = obj.get("num_hidden_layers").and_then(|v| v.as_u64()).unwrap_or(0);
    // `"sliding_window": null` (Qwen2.5) or absent → candle wants a usize.
    if obj.get("sliding_window").map(|v| v.is_null()).unwrap_or(true) {
        obj.insert("sliding_window".to_string(), serde_json::json!(max_pos));
    }
    if !obj.contains_key("max_window_layers") {
        obj.insert("max_window_layers".to_string(), serde_json::json!(n_layers));
    }
    if !obj.contains_key("use_sliding_window") {
        obj.insert("use_sliding_window".to_string(), serde_json::json!(false));
    }
    if !obj.contains_key("tie_word_embeddings") {
        obj.insert("tie_word_embeddings".to_string(), serde_json::json!(false));
    }
    Ok(config)
}

fn read_json_if_present(path: &Path) -> Option<Value> {
    let text = std::fs::read_to_string(path).ok()?;
    match serde_json::from_str(&text) {
        Ok(v) => Some(v),
        Err(e) => {
            tracing::warn!("unparseable {}: {e}", path.display());
            None
        }
    }
}

/// HF sidecars write special tokens either as a bare string or as an
/// AddedToken object `{"content": "..."}`.
fn token_string(v: Option<&Value>) -> Option<String> {
    match v {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Object(o)) => o.get("content").and_then(|c| c.as_str()).map(String::from),
        _ => None,
    }
}

/// Collect eos id(s) from a config field that is either one int or a list.
fn push_eos_ids(v: Option<&Value>, out: &mut Vec<u32>) {
    match v {
        Some(Value::Number(n)) => {
            if let Some(i) = n.as_u64() {
                out.push(i as u32);
            }
        }
        Some(Value::Array(items)) => {
            for item in items {
                if let Some(i) = item.as_u64() {
                    out.push(i as u32);
                }
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── chat template rendering ─────────────────────────────────────────

    /// Representative qwen2-style template (trimmed from
    /// Qwen2.5-Instruct's tokenizer_config.json — system default + message
    /// loop + generation prompt).
    const QWEN2_STYLE_TEMPLATE: &str = "{%- if messages[0]['role'] == 'system' -%}\
{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' -}}\
{%- else -%}\
{{- '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' -}}\
{%- endif -%}\
{%- for message in messages -%}\
{%- if message.role != 'system' -%}\
{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' -}}\
{%- endif -%}\
{%- endfor -%}\
{%- if add_generation_prompt -%}\
{{- '<|im_start|>assistant\\n' -}}\
{%- endif -%}";

    fn msgs(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs.iter().map(|(r, c)| (r.to_string(), c.to_string())).collect()
    }

    #[test]
    fn qwen2_style_template_renders_expected_prompt() {
        let m = msgs(&[("system", "Be terse."), ("user", "hi"), ("assistant", "hello"), ("user", "bye?")]);
        let out = render_chat_template(QWEN2_STYLE_TEMPLATE, &m, None, Some("<|im_end|>")).unwrap();
        assert_eq!(
            out,
            "<|im_start|>system\nBe terse.<|im_end|>\n\
             <|im_start|>user\nhi<|im_end|>\n\
             <|im_start|>assistant\nhello<|im_end|>\n\
             <|im_start|>user\nbye?<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    #[test]
    fn template_with_default_system_when_none_given() {
        let m = msgs(&[("user", "hi")]);
        let out = render_chat_template(QWEN2_STYLE_TEMPLATE, &m, None, None).unwrap();
        assert!(out.starts_with("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"));
        assert!(out.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn template_pycompat_strip_works() {
        // Llama-2-era templates call `.strip()` — minijinja-contrib's
        // pycompat layer must be wired in.
        let tpl = "{{ messages[0].content.strip() }}";
        let out = render_chat_template(tpl, &msgs(&[("user", "  padded  ")]), None, None).unwrap();
        assert_eq!(out, "padded");
    }

    #[test]
    fn template_raise_exception_surfaces_as_error() {
        let tpl = "{{ raise_exception('nope') }}";
        let err = render_chat_template(tpl, &msgs(&[("user", "x")]), None, None).unwrap_err();
        assert!(format!("{err:#}").contains("nope"), "got: {err:#}");
    }

    #[test]
    fn chatml_fallback_shape() {
        let out = chatml_prompt(&msgs(&[("system", "s"), ("user", "u")]));
        assert_eq!(
            out,
            "<|im_start|>system\ns<|im_end|>\n<|im_start|>user\nu<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    // ── config normalization + eos collection ──────────────────────────

    #[test]
    fn qwen2_config_null_sliding_window_normalized() {
        let raw = serde_json::json!({
            "model_type": "qwen2",
            "vocab_size": 100, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
            "max_position_embeddings": 2048, "sliding_window": null,
            "rope_theta": 10000.0, "rms_norm_eps": 1e-6, "hidden_act": "silu",
        });
        let v = normalize_qwen2_config(raw).unwrap();
        let cfg: qwen2::Config = serde_json::from_value(v).expect("normalized config deserializes");
        assert_eq!(cfg.sliding_window, 2048);
        assert_eq!(cfg.max_window_layers, 2);
        assert!(!cfg.use_sliding_window);
        assert!(!cfg.tie_word_embeddings);
    }

    #[test]
    fn eos_ids_collected_from_int_and_array() {
        let mut out = Vec::new();
        push_eos_ids(Some(&serde_json::json!(2)), &mut out);
        push_eos_ids(Some(&serde_json::json!([128001, 128009])), &mut out);
        push_eos_ids(None, &mut out);
        assert_eq!(out, vec![2, 128001, 128009]);
    }

    #[test]
    fn token_string_handles_both_shapes() {
        assert_eq!(token_string(Some(&serde_json::json!("</s>"))), Some("</s>".into()));
        assert_eq!(
            token_string(Some(&serde_json::json!({"content": "<|im_end|>"}))),
            Some("<|im_end|>".into())
        );
        assert_eq!(token_string(Some(&serde_json::json!(7))), None);
    }

    // ── incremental detokenization ──────────────────────────────────────

    /// Minimal byte-level BPE tokenizer built inline: enough vocab to
    /// split "é" (0xC3 0xA9) across two tokens, which is exactly the
    /// multi-token-UTF-8 case DecodeStream must buffer.
    fn tiny_tokenizer() -> Tokenizer {
        let json = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
            "post_processor": null,
            "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
            "model": {
                "type": "BPE", "dropout": null, "unk_token": null,
                "continuing_subword_prefix": null, "end_of_word_suffix": null,
                "fuse_unk": false, "byte_fallback": false,
                // GPT-2 byte-encoder forms: 0xC3 → "Ã", 0xA9 → "©".
                "vocab": {"h": 0, "Ã": 1, "©": 2},
                "merges": []
            }
        });
        json.to_string().parse().expect("tiny tokenizer parses")
    }

    #[test]
    fn decode_stream_buffers_split_utf8() {
        let tok = tiny_tokenizer();
        let enc = tok.encode("hé", false).unwrap();
        assert_eq!(enc.get_ids(), &[0, 1, 2], "h + the two bytes of é");

        let mut ds = tok.decode_stream(true);
        assert_eq!(ds.step(0).unwrap().as_deref(), Some("h"));
        // First byte of é is not valid UTF-8 on its own — must buffer.
        assert_eq!(ds.step(1).unwrap(), None);
        // Second byte completes the char.
        assert_eq!(ds.step(2).unwrap().as_deref(), Some("é"));
    }

    // ── full-model test, fixture-gated ──────────────────────────────────

    /// Real-checkpoint engine test: skips cleanly unless
    /// `LAMU_TEST_HF_MODEL` points at a directory holding config.json +
    /// tokenizer.json + *.safetensors of a llama/mistral/qwen2 model, e.g.
    ///   LAMU_TEST_HF_MODEL=~/models/Qwen2.5-0.5B-Instruct \
    ///     cargo test -p lamu-hf -- --nocapture
    /// (Fetch one with: `huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct`.)
    /// CPU device on purpose — tests never touch the GPU (a training job
    /// may hold the scheduler lock).
    #[test]
    fn engine_generates_with_real_fixture() {
        let Ok(dir) = std::env::var("LAMU_TEST_HF_MODEL") else {
            eprintln!(
                "SKIP engine_generates_with_real_fixture: set LAMU_TEST_HF_MODEL=<dir with config.json + tokenizer.json + *.safetensors>"
            );
            return;
        };
        let engine = CandleEngine::load(Path::new(&dir), Device::Cpu).expect("engine load");
        assert!(engine.context_max() > 0);
        assert!(engine.tokenize_count("hello world").unwrap() > 0);

        let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(64);
        let req = ChatRequestIn {
            messages: vec![("user".to_string(), "Say the word 'hello' and nothing else.".to_string())],
            max_tokens: 16,
            temperature: 0.0, // greedy — deterministic across runs
            top_p: None,
            top_k: None,
            stream: false,
        };
        let (prompt_tokens, completion_tokens, finish) =
            engine.generate_sync(req, tx).expect("generate");
        let mut text = String::new();
        while let Ok(frag) = rx.try_recv() {
            text.push_str(&frag);
        }
        eprintln!("fixture output: {text:?} (finish={finish})");
        assert!(prompt_tokens > 0);
        assert!(completion_tokens > 0);
        assert!(!text.is_empty());
        assert!(finish == "stop" || finish == "length");
    }

    #[test]
    fn load_missing_dir_is_clear_error() {
        let Err(err) = CandleEngine::load(Path::new("/nonexistent/model-dir"), Device::Cpu) else {
            panic!("load of a nonexistent dir must fail");
        };
        assert!(format!("{err:#}").contains("not found"), "got: {err:#}");
    }

    #[test]
    fn load_unsupported_model_type_names_the_arch() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            serde_json::json!({"model_type": "bert", "max_position_embeddings": 512}).to_string(),
        )
        .unwrap();
        std::fs::write(dir.path().join("model.safetensors"), b"x").unwrap();
        std::fs::write(dir.path().join("tokenizer.json"), tiny_tokenizer().to_string(false).unwrap()).unwrap();
        let Err(err) = CandleEngine::load(dir.path(), Device::Cpu) else {
            panic!("unsupported model_type must fail");
        };
        let msg = format!("{err:#}");
        assert!(msg.contains("bert"), "must name the arch: {msg}");
        assert!(msg.contains("not served"), "got: {msg}");
    }
}
