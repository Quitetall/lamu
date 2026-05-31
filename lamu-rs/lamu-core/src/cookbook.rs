//! Hardware-aware model-fit scoring.
//!
//! Ported from Odysseus `services/hwfit` (`fit.py` + `models.py`) — a roofline
//! tok/s estimate + a 4-factor weighted composite — replacing LAMU's old
//! 3-bucket `fit_bucket` heuristic. Pure arithmetic, no I/O: the caller
//! supplies a [`Hardware`] (from `VramScheduler`) and per-model [`ModelSpec`]s
//! (from registry entries), and gets back ranked [`FitResult`]s.
//!
//! Units are GB internally (matching hwfit, so parity tests transliterate
//! directly); the CLI converts MB↔GB at the boundary. Phase 1 treats every
//! model as DENSE (active == total params); MoE fidelity (Phase 2) adds the
//! active-param split. See docs/decisions/.

// ── tables (ported verbatim from models.py / fit.py) ────────────────────

/// Quant → bytes-per-param for the VRAM estimate (`QUANT_BPP`).
const QUANT_BPP: &[(&str, f32)] = &[
    ("F32", 4.0), ("F16", 2.0), ("BF16", 2.0), ("FP8", 1.0),
    ("Q8_0", 1.05), ("Q6_K", 0.80), ("Q5_K_M", 0.68),
    ("Q4_K_M", 0.58), ("Q4_0", 0.58), ("Q3_K_M", 0.48), ("Q2_K", 0.37),
    ("AWQ-4bit", 0.50), ("AWQ-8bit", 1.0),
    ("GPTQ-Int4", 0.50), ("GPTQ-Int8", 1.0),
    ("mlx-4bit", 0.55), ("mlx-8bit", 1.0), ("mlx-6bit", 0.75),
];
const QUANT_BPP_DEFAULT: f32 = 0.58;

/// Quant → bytes-per-param for the SPEED (roofline) estimate.
const QUANT_BYTES_PER_PARAM: &[(&str, f32)] = &[
    ("F16", 2.0), ("BF16", 2.0), ("FP8", 1.0),
    ("Q8_0", 1.0), ("Q6_K", 0.75), ("Q5_K_M", 0.625),
    ("Q4_K_M", 0.5), ("Q4_0", 0.5), ("Q3_K_M", 0.375), ("Q2_K", 0.25),
    ("AWQ-4bit", 0.5), ("AWQ-8bit", 1.0),
    ("GPTQ-Int4", 0.5), ("GPTQ-Int8", 1.0),
    ("mlx-4bit", 0.5), ("mlx-8bit", 1.0), ("mlx-6bit", 0.75),
];
const QUANT_BYTES_PER_PARAM_DEFAULT: f32 = 0.5;

/// Quant → speed multiplier for the fallback (no-bandwidth) tok/s estimate.
const QUANT_SPEED_MULT: &[(&str, f32)] = &[
    ("F16", 0.6), ("BF16", 0.6), ("FP8", 0.85),
    ("Q8_0", 0.8), ("Q6_K", 0.95), ("Q5_K_M", 1.0),
    ("Q4_K_M", 1.15), ("Q4_0", 1.15), ("Q3_K_M", 1.25), ("Q2_K", 1.35),
    ("AWQ-4bit", 1.2), ("AWQ-8bit", 0.85),
    ("GPTQ-Int4", 1.2), ("GPTQ-Int8", 0.85),
    ("mlx-4bit", 1.15), ("mlx-8bit", 0.85), ("mlx-6bit", 1.0),
];
const QUANT_SPEED_MULT_DEFAULT: f32 = 1.0;

/// Quant → quality penalty (added to the base quality score).
const QUANT_QUALITY_PENALTY: &[(&str, f32)] = &[
    ("F16", 0.0), ("BF16", 0.0), ("FP8", 0.0),
    ("Q8_0", 0.0), ("Q6_K", -1.0), ("Q5_K_M", -2.0),
    ("Q4_K_M", -5.0), ("Q4_0", -5.0), ("Q3_K_M", -8.0), ("Q2_K", -12.0),
    ("AWQ-4bit", -3.0), ("AWQ-8bit", 0.0),
    ("GPTQ-Int4", -3.0), ("GPTQ-Int8", 0.0),
    ("mlx-4bit", -4.0), ("mlx-8bit", 0.0), ("mlx-6bit", -1.0),
];

/// GGUF quant tiers, best quality first — the fallback walk when a target
/// quant doesn't fit.
pub const QUANT_HIERARCHY: &[&str] = &["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"];

/// GPU model substring → memory bandwidth (GB/s). Matched case-insensitively;
/// the LONGEST matching key wins (so "4080 super" beats "4080").
const GPU_BANDWIDTH: &[(&str, u32)] = &[
    ("5090", 1792), ("5080", 960), ("5070 ti", 896), ("5070", 672), ("5060 ti", 448), ("5060", 256),
    ("4090", 1008), ("4080 super", 736), ("4080", 717), ("4070 ti super", 672), ("4070 ti", 504),
    ("4070 super", 504), ("4070", 504), ("4060 ti", 288), ("4060", 272),
    ("3090 ti", 1008), ("3090", 936), ("3080 ti", 912), ("3080", 760), ("3070 ti", 608), ("3070", 448),
    ("3060 ti", 448), ("3060", 360),
    ("2080 ti", 616), ("2080 super", 496), ("2080", 448), ("2070 super", 448), ("2070", 448),
    ("2060 super", 448), ("2060", 336),
    ("1660 ti", 288), ("1660 super", 336), ("1660", 192), ("1650 super", 192), ("1650", 128),
    ("h100 sxm", 3350), ("h100", 2039), ("h200", 4800), ("a100 sxm", 2039), ("a100", 1555),
    ("l40s", 864), ("l40", 864), ("l4", 300), ("a10g", 600), ("a10", 600), ("t4", 320),
    ("v100 sxm", 900), ("v100", 897), ("a6000", 768), ("a5000", 768), ("a4000", 448),
    ("7900 xtx", 960), ("7900 xt", 800), ("7900 gre", 576), ("7800 xt", 624), ("7700 xt", 432), ("7600", 288),
    ("6950 xt", 576), ("6900 xt", 512), ("6800 xt", 512), ("6800", 512), ("6700 xt", 384),
    ("6600 xt", 256), ("6600", 224),
    ("mi300x", 5300), ("mi300", 5300), ("mi250x", 3277), ("mi250", 3277), ("mi210", 1638), ("mi100", 1229),
    ("9070 xt", 624), ("9070", 488),
];

/// use-case → composite weights (quality, speed, fit, context).
const USE_CASE_WEIGHTS: &[(&str, (f32, f32, f32, f32))] = &[
    ("general", (0.45, 0.30, 0.15, 0.10)),
    ("coding", (0.50, 0.20, 0.15, 0.15)),
    ("reasoning", (0.55, 0.15, 0.15, 0.15)),
    ("chat", (0.40, 0.35, 0.15, 0.10)),
    ("multimodal", (0.50, 0.20, 0.15, 0.15)),
    ("embedding", (0.30, 0.40, 0.20, 0.10)),
    ("tts", (0.40, 0.35, 0.15, 0.10)),
    ("stt", (0.40, 0.35, 0.15, 0.10)),
];
const USE_CASE_WEIGHTS_DEFAULT: (f32, f32, f32, f32) = (0.45, 0.30, 0.15, 0.10);

/// use-case → target tok/s for the speed sub-score.
const SPEED_TARGET: &[(&str, f32)] = &[
    ("general", 40.0), ("coding", 40.0), ("multimodal", 40.0), ("chat", 40.0),
    ("reasoning", 25.0), ("embedding", 200.0), ("tts", 40.0), ("stt", 40.0),
];
const SPEED_TARGET_DEFAULT: f32 = 40.0;

/// use-case → target context for the context sub-score.
const CONTEXT_TARGET: &[(&str, u32)] = &[
    ("general", 4096), ("chat", 4096), ("coding", 8192),
    ("reasoning", 8192), ("multimodal", 4096), ("embedding", 512),
    ("tts", 2048), ("stt", 2048),
];
const CONTEXT_TARGET_DEFAULT: u32 = 4096;

/// Compute backend → fallback tok/s constant `k` (when GPU bandwidth unknown).
const FALLBACK_K: &[(Backend, f32)] = &[
    (Backend::Cuda, 220.0), (Backend::Rocm, 180.0),
    (Backend::CpuX86, 70.0), (Backend::CpuArm, 90.0),
];
const FALLBACK_K_DEFAULT: f32 = 70.0;

// ── types ───────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Cuda,
    Rocm,
    CpuX86,
    CpuArm,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunMode {
    Gpu,
    CpuOffload,
    CpuOnly,
    NoFit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FitLevel {
    Perfect,
    Good,
    Marginal,
    TooTight,
}

/// Detected hardware. `avail_ram_gb == 0` means GPU-only (the Phase-1
/// default): models that only "fit" by spilling to system RAM are reported as
/// too-tight rather than recommended.
#[derive(Clone, Debug)]
pub struct Hardware {
    pub gpu_name: Option<String>,
    pub gpu_vram_gb: f32,
    pub avail_ram_gb: f32,
    pub backend: Backend,
}

/// One model's scoring inputs, derived from a registry `ModelEntry`.
#[derive(Clone, Debug)]
pub struct ModelSpec {
    pub name: String,
    pub params_b: f32,
    /// Active params per token. Equal to `params_b` for dense models (Phase 1).
    pub active_params_b: f32,
    pub is_moe: bool,
    pub quant: String,
    pub context_max: u32,
    pub use_case: String,
}

#[derive(Clone, Debug)]
pub struct FitResult {
    pub name: String,
    pub fit_level: FitLevel,
    pub run_mode: RunMode,
    pub quant: String,
    pub context: u32,
    pub required_gb: f32,
    pub tps_est: f32,
    pub score: f32,
    pub quality: f32,
    pub speed: f32,
    pub fit: f32,
    pub context_score: f32,
}

// ── table lookups ─────────────────────────────────────────────────────────

fn lookup<'a, V: Copy>(table: &'a [(&str, V)], key: &str, default: V) -> V {
    table
        .iter()
        .find(|(k, _)| *k == key)
        .map(|(_, v)| *v)
        .unwrap_or(default)
}

fn quant_bpp(q: &str) -> f32 {
    lookup(QUANT_BPP, q, QUANT_BPP_DEFAULT)
}
fn quant_bytes_per_param(q: &str) -> f32 {
    lookup(QUANT_BYTES_PER_PARAM, q, QUANT_BYTES_PER_PARAM_DEFAULT)
}
fn quant_speed_mult(q: &str) -> f32 {
    lookup(QUANT_SPEED_MULT, q, QUANT_SPEED_MULT_DEFAULT)
}
fn quant_quality_penalty(q: &str) -> f32 {
    lookup(QUANT_QUALITY_PENALTY, q, 0.0)
}
fn use_case_weights(uc: &str) -> (f32, f32, f32, f32) {
    lookup(USE_CASE_WEIGHTS, uc, USE_CASE_WEIGHTS_DEFAULT)
}
fn speed_target(uc: &str) -> f32 {
    lookup(SPEED_TARGET, uc, SPEED_TARGET_DEFAULT)
}
fn context_target(uc: &str) -> u32 {
    lookup(CONTEXT_TARGET, uc, CONTEXT_TARGET_DEFAULT)
}
fn fallback_k(backend: Backend) -> f32 {
    FALLBACK_K
        .iter()
        .find(|(b, _)| *b == backend)
        .map(|(_, k)| *k)
        .unwrap_or(FALLBACK_K_DEFAULT)
}

/// Memory bandwidth (GB/s) for a GPU name, by longest case-insensitive
/// substring match. `None` when the name is absent or unrecognized.
fn lookup_bandwidth(gpu_name: Option<&str>) -> Option<u32> {
    let gn = gpu_name?.to_ascii_lowercase();
    let mut best: Option<(&str, u32)> = None;
    for (key, bw) in GPU_BANDWIDTH {
        if gn.contains(key) && best.map(|(bk, _)| key.len() > bk.len()).unwrap_or(true) {
            best = Some((key, *bw));
        }
    }
    best.map(|(_, bw)| bw)
}

// ── scoring (ported from fit.py) ────────────────────────────────────────

/// VRAM (GB) to serve `params_b` at `quant`/`ctx`. All weights resident
/// (incl. all MoE experts); KV cache scales with ACTIVE params.
///
/// `0.000_008` (GB per billion-active-params per token) and the `+0.5` (GB
/// runtime overhead) are hwfit's empirical constants, transcribed verbatim
/// from `models.py::estimate_memory_gb` — not re-derived. The parity tests pin
/// the result against hwfit, so any drift surfaces as a test failure.
fn estimate_mem_gb(params_b: f32, active_params_b: f32, quant: &str, ctx: u32) -> f32 {
    params_b * quant_bpp(quant) + 0.000_008 * active_params_b * ctx as f32 + 0.5
}

/// Roofline tok/s. Uses active params (MoE runs only active experts/token).
fn estimate_tps(spec: &ModelSpec, quant: &str, run_mode: RunMode, hw: &Hardware) -> f32 {
    let pb = spec.active_params_b;
    let bw = lookup_bandwidth(hw.gpu_name.as_deref());
    if let (Some(bw), RunMode::Gpu | RunMode::CpuOffload) = (bw, run_mode) {
        let model_gb = pb * quant_bytes_per_param(quant);
        if model_gb <= 0.0 {
            return 0.0;
        }
        let raw_tps = (bw as f32 / model_gb) * 0.55;
        let mode_factor = match run_mode {
            RunMode::CpuOffload => 0.5,
            _ if spec.is_moe => 0.8,
            _ => 1.0,
        };
        return raw_tps * mode_factor;
    }
    if pb <= 0.0 {
        return 0.0;
    }
    fallback_k(hw.backend) / pb * quant_speed_mult(quant)
}

fn quality_score(params_b: f32, name: &str, quant: &str, model_uc: &str, use_case: &str) -> f32 {
    let mut base: f32 = if params_b < 1.0 {
        30.0
    } else if params_b < 3.0 {
        45.0
    } else if params_b < 7.0 {
        60.0
    } else if params_b < 10.0 {
        75.0
    } else if params_b < 20.0 {
        82.0
    } else if params_b < 40.0 {
        89.0
    } else {
        95.0
    };

    let n = name.to_ascii_lowercase();
    if n.contains("qwen") {
        base += 2.0;
    }
    if n.contains("deepseek") {
        base += 3.0;
    }
    if n.contains("llama") {
        base += 2.0;
    }
    if n.contains("mistral") || n.contains("mixtral") {
        base += 1.0;
    }
    if n.contains("gemma") {
        base += 1.0;
    }

    base += quant_quality_penalty(quant);

    if model_uc == "coding" && use_case == "coding" {
        base += 6.0;
    }
    if model_uc == "reasoning" && use_case == "reasoning" && params_b >= 13.0 {
        base += 5.0;
    }
    if model_uc == "multimodal" && use_case == "multimodal" {
        base += 6.0;
    }

    base.clamp(0.0, 100.0)
}

fn speed_score(tps: f32, use_case: &str) -> f32 {
    ((tps / speed_target(use_case)) * 100.0).clamp(0.0, 100.0)
}

fn fit_score(required: f32, available: f32) -> f32 {
    if required > available || available <= 0.0 {
        return 0.0;
    }
    let ratio = required / available;
    if ratio <= 0.5 {
        60.0 + (ratio / 0.5) * 40.0
    } else if ratio <= 0.8 {
        100.0
    } else if ratio <= 0.9 {
        70.0
    } else {
        50.0
    }
}

fn context_score(ctx: u32, use_case: &str) -> f32 {
    let target = context_target(use_case);
    if ctx >= target {
        100.0
    } else if ctx >= target / 2 {
        70.0
    } else {
        30.0
    }
}

/// Try to fit `quant` at `ctx`, halving context down to 1024 if needed.
/// Returns (run_mode, quant, fit_ctx, mem_gb) or None.
fn try_fit(
    spec: &ModelSpec,
    quant: &str,
    ctx: u32,
    gpu_vram: f32,
    avail_ram: f32,
) -> Option<(RunMode, String, u32, f32)> {
    let mem = estimate_mem_gb(spec.params_b, spec.active_params_b, quant, ctx);
    if gpu_vram > 0.0 && mem <= gpu_vram {
        return Some((RunMode::Gpu, quant.to_string(), ctx, mem));
    }
    if gpu_vram > 0.0 && mem <= avail_ram {
        return Some((RunMode::CpuOffload, quant.to_string(), ctx, mem));
    }
    if gpu_vram <= 0.0 && mem <= avail_ram {
        return Some((RunMode::CpuOnly, quant.to_string(), ctx, mem));
    }
    let mut cur = ctx / 2;
    while cur >= 1024 {
        let mem = estimate_mem_gb(spec.params_b, spec.active_params_b, quant, cur);
        if gpu_vram > 0.0 && mem <= gpu_vram {
            return Some((RunMode::Gpu, quant.to_string(), cur, mem));
        }
        if mem <= avail_ram {
            let rm = if gpu_vram > 0.0 { RunMode::CpuOffload } else { RunMode::CpuOnly };
            return Some((rm, quant.to_string(), cur, mem));
        }
        cur /= 2;
    }
    None
}

/// Score one model against `hw` for `use_case`. `target_quant` overrides the
/// model's native quant (and triggers the lower-quant fallback walk). Returns
/// `None` only for a degenerate (params_b <= 0) entry; a model that doesn't
/// fit is returned with `fit_level == TooTight` and score 0 (so a
/// `--simulate-vram` view can show red rows).
pub fn score_model(spec: &ModelSpec, hw: &Hardware, use_case: &str, target_quant: Option<&str>) -> Option<FitResult> {
    if spec.params_b <= 0.0 {
        return None;
    }
    let ctx = if spec.context_max == 0 { 4096 } else { spec.context_max };
    let quant_to_try = target_quant.unwrap_or(&spec.quant);

    let mut result = try_fit(spec, quant_to_try, ctx, hw.gpu_vram_gb, hw.avail_ram_gb);
    // Target quant didn't fit → walk down the GGUF hierarchy.
    if result.is_none() {
        let start = QUANT_HIERARCHY.iter().position(|q| *q == quant_to_try);
        if let Some(idx) = start {
            for q in &QUANT_HIERARCHY[idx + 1..] {
                result = try_fit(spec, q, ctx, hw.gpu_vram_gb, hw.avail_ram_gb);
                if result.is_some() {
                    break;
                }
            }
        }
    }

    let Some((run_mode, quant, fit_ctx, required_gb)) = result else {
        let required = estimate_mem_gb(spec.params_b, spec.active_params_b, quant_to_try, ctx);
        return Some(FitResult {
            name: spec.name.clone(),
            fit_level: FitLevel::TooTight,
            run_mode: RunMode::NoFit,
            quant: quant_to_try.to_string(),
            context: ctx,
            required_gb: required,
            tps_est: 0.0,
            score: 0.0,
            quality: 0.0,
            speed: 0.0,
            fit: 0.0,
            context_score: 0.0,
        });
    };

    let budget = if run_mode == RunMode::Gpu { hw.gpu_vram_gb } else { hw.avail_ram_gb };
    // LAMU has no per-model `recommended_ram_gb`, so fit_level is ratio-based
    // (hwfit keys off recommended headroom; this is the lean equivalent).
    let fit_level = match run_mode {
        RunMode::Gpu => {
            let ratio = required_gb / hw.gpu_vram_gb;
            if ratio <= 0.7 {
                FitLevel::Perfect
            } else if ratio <= 0.9 {
                FitLevel::Good
            } else {
                FitLevel::Marginal
            }
        }
        RunMode::CpuOffload => {
            if hw.avail_ram_gb >= required_gb * 1.2 {
                FitLevel::Good
            } else {
                FitLevel::Marginal
            }
        }
        _ => FitLevel::Marginal,
    };

    let tps = estimate_tps(spec, &quant, run_mode, hw);
    let q = quality_score(spec.params_b, &spec.name, &quant, &spec.use_case, use_case);
    let s = speed_score(tps, use_case);
    let f = fit_score(required_gb, budget);
    let c = context_score(fit_ctx, use_case);
    let (wq, ws, wf, wc) = use_case_weights(use_case);
    let composite = q * wq + s * ws + f * wf + c * wc;

    Some(FitResult {
        name: spec.name.clone(),
        fit_level,
        run_mode,
        quant,
        context: fit_ctx,
        required_gb,
        tps_est: tps,
        score: composite,
        quality: q,
        speed: s,
        fit: f,
        context_score: c,
    })
}

/// Score + rank a set of models, best composite first.
pub fn rank(specs: &[ModelSpec], hw: &Hardware, use_case: &str, target_quant: Option<&str>) -> Vec<FitResult> {
    let mut out: Vec<FitResult> = specs
        .iter()
        .filter_map(|s| score_model(s, hw, use_case, target_quant))
        .collect();
    out.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rtx4090() -> Hardware {
        Hardware {
            gpu_name: Some("NVIDIA GeForce RTX 4090".into()),
            gpu_vram_gb: 24.0,
            avail_ram_gb: 0.0, // GPU-only
            backend: Backend::Cuda,
        }
    }

    fn dense(name: &str, pb: f32, quant: &str, ctx: u32, uc: &str) -> ModelSpec {
        ModelSpec {
            name: name.into(),
            params_b: pb,
            active_params_b: pb,
            is_moe: false,
            quant: quant.into(),
            context_max: ctx,
            use_case: uc.into(),
        }
    }

    #[test]
    fn bandwidth_longest_match_wins() {
        // "4080 super" must beat "4080".
        assert_eq!(lookup_bandwidth(Some("NVIDIA RTX 4080 SUPER")), Some(736));
        assert_eq!(lookup_bandwidth(Some("NVIDIA RTX 4080")), Some(717));
        assert_eq!(lookup_bandwidth(Some("RTX 4090")), Some(1008));
        assert_eq!(lookup_bandwidth(Some("some unknown gpu")), None);
        assert_eq!(lookup_bandwidth(None), None);
    }

    #[test]
    fn mem_estimate_matches_hwfit_formula() {
        // 7B Q4_K_M @ 8192 ctx: 7*0.58 + 8e-6*7*8192 + 0.5 = 5.0188 GB
        let m = estimate_mem_gb(7.0, 7.0, "Q4_K_M", 8192);
        assert!((m - 5.0188).abs() < 1e-3, "got {m}");
    }

    #[test]
    fn tps_roofline_matches_hwfit() {
        // bw 1008, Q4_K_M bytes/param 0.5, model_gb=3.5 → (1008/3.5)*0.55 = 158.4
        let spec = dense("qwen3-7b", 7.0, "Q4_K_M", 8192, "general");
        let tps = estimate_tps(&spec, "Q4_K_M", RunMode::Gpu, &rtx4090());
        assert!((tps - 158.4).abs() < 0.5, "got {tps}");
    }

    #[test]
    fn composite_parity_7b_general_on_4090() {
        // Hand-computed against fit.py:
        //   quality: base 75 (7<10) +0 +Q4_K_M(-5) = 70 (name "qwen3-7b" has no
        //     "qwen " token? it does contain "qwen" → +2 → 72). Recompute: 72.
        //   speed: tps 158.4 / 40 → clamp 100
        //   fit: required 5.0188 / 24 = 0.2091 → 60 + (0.2091/0.5)*40 = 76.73
        //   context: 8192 >= 4096 → 100
        //   weights general (.45,.30,.15,.10):
        //     72*.45 + 100*.30 + 76.73*.15 + 100*.10 = 32.4+30+11.51+10 = 83.91
        let spec = dense("qwen3-7b", 7.0, "Q4_K_M", 8192, "general");
        let r = score_model(&spec, &rtx4090(), "general", None).unwrap();
        assert_eq!(r.run_mode, RunMode::Gpu);
        assert_eq!(r.fit_level, FitLevel::Perfect); // 5.02/24 = 0.21 <= 0.7
        assert!((r.quality - 72.0).abs() < 1e-3, "quality {}", r.quality);
        assert!((r.speed - 100.0).abs() < 1e-3, "speed {}", r.speed);
        assert!((r.fit - 76.73).abs() < 0.1, "fit {}", r.fit);
        assert!((r.context_score - 100.0).abs() < 1e-3);
        assert!((r.score - 83.91).abs() < 0.1, "score {}", r.score);
    }

    #[test]
    fn oversized_model_is_too_tight_not_dropped() {
        // 235B Q4_K_M cannot fit 24GB even at min ctx → TooTight, score 0, kept.
        let spec = dense("huge-235b", 235.0, "Q4_K_M", 8192, "general");
        let r = score_model(&spec, &rtx4090(), "general", None).unwrap();
        assert_eq!(r.fit_level, FitLevel::TooTight);
        assert_eq!(r.run_mode, RunMode::NoFit);
        assert_eq!(r.score, 0.0);
    }

    #[test]
    fn lower_quant_fallback_when_target_too_big() {
        // A 30B at Q8_0 (~31GB+) won't fit 24GB, but Q4_K_M (~18GB) will → the
        // walk down QUANT_HIERARCHY finds a fitting quant.
        let spec = dense("model-30b", 30.0, "Q8_0", 8192, "general");
        let r = score_model(&spec, &rtx4090(), "general", Some("Q8_0")).unwrap();
        assert_ne!(r.fit_level, FitLevel::TooTight);
        assert_ne!(r.quant, "Q8_0"); // fell back to a smaller quant
    }

    #[test]
    fn rank_orders_by_score_desc() {
        let specs = vec![
            dense("small-1b", 1.0, "Q4_K_M", 4096, "general"),
            dense("mid-13b", 13.0, "Q4_K_M", 8192, "general"),
        ];
        let ranked = rank(&specs, &rtx4090(), "general", None);
        assert_eq!(ranked.len(), 2);
        assert!(ranked[0].score >= ranked[1].score);
    }
}
