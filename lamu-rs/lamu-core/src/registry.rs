//! Model registry — auto-discover GGUFs, write/read YAML.
//! Direct port of `lamu/core/registry.py`.

use crate::types::{
    BackendType, Capability, ModelEntry, ModelFormat, ReasoningMarker, SpeculativeConfig,
};
use crate::Result;
use byteorder::{LittleEndian, ReadBytesExt};
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

const GGUF_MAGIC: &[u8; 4] = b"GGUF";

static ARCH_CAPABILITIES: Lazy<HashMap<&'static str, Vec<Capability>>> = Lazy::new(|| {
    use Capability::*;
    let mut m = HashMap::new();
    m.insert("qwen35", vec![Chat, Code, Reasoning]);
    m.insert("qwen3", vec![Chat, Code]);
    m.insert("gpt2", vec![Chat]);
    m.insert("phi3", vec![Chat, Code, Reasoning]);
    m.insert("llama", vec![Chat, Code]);
    m.insert("gemma", vec![Chat]);
    m.insert("dflash", vec![Chat, Code, Reasoning]);
    m
});

static ARCH_REASONING: Lazy<HashMap<&'static str, ReasoningMarker>> = Lazy::new(|| {
    let mut m = HashMap::new();
    m.insert("qwen35", ReasoningMarker {
        open_tag: "<think>".to_string(),
        close_tag: "</think>".to_string(),
        family: "qwen35".to_string(),
    });
    m.insert("qwen3", ReasoningMarker {
        open_tag: "<think>".to_string(),
        close_tag: "</think>".to_string(),
        family: "qwen3".to_string(),
    });
    m
});

#[derive(Debug, Default)]
struct GgufMeta {
    arch: String,
    file_type: Option<u32>,
    n_tensors: u64,
    file_size_mb: u32,
}

/// Parse minimal GGUF metadata. Reads only what we need for scan.
fn parse_gguf_meta(path: &Path) -> Result<GgufMeta> {
    let file = File::open(path)?;
    let file_size_mb = (file.metadata()?.len() / (1024 * 1024)) as u32;
    let mut r = BufReader::new(file);

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != GGUF_MAGIC {
        return Ok(GgufMeta { file_size_mb, ..Default::default() });
    }

    let _version = r.read_u32::<LittleEndian>()?;
    let n_tensors = r.read_u64::<LittleEndian>()?;
    let n_kv = r.read_u64::<LittleEndian>()?;

    let mut meta = GgufMeta {
        n_tensors,
        file_size_mb,
        ..Default::default()
    };

    // Per-string allocation cap. GGUF headers in practice contain
    // strings of a few hundred bytes max (model names, arch tags). 4 MiB
    // is far above realistic and well below DoS-via-OOM territory.
    const MAX_STR_BYTES: u64 = 4 * 1024 * 1024;
    // Sanity cap on integer offsets we hand to SeekFrom::Current. i64::MAX
    // / 4 leaves headroom for `arr_len * 4` style multiplications below.
    const MAX_SEEK: u64 = (i64::MAX as u64) / 4;

    let read_capped_string = |r: &mut BufReader<File>, len: u64| -> Result<Vec<u8>> {
        if len > MAX_STR_BYTES {
            return Err(crate::Error::Backend(format!(
                "GGUF string length {} exceeds {} byte cap",
                len, MAX_STR_BYTES
            )));
        }
        let mut buf = vec![0u8; len as usize];
        r.read_exact(&mut buf)?;
        Ok(buf)
    };

    // Parse KV pairs (cap to 100 to avoid huge reads)
    let max_kv = std::cmp::min(n_kv, 100);
    for _ in 0..max_kv {
        let key_len = r.read_u64::<LittleEndian>()?;
        let key_bytes = read_capped_string(&mut r, key_len)?;
        let key = String::from_utf8_lossy(&key_bytes).into_owned();

        let val_type = r.read_u32::<LittleEndian>()?;

        match val_type {
            8 => {
                let s_len = r.read_u64::<LittleEndian>()?;
                let s_bytes = read_capped_string(&mut r, s_len)?;
                let val = String::from_utf8_lossy(&s_bytes).trim_end_matches('\0').to_string();
                if key == "general.architecture" {
                    meta.arch = val.to_lowercase();
                }
            }
            4 => {
                let v = r.read_u32::<LittleEndian>()?;
                if key == "general.file_type" {
                    meta.file_type = Some(v);
                }
            }
            5 => { let _ = r.read_i32::<LittleEndian>()?; }
            6 => { let _ = r.read_f32::<LittleEndian>()?; }
            10 => { let _ = r.read_u64::<LittleEndian>()?; }
            7 => { let mut b = [0u8; 1]; r.read_exact(&mut b)?; }
            9 => {
                let arr_type = r.read_u32::<LittleEndian>()?;
                let arr_len = r.read_u64::<LittleEndian>()?;
                if arr_len > MAX_SEEK {
                    return Err(crate::Error::Backend(format!(
                        "GGUF array length {} exceeds seek cap", arr_len
                    )));
                }
                match arr_type {
                    8 => {
                        let cap = std::cmp::min(arr_len, 5);
                        for _ in 0..cap {
                            let sl = r.read_u64::<LittleEndian>()?;
                            if sl > MAX_SEEK {
                                return Err(crate::Error::Backend(format!(
                                    "GGUF inner string {} exceeds seek cap", sl
                                )));
                            }
                            r.seek(SeekFrom::Current(sl as i64))?;
                        }
                        if arr_len > cap {
                            break;
                        }
                    }
                    4 | 5 => {
                        let bytes = arr_len.checked_mul(4)
                            .ok_or_else(|| crate::Error::Backend("array byte count overflow".into()))?;
                        if bytes > MAX_SEEK {
                            return Err(crate::Error::Backend("GGUF array seek overflow".into()));
                        }
                        r.seek(SeekFrom::Current(bytes as i64))?;
                    }
                    6 => {
                        let bytes = arr_len.checked_mul(4)
                            .ok_or_else(|| crate::Error::Backend("array byte count overflow".into()))?;
                        if bytes > MAX_SEEK {
                            return Err(crate::Error::Backend("GGUF array seek overflow".into()));
                        }
                        r.seek(SeekFrom::Current(bytes as i64))?;
                    }
                    _ => break,
                }
            }
            _ => break,
        }
    }

    Ok(meta)
}

fn detect_quant(meta: &GgufMeta, filename: &str) -> String {
    // GGUF metadata file_type
    if let Some(ft) = meta.file_type {
        let name = match ft {
            0 => Some("F32"),
            1 => Some("F16"),
            2 => Some("Q4_0"),
            7 => Some("Q8_0"),
            14 => Some("Q4_K_S"),
            15 => Some("Q4_K_M"),
            16 => Some("Q5_K_S"),
            17 => Some("Q5_K_M"),
            18 => Some("Q6_K"),
            19 => Some("Q2_K"),
            20 => Some("Q3_K_S"),
            21 => Some("Q3_K_M"),
            _ => None,
        };
        if let Some(n) = name {
            return n.to_string();
        }
    }

    // Fallback: filename
    let fn_upper = filename.to_uppercase();
    let candidates = [
        "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0",
        "Q3_K_M", "Q3_K_S", "Q2_K", "Q4_0", "Q4_1", "Q5_0", "Q5_1",
        "IQ4_NL", "IQ4_XS", "IQ3_XXS", "BF16", "F16", "F32",
    ];
    for c in candidates {
        if fn_upper.contains(c) || fn_upper.contains(&c.replace('_', "-")) {
            return c.to_string();
        }
    }
    "unknown".to_string()
}

fn estimate_params_b(meta: &GgufMeta) -> f32 {
    if meta.n_tensors > 0 && meta.file_size_mb > 0 {
        // Rough: Q4_K_M ≈ 0.6 bytes per param
        let size_bytes = (meta.file_size_mb as f64) * 1024.0 * 1024.0;
        ((size_bytes / 0.6 / 1e9) * 10.0).round() as f32 / 10.0
    } else {
        0.0
    }
}

fn estimate_vram_mb(file_size_mb: u32) -> u32 {
    (file_size_mb as f64 * 1.1) as u32
}

/// Scan directory recursively for .gguf files.
pub fn scan_directory(models_dir: &Path) -> Result<Vec<ModelEntry>> {
    let mut discovered: Vec<ModelEntry> = Vec::new();

    for entry in WalkDir::new(models_dir).into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(ext) = path.extension() else { continue };
        if ext != "gguf" {
            continue;
        }

        let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

        // Skip draft models
        let lower = filename.to_lowercase();
        let path_lower = path.to_string_lossy().to_lowercase();
        if lower.contains("dflash") && path_lower.contains("draft") {
            continue;
        }

        let meta = match parse_gguf_meta(path) {
            Ok(m) => m,
            Err(_) => continue,
        };

        let arch = if meta.arch.is_empty() { "unknown".to_string() } else { meta.arch.clone() };
        let quant = detect_quant(&meta, filename);
        let params_b = estimate_params_b(&meta);
        let vram_mb = estimate_vram_mb(meta.file_size_mb);

        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or(filename);
        let name = stem.to_lowercase().replace(' ', "-");

        let mut capabilities = ARCH_CAPABILITIES
            .get(arch.as_str())
            .cloned()
            .unwrap_or_else(|| vec![Capability::Chat]);

        let context_max = 131072u32;
        if context_max > 65536 {
            capabilities.push(Capability::LongContext);
        }

        let reasoning_marker = ARCH_REASONING.get(arch.as_str()).cloned();

        discovered.push(ModelEntry {
            name,
            path: path.to_path_buf(),
            format: ModelFormat::Gguf,
            backend: BackendType::LlamaCpp,
            arch,
            params_b,
            quant,
            vram_mb,
            context_max,
            capabilities,
            reasoning_marker,
            speculative: None,
            pinned: false,
            notes: String::new(),
            status: crate::types::ModelStatus::default(),
        });
    }

    // Stable sort by name (matches Python's `sorted()`)
    discovered.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(discovered)
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct RegistryFile {
    models: HashMap<String, ModelEntryYaml>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct ModelEntryYaml {
    path: PathBuf,
    format: ModelFormat,
    backend: BackendType,
    arch: String,
    params_b: f32,
    quant: String,
    vram_mb: u32,
    context_max: u32,
    capabilities: Vec<Capability>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    reasoning_marker: Option<ReasoningMarker>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    speculative: Option<SpeculativeConfig>,
    #[serde(default, skip_serializing_if = "is_false")]
    pinned: bool,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    notes: String,
    #[serde(default, deserialize_with = "crate::types::ModelStatus::deserialize_lenient",
            skip_serializing_if = "crate::types::ModelStatus::is_unspecified")]
    status: crate::types::ModelStatus,
}

fn is_false(b: &bool) -> bool { !*b }

impl From<ModelEntry> for ModelEntryYaml {
    fn from(e: ModelEntry) -> Self {
        Self {
            path: e.path,
            format: e.format,
            backend: e.backend,
            arch: e.arch,
            params_b: e.params_b,
            quant: e.quant,
            vram_mb: e.vram_mb,
            context_max: e.context_max,
            capabilities: e.capabilities,
            reasoning_marker: e.reasoning_marker,
            speculative: e.speculative,
            pinned: e.pinned,
            notes: e.notes,
            status: e.status,
        }
    }
}

impl ModelEntryYaml {
    fn into_entry(self, name: String) -> ModelEntry {
        ModelEntry {
            name,
            path: self.path,
            format: self.format,
            backend: self.backend,
            arch: self.arch,
            params_b: self.params_b,
            quant: self.quant,
            vram_mb: self.vram_mb,
            context_max: self.context_max,
            capabilities: self.capabilities,
            reasoning_marker: self.reasoning_marker,
            speculative: self.speculative,
            pinned: self.pinned,
            notes: self.notes,
            status: self.status,
        }
    }
}

pub fn write_registry(models: &[ModelEntry], output: &Path) -> Result<()> {
    let mut models_map: HashMap<String, ModelEntryYaml> = HashMap::new();
    for m in models {
        models_map.insert(m.name.clone(), m.clone().into());
    }
    let registry = RegistryFile { models: models_map };

    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let yaml = serde_yaml::to_string(&registry)?;
    write_atomic(output, yaml.as_bytes())?;
    Ok(())
}

/// Add (or replace, when `replace=true`) a single entry in the
/// registry on disk. Loads the current registry, applies the change,
/// writes it back atomically.
///
/// Returns `Error::Config` if `replace=false` and an entry with the
/// same name already exists — caller has to opt in to overwriting,
/// since trained-model names usually want a unique tag and a silent
/// overwrite would erase the prior run.
///
/// Concurrency note: this is read-modify-write without a file lock.
/// Two simultaneous `add_entry` calls can clobber one another. The
/// expected caller is `lamu-train`, which holds the scheduler
/// lockfile (step 4) for the duration of a job — that's the
/// serialisation point. Don't call this from arbitrary parallel
/// contexts.
pub fn add_entry(entry: ModelEntry, registry_path: &Path, replace: bool) -> Result<()> {
    let mut entries = load_registry(registry_path)?;
    if let Some(idx) = entries.iter().position(|e| e.name == entry.name) {
        if !replace {
            return Err(crate::error::Error::Config(format!(
                "registry already has an entry named '{}'; pass replace=true to overwrite",
                entry.name
            )));
        }
        entries[idx] = entry;
    } else {
        entries.push(entry);
    }
    write_registry(&entries, registry_path)
}

/// Write `bytes` to `dest` atomically: write to a sibling temp file,
/// `fsync` (best effort), then `rename`. `rename` is atomic on the
/// same filesystem, so a crash mid-write leaves either the old file
/// or the new one — never a half-written one.
fn write_atomic(dest: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    let parent = dest.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)?;

    // Temp file lives next to the destination so `rename` stays
    // intra-filesystem (cross-fs rename would lose the atomicity
    // guarantee).
    let tmp = dest.with_file_name(format!(
        ".{}.tmp.{}",
        dest.file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "registry".into()),
        std::process::id()
    ));

    let mut f = std::fs::File::create(&tmp)?;
    f.write_all(bytes)?;
    let _ = f.sync_all(); // best-effort; failure here doesn't justify aborting
    drop(f);

    // Rename overwrites the destination on POSIX.
    if let Err(e) = std::fs::rename(&tmp, dest) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e.into());
    }
    Ok(())
}

pub fn load_registry(path: &Path) -> Result<Vec<ModelEntry>> {
    if !path.exists() {
        return Ok(vec![]);
    }
    let content = std::fs::read_to_string(path)?;
    let registry: RegistryFile = serde_yaml::from_str(&content)?;
    let mut entries: Vec<ModelEntry> = registry
        .models
        .into_iter()
        .map(|(name, e)| e.into_entry(name))
        .collect();
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ModelStatus;

    fn sample_entry(name: &str, path: &Path) -> ModelEntry {
        ModelEntry {
            name: name.into(),
            path: path.into(),
            format: ModelFormat::Gguf,
            backend: BackendType::LlamaCpp,
            arch: "qwen3".into(),
            params_b: 7.0,
            quant: "Q4_K_M".into(),
            vram_mb: 8000,
            context_max: 32768,
            capabilities: vec![Capability::Chat],
            reasoning_marker: None,
            speculative: None,
            pinned: false,
            notes: String::new(),
            status: ModelStatus::default(),
        }
    }

    #[test]
    fn add_entry_to_empty_registry() {
        let dir = tempfile::tempdir().unwrap();
        let reg = dir.path().join("models.yaml");
        let entry = sample_entry("alpha", Path::new("/models/alpha.gguf"));
        add_entry(entry.clone(), &reg, false).expect("add");
        let loaded = load_registry(&reg).expect("load");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].name, "alpha");
        assert_eq!(loaded[0].path, entry.path);
    }

    #[test]
    fn add_entry_appends_alongside_existing() {
        let dir = tempfile::tempdir().unwrap();
        let reg = dir.path().join("models.yaml");
        add_entry(
            sample_entry("alpha", Path::new("/models/a.gguf")),
            &reg,
            false,
        )
        .unwrap();
        add_entry(
            sample_entry("bravo", Path::new("/models/b.gguf")),
            &reg,
            false,
        )
        .unwrap();
        let loaded = load_registry(&reg).unwrap();
        assert_eq!(loaded.len(), 2);
        let names: Vec<&str> = loaded.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, vec!["alpha", "bravo"]);
    }

    #[test]
    fn add_entry_refuses_duplicate_without_replace() {
        let dir = tempfile::tempdir().unwrap();
        let reg = dir.path().join("models.yaml");
        add_entry(
            sample_entry("dup", Path::new("/models/x.gguf")),
            &reg,
            false,
        )
        .unwrap();
        let err = add_entry(
            sample_entry("dup", Path::new("/models/y.gguf")),
            &reg,
            false,
        )
        .expect_err("must refuse duplicate");
        assert!(format!("{err}").contains("dup"));
    }

    #[test]
    fn add_entry_replaces_when_asked() {
        let dir = tempfile::tempdir().unwrap();
        let reg = dir.path().join("models.yaml");
        add_entry(
            sample_entry("dup", Path::new("/models/old.gguf")),
            &reg,
            false,
        )
        .unwrap();
        let mut newer = sample_entry("dup", Path::new("/models/new.gguf"));
        newer.vram_mb = 16000;
        add_entry(newer, &reg, true).expect("replace must succeed");
        let loaded = load_registry(&reg).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].path, PathBuf::from("/models/new.gguf"));
        assert_eq!(loaded[0].vram_mb, 16000);
    }

    #[test]
    fn write_atomic_creates_file_with_payload() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("a/b/c/output.txt");
        write_atomic(&dest, b"hello").expect("write");
        let read = std::fs::read(&dest).unwrap();
        assert_eq!(read, b"hello");
    }

    #[test]
    fn write_atomic_overwrites_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("payload.bin");
        std::fs::write(&dest, b"old contents").unwrap();
        write_atomic(&dest, b"new contents").expect("write");
        let read = std::fs::read(&dest).unwrap();
        assert_eq!(read, b"new contents");
    }

    #[test]
    fn write_atomic_leaves_no_temp_files_on_success() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("payload.bin");
        write_atomic(&dest, b"x").unwrap();
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_name()
                    .to_string_lossy()
                    .contains(".tmp.")
            })
            .collect();
        assert!(leftovers.is_empty(), "tmp files must not survive success");
    }
}
