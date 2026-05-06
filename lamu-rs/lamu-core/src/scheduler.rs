//! VRAM Budget Scheduler — bin-packing for GPU model management.
//! Direct port of `lamu/core/scheduler.py`.
//!
//! Uses NVML (nvidia-ml-rs) instead of nvidia-smi subprocess.

use crate::types::{LoadedModel, ModelEntry, ModelState, VramBudget};
use crate::{Error, Result};
use nvml_wrapper::Nvml;
use std::collections::HashMap;
use std::time::Instant;

const VRAM_RESERVED_MB: u32 = 1500;

pub struct VramScheduler {
    reserved_mb: u32,
    loaded: HashMap<String, LoadedModel>,
    total_mb: u32,
    nvml: Option<Nvml>,
}

impl VramScheduler {
    pub fn new() -> Self {
        let nvml = Nvml::init().ok();
        let total_mb = nvml.as_ref()
            .and_then(|n| n.device_by_index(0).ok())
            .and_then(|d| d.memory_info().ok())
            .map(|info| (info.total / (1024 * 1024)) as u32)
            .unwrap_or(0);

        Self {
            reserved_mb: VRAM_RESERVED_MB,
            loaded: HashMap::new(),
            total_mb,
            nvml,
        }
    }

    pub fn total_mb(&self) -> u32 { self.total_mb }

    pub fn available_mb(&self) -> u32 {
        // Use the bigger of (sum of scheduler-registered models) vs
        // (actual NVML-reported usage). Otherwise an orphan llama-server
        // that lamu didn't spawn would get reported as free VRAM, and
        // plan_load() would hand out a model that doesn't fit.
        let registered: u32 = self.loaded.values().map(|m| m.vram_actual_mb).sum();
        let (actual_used, _) = self.query_vram();
        let used = registered.max(actual_used);
        self.total_mb.saturating_sub(used).saturating_sub(self.reserved_mb)
    }

    /// True when NVML is reachable and reports a non-zero total. Mirrors
    /// the Python `gpu_available` property.
    pub fn gpu_available(&self) -> bool {
        self.nvml.is_some() && self.total_mb > 0
    }

    /// Human-readable reason when the GPU is in unavailable state. None
    /// means healthy. Mirrors Python `gpu_unavailable_reason`.
    pub fn gpu_unavailable_reason(&self) -> Option<&str> {
        if self.gpu_available() {
            None
        } else if self.nvml.is_none() {
            Some("nvml unavailable (driver missing or no CUDA device)")
        } else {
            Some("nvml reports total_mb=0 (no device 0?)")
        }
    }

    /// Query current VRAM usage from NVML. Returns (used_mb, total_mb).
    pub fn query_vram(&self) -> (u32, u32) {
        let Some(nvml) = &self.nvml else {
            return (0, 0);
        };
        let Ok(device) = nvml.device_by_index(0) else {
            return (0, 0);
        };
        let Ok(info) = device.memory_info() else {
            return (0, 0);
        };
        let used = (info.used / (1024 * 1024)) as u32;
        let total = (info.total / (1024 * 1024)) as u32;
        (used, total)
    }

    /// Query GPU processes. Returns [(pid, used_mb), ...].
    pub fn query_gpu_pids(&self) -> Vec<(u32, u32)> {
        let Some(nvml) = &self.nvml else {
            return vec![];
        };
        let Ok(device) = nvml.device_by_index(0) else {
            return vec![];
        };
        let Ok(procs) = device.running_compute_processes() else {
            return vec![];
        };
        procs.into_iter().filter_map(|p| {
            let mem = match p.used_gpu_memory {
                nvml_wrapper::enums::device::UsedGpuMemory::Used(b) => Some((b / (1024 * 1024)) as u32),
                _ => None,
            }?;
            Some((p.pid, mem))
        }).collect()
    }

    pub fn budget(&self) -> VramBudget {
        let (used_mb, total_mb) = self.query_vram();
        let loaded_pairs: Vec<(String, u32)> = self.loaded.iter()
            .map(|(name, m)| (name.clone(), m.vram_actual_mb))
            .collect();
        VramBudget {
            total_mb,
            used_mb,
            free_mb: total_mb.saturating_sub(used_mb),
            loaded_models: loaded_pairs,
            available_mb: self.available_mb(),
        }
    }

    pub fn register_loaded(
        &mut self,
        entry: ModelEntry,
        pid: Option<u32>,
        port: u16,
        vram_actual_mb: u32,
    ) -> &LoadedModel {
        let model = LoadedModel {
            entry: entry.clone(),
            state: ModelState::Loaded,
            pid,
            port,
            vram_actual_mb,
            last_used: Instant::now(),
        };
        self.loaded.insert(entry.name.clone(), model);
        self.loaded.get(&entry.name).expect("just inserted")
    }

    pub fn mark_used(&mut self, name: &str) {
        if let Some(m) = self.loaded.get_mut(name) {
            m.last_used = Instant::now();
        }
    }

    pub fn is_loaded(&self, name: &str) -> bool {
        matches!(
            self.loaded.get(name).map(|m| m.state),
            Some(ModelState::Loaded)
        )
    }

    pub fn get_loaded(&self, name: &str) -> Option<&LoadedModel> {
        self.loaded.get(name)
    }

    pub fn loaded_models(&self) -> Vec<&LoadedModel> {
        self.loaded.values().collect()
    }

    pub fn can_fit(&self, entry: &ModelEntry) -> bool {
        entry.vram_mb <= self.available_mb()
    }

    /// Determine which models to evict to free `needed_mb`.
    /// Returns model names in LRU order (oldest first), skips pinned.
    pub fn plan_eviction(&self, needed_mb: u32) -> Vec<String> {
        if needed_mb == 0 {
            return vec![];
        }

        let mut evictable: Vec<(&String, &LoadedModel)> = self.loaded.iter()
            .filter(|(_, m)| !m.entry.pinned && m.state == ModelState::Loaded)
            .collect();
        evictable.sort_by_key(|(_, m)| m.last_used);

        let mut to_evict = vec![];
        let mut freed: u32 = 0;
        for (name, m) in evictable {
            to_evict.push(name.clone());
            freed = freed.saturating_add(m.vram_actual_mb);
            if freed >= needed_mb {
                return to_evict;
            }
        }

        // Can't free enough
        vec![]
    }

    /// Plan loading. Returns (can_load, models_to_evict).
    pub fn plan_load(&self, entry: &ModelEntry) -> (bool, Vec<String>) {
        if self.is_loaded(&entry.name) {
            return (true, vec![]);
        }
        if self.can_fit(entry) {
            return (true, vec![]);
        }

        let deficit = entry.vram_mb.saturating_sub(self.available_mb());
        let to_evict = self.plan_eviction(deficit);
        if to_evict.is_empty() {
            return (false, vec![]);
        }
        (true, to_evict)
    }

    pub fn mark_unloaded(&mut self, name: &str) {
        self.loaded.remove(name);
    }

    pub fn mark_loading(&mut self, entry: ModelEntry) {
        let model = LoadedModel {
            entry: entry.clone(),
            state: ModelState::Loading,
            pid: None,
            port: 0,
            vram_actual_mb: entry.vram_mb,
            last_used: Instant::now(),
        };
        self.loaded.insert(entry.name, model);
    }

    pub fn confirm_loaded(
        &mut self,
        name: &str,
        pid: u32,
        port: u16,
        vram_actual_mb: u32,
    ) -> Result<()> {
        let m = self.loaded.get_mut(name)
            .ok_or_else(|| Error::ModelNotFound(name.to_string()))?;
        m.state = ModelState::Loaded;
        m.pid = Some(pid);
        m.port = port;
        m.vram_actual_mb = vram_actual_mb;
        m.last_used = Instant::now();
        Ok(())
    }
}

impl Default for VramScheduler {
    fn default() -> Self { Self::new() }
}
