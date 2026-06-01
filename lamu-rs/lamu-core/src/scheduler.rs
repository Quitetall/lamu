//! VRAM Budget Scheduler — per-device bin-packing for GPU model management.
//!
//! Multi-GPU (ADR 0017): the scheduler holds a `Vec<DeviceBudget>`. A
//! single-GPU rig is a one-element Vec, and every scalar method below is an
//! **aggregate facade** that reduces to the prior single-pool behavior
//! byte-for-byte. Device selection honors `LAMU_GPU_INDEX` (pin one) /
//! `LAMU_GPU_INDICES` (subset), else all visible devices. Placement-aware
//! physical load (threading the chosen device into the backend spawn) is the
//! next phase; P1 establishes the per-device bookkeeping + best-fit placement.
//!
//! Uses NVML (nvml-wrapper) instead of an nvidia-smi subprocess.

use crate::types::{DevicePlacement, DeviceVram, LoadedModel, ModelEntry, ModelState, VramBudget};
use crate::{Error, Result};
use nvml_wrapper::Nvml;
use std::collections::HashMap;
use std::time::Instant;

const VRAM_RESERVED_MB: u32 = 1500;

/// One GPU: NVML index, total VRAM, per-context reserve, and the models placed
/// on it. The reserve is charged **per device** because CUDA driver/context
/// overhead is incurred once per device (intentionally slightly more
/// conservative than charging it once globally).
pub struct DeviceBudget {
    pub index: u32,
    pub name: String,
    pub total_mb: u32,
    pub reserved_mb: u32,
    pub loaded: HashMap<String, LoadedModel>,
}

impl DeviceBudget {
    fn registered_mb(&self) -> u32 {
        self.loaded
            .values()
            .map(|m| m.vram_actual_mb)
            .fold(0u32, |a, v| a.saturating_add(v))
    }
}

pub struct VramScheduler {
    devices: Vec<DeviceBudget>,
    nvml: Option<Nvml>,
}

impl VramScheduler {
    pub fn new() -> Self {
        let nvml = Nvml::init().ok();
        let devices = Self::enumerate_devices(nvml.as_ref());
        Self { devices, nvml }
    }

    /// Which NVML indices to manage: `LAMU_GPU_INDEX` pins exactly one;
    /// `LAMU_GPU_INDICES` is a comma list; else every visible device. No NVML
    /// → empty (a `set_*_for_tests` helper fills it for unit tests).
    fn enumerate_devices(nvml: Option<&Nvml>) -> Vec<DeviceBudget> {
        let Some(nvml) = nvml else {
            return Vec::new();
        };
        let count = nvml.device_count().unwrap_or(0);
        let mut indices: Vec<u32> = if std::env::var("LAMU_GPU_INDEX").is_ok() {
            // m14: an unparseable LAMU_GPU_INDEX must NOT yield an empty device
            // list (scheduler manages 0 GPUs → refuses every load) while the
            // backends fall back to crate::config::gpu_index()'s default of 0 and
            // run on device 0. Use the SAME validated parse so both agree.
            vec![crate::config::gpu_index()]
        } else if let Ok(list) = std::env::var("LAMU_GPU_INDICES") {
            list.split(',').filter_map(|s| s.trim().parse::<u32>().ok()).collect()
        } else {
            (0..count).collect()
        };
        // m8: dedup so `LAMU_GPU_INDICES=0,0` doesn't build two DeviceBudgets for
        // one physical card (which would double-count its VRAM and over-commit it).
        {
            let mut seen = std::collections::HashSet::new();
            indices.retain(|i| seen.insert(*i));
        }
        indices
            .into_iter()
            .filter(|i| *i < count)
            .filter_map(|i| {
                let d = nvml.device_by_index(i).ok()?;
                let total_mb = d
                    .memory_info()
                    .ok()
                    .map(|m| (m.total / (1024 * 1024)) as u32)
                    .unwrap_or(0);
                let name = d.name().ok().unwrap_or_else(|| format!("gpu{i}"));
                Some(DeviceBudget {
                    index: i,
                    name,
                    total_mb,
                    reserved_mb: VRAM_RESERVED_MB,
                    loaded: HashMap::new(),
                })
            })
            .collect()
    }

    // ── per-device internals ─────────────────────────────────────────────

    fn nvml_used_mb(&self, index: u32) -> u32 {
        self.nvml
            .as_ref()
            .and_then(|n| n.device_by_index(index).ok())
            .and_then(|d| d.memory_info().ok())
            .map(|m| (m.used / (1024 * 1024)) as u32)
            .unwrap_or(0)
    }

    fn nvml_pids(&self, index: u32) -> Vec<(u32, u32)> {
        let Some(nvml) = &self.nvml else {
            return vec![];
        };
        let Ok(d) = nvml.device_by_index(index) else {
            return vec![];
        };
        let Ok(procs) = d.running_compute_processes() else {
            return vec![];
        };
        procs
            .into_iter()
            .filter_map(|p| {
                let mem = match p.used_gpu_memory {
                    nvml_wrapper::enums::device::UsedGpuMemory::Used(b) => Some((b / (1024 * 1024)) as u32),
                    _ => None,
                }?;
                Some((p.pid, mem))
            })
            .collect()
    }

    /// Free VRAM on one device: total − max(registered-here, NVML-actual-here)
    /// − reserve. The `max` guard means an orphan server (or a peer process)
    /// can't be handed out as free VRAM.
    fn device_available(&self, dev: &DeviceBudget) -> u32 {
        let used = dev.registered_mb().max(self.nvml_used_mb(dev.index));
        dev.total_mb.saturating_sub(used).saturating_sub(dev.reserved_mb)
    }

    fn device_holding(&self, name: &str) -> Option<usize> {
        self.devices.iter().position(|d| d.loaded.contains_key(name))
    }

    /// Device with the most available VRAM that fits `need_mb` (best-fit);
    /// None if no single device fits.
    fn best_fit(&self, need_mb: u32) -> Option<usize> {
        self.devices
            .iter()
            .enumerate()
            .map(|(i, d)| (i, self.device_available(d)))
            .filter(|(_, avail)| *avail >= need_mb)
            .max_by_key(|(_, avail)| *avail)
            .map(|(i, _)| i)
    }

    fn most_available(&self) -> Option<usize> {
        self.devices
            .iter()
            .enumerate()
            .max_by_key(|(_, d)| self.device_available(d))
            .map(|(i, _)| i)
    }

    /// Where a NEW model goes: best-fit, else most-available, else device 0.
    /// `register_loaded`/`mark_loading` record reality so they never reject —
    /// they always land on *some* device.
    fn placement_for(&self, need_mb: u32) -> usize {
        if self.devices.is_empty() {
            return 0;
        }
        self.best_fit(need_mb).or_else(|| self.most_available()).unwrap_or(0)
    }

    fn ensure_one_device(&mut self) {
        if self.devices.is_empty() {
            self.devices.push(DeviceBudget {
                index: 0,
                name: "gpu0".into(),
                total_mb: 0,
                reserved_mb: VRAM_RESERVED_MB,
                loaded: HashMap::new(),
            });
        }
    }

    // ── scalar facades (single-GPU → identical to the old single pool) ────

    pub fn total_mb(&self) -> u32 {
        self.devices.iter().map(|d| d.total_mb).fold(0u32, |a, v| a.saturating_add(v))
    }

    /// First device's product name (feeds the cookbook bandwidth lookup). On a
    /// single-GPU rig this is the one card; multi-GPU per-device names live in
    /// `budget().per_device`.
    pub fn gpu_name(&self) -> Option<String> {
        self.devices.first().map(|d| d.name.clone())
    }

    pub fn available_mb(&self) -> u32 {
        self.devices.iter().map(|d| self.device_available(d)).fold(0u32, |a, v| a.saturating_add(v))
    }

    pub fn gpu_available(&self) -> bool {
        self.nvml.is_some() && self.total_mb() > 0
    }

    pub fn gpu_unavailable_reason(&self) -> Option<&str> {
        if self.gpu_available() {
            None
        } else if self.nvml.is_none() {
            Some("nvml unavailable (driver missing or no CUDA device)")
        } else {
            Some("nvml reports no usable device VRAM")
        }
    }

    /// Aggregate (used_mb, total_mb) across all managed devices.
    pub fn query_vram(&self) -> (u32, u32) {
        let used = self
            .devices
            .iter()
            .map(|d| self.nvml_used_mb(d.index))
            .fold(0u32, |a, v| a.saturating_add(v));
        (used, self.total_mb())
    }

    /// Union of GPU compute PIDs across all managed devices.
    pub fn query_gpu_pids(&self) -> Vec<(u32, u32)> {
        self.devices.iter().flat_map(|d| self.nvml_pids(d.index)).collect()
    }

    /// GPU PIDs holding VRAM that lamu did NOT spawn (not in any device's
    /// loaded set). DIAGNOSTIC ONLY — lamu never kills them (a legit non-lamu
    /// GPU job is indistinguishable from a leak).
    pub fn orphan_pids(&self) -> Vec<(u32, u32)> {
        let mine: std::collections::HashSet<u32> = self
            .devices
            .iter()
            .flat_map(|d| d.loaded.values().filter_map(|m| m.pid))
            .collect();
        self.query_gpu_pids().into_iter().filter(|(pid, _)| !mine.contains(pid)).collect()
    }

    pub fn budget(&self) -> VramBudget {
        let (used_mb, total_mb) = self.query_vram();
        let loaded_models: Vec<(String, u32)> = self
            .devices
            .iter()
            .flat_map(|d| d.loaded.iter().map(|(n, m)| (n.clone(), m.vram_actual_mb)))
            .collect();
        let per_device = self
            .devices
            .iter()
            .map(|d| DeviceVram {
                index: d.index,
                name: d.name.clone(),
                total_mb: d.total_mb,
                used_mb: d.registered_mb().max(self.nvml_used_mb(d.index)),
                available_mb: self.device_available(d),
            })
            .collect();
        VramBudget {
            total_mb,
            used_mb,
            free_mb: total_mb.saturating_sub(used_mb),
            loaded_models,
            available_mb: self.available_mb(),
            per_device,
        }
    }

    pub fn register_loaded(
        &mut self,
        entry: ModelEntry,
        pid: Option<u32>,
        port: u16,
        vram_actual_mb: u32,
    ) -> &LoadedModel {
        self.ensure_one_device();
        let dev = self.placement_for(vram_actual_mb).min(self.devices.len() - 1);
        let nvml_index = self.devices[dev].index;
        let name = entry.name.clone();
        // m9: drop any prior placement of this model on ANY device before
        // inserting, so re-registering a model whose device changed doesn't
        // leave a stale entry double-counting its VRAM across two devices.
        for d in &mut self.devices {
            d.loaded.remove(&name);
        }
        let model = LoadedModel {
            entry,
            state: ModelState::Loaded,
            pid,
            port,
            vram_actual_mb,
            last_used: Instant::now(),
            device: DevicePlacement::Single(nvml_index),
        };
        self.devices[dev].loaded.insert(name.clone(), model);
        self.devices[dev].loaded.get(&name).expect("just inserted")
    }

    pub fn mark_used(&mut self, name: &str) {
        for d in &mut self.devices {
            if let Some(m) = d.loaded.get_mut(name) {
                m.last_used = Instant::now();
                return;
            }
        }
    }

    pub fn is_loaded(&self, name: &str) -> bool {
        self.devices
            .iter()
            .any(|d| matches!(d.loaded.get(name).map(|m| m.state), Some(ModelState::Loaded)))
    }

    pub fn get_loaded(&self, name: &str) -> Option<&LoadedModel> {
        self.devices.iter().find_map(|d| d.loaded.get(name))
    }

    /// The GPU placement recorded for `name` (ADR 0017 P2), or `None` if
    /// not loaded/loading. The loader reads this between `mark_loading`
    /// and `Backend::load` to thread the chosen NVML index into the spawn
    /// via `Backend::set_device`.
    pub fn placement_of(&self, name: &str) -> Option<DevicePlacement> {
        self.get_loaded(name).map(|m| m.device.clone())
    }

    pub fn loaded_models(&self) -> Vec<&LoadedModel> {
        self.devices.iter().flat_map(|d| d.loaded.values()).collect()
    }

    /// True when some single device can fit `entry` right now (best-fit).
    pub fn can_fit(&self, entry: &ModelEntry) -> bool {
        self.best_fit(entry.vram_mb).is_some()
    }

    /// Determine which models to evict to free `needed_mb`. Modality-tiered
    /// LRU across ALL devices (non-LLM image/tts before LLMs, LRU within tier),
    /// skips pinned. Returns names in eviction order, or empty if it can't free
    /// enough. (P1 evicts globally; placement-aware per-target-device eviction
    /// is P2.)
    pub fn plan_eviction(&self, needed_mb: u32) -> Vec<String> {
        self.plan_eviction_on(None, needed_mb)
    }

    /// Eviction plan to free `needed_mb`, optionally restricted to a single
    /// `device` (M9). `None` = across all devices (the global P1 behavior).
    /// Modality-tiered LRU (non-LLM image/tts before LLMs, LRU within tier),
    /// skips pinned. Returns names in eviction order, or empty if it can't free
    /// enough on the chosen scope.
    fn plan_eviction_on(&self, device: Option<usize>, needed_mb: u32) -> Vec<String> {
        if needed_mb == 0 {
            return vec![];
        }
        let mut evictable: Vec<&LoadedModel> = self
            .devices
            .iter()
            .enumerate()
            .filter(|(i, _)| device.map_or(true, |d| *i == d))
            .flat_map(|(_, d)| d.loaded.values())
            .filter(|m| !m.entry.pinned && m.state == ModelState::Loaded)
            .collect();
        evictable.sort_by_key(|m| (m.entry.modality.is_llm(), m.last_used));

        let mut to_evict = vec![];
        let mut freed: u32 = 0;
        for m in evictable {
            to_evict.push(m.entry.name.clone());
            freed = freed.saturating_add(m.vram_actual_mb);
            if freed >= needed_mb {
                return to_evict;
            }
        }
        vec![]
    }

    /// Plan loading. Returns (can_load, models_to_evict).
    ///
    /// M9: a model must fit on ONE device, so when no device fits as-is we
    /// compute the deficit against the TARGET device (the most-free one), NOT
    /// the aggregate free across devices, and scope eviction to that device.
    /// Using the aggregate (the old behavior) could refuse a load that one
    /// device's eviction would satisfy, or evict on the wrong card. On a single
    /// GPU the target IS the only device and the aggregate equals its free, so
    /// this is byte-identical to before.
    pub fn plan_load(&self, entry: &ModelEntry) -> (bool, Vec<String>) {
        if self.is_loaded(&entry.name) {
            return (true, vec![]);
        }
        if self.can_fit(entry) {
            return (true, vec![]);
        }
        let Some(target) = self.most_available() else {
            return (false, vec![]);
        };
        let target_free = self.device_available(&self.devices[target]);
        let deficit = entry.vram_mb.saturating_sub(target_free);
        let to_evict = self.plan_eviction_on(Some(target), deficit);
        if to_evict.is_empty() {
            return (false, vec![]);
        }
        (true, to_evict)
    }

    pub fn mark_unloaded(&mut self, name: &str) {
        for d in &mut self.devices {
            d.loaded.remove(name);
        }
    }

    pub fn mark_loading(&mut self, entry: ModelEntry) {
        self.ensure_one_device();
        let dev = self.placement_for(entry.vram_mb).min(self.devices.len() - 1);
        let nvml_index = self.devices[dev].index;
        // m9: clear any prior placement on any device first (see register_loaded).
        for d in &mut self.devices {
            d.loaded.remove(&entry.name);
        }
        let model = LoadedModel {
            entry: entry.clone(),
            state: ModelState::Loading,
            pid: None,
            port: 0,
            vram_actual_mb: entry.vram_mb,
            last_used: Instant::now(),
            device: DevicePlacement::Single(nvml_index),
        };
        self.devices[dev].loaded.insert(entry.name, model);
    }

    pub fn confirm_loaded(&mut self, name: &str, pid: u32, port: u16, vram_actual_mb: u32) -> Result<()> {
        let dev = self
            .device_holding(name)
            .ok_or_else(|| Error::ModelNotFound(name.to_string()))?;
        let m = self.devices[dev]
            .loaded
            .get_mut(name)
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
    fn default() -> Self {
        Self::new()
    }
}

impl VramScheduler {
    /// **Test fixture only.** Replace the device pool with a single synthetic
    /// device of `mb` total VRAM and null out NVML, so `available_mb` derives
    /// usage entirely from `register_loaded` state. Production reads NVML at
    /// construction; this exists because dev-machine GPU contention otherwise
    /// leaks into unit tests. `pub` (not `#[cfg(test)]`) because integration
    /// tests under `lamu-core/tests/` build as separate crates.
    pub fn set_total_mb_for_tests(&mut self, mb: u32) {
        self.set_devices_for_tests(&[(mb, "test-gpu")]);
    }

    /// **Test fixture only.** Replace the pool with N synthetic devices
    /// `(total_mb, name)` and null out NVML — for multi-GPU placement/eviction
    /// unit tests on a single-GPU (or no-GPU) box.
    pub fn set_devices_for_tests(&mut self, specs: &[(u32, &str)]) {
        self.nvml = None;
        self.devices = specs
            .iter()
            .enumerate()
            .map(|(i, (total, name))| DeviceBudget {
                index: i as u32,
                name: (*name).to_string(),
                total_mb: *total,
                reserved_mb: VRAM_RESERVED_MB,
                loaded: HashMap::new(),
            })
            .collect();
    }

    /// **Test fixture only.** NVML index of the device currently holding
    /// `name`, for placement assertions.
    pub fn device_of_for_tests(&self, name: &str) -> Option<u32> {
        self.device_holding(name).map(|i| self.devices[i].index)
    }

    /// **Test fixture only.** Number of managed devices.
    pub fn device_count_for_tests(&self) -> usize {
        self.devices.len()
    }
}
