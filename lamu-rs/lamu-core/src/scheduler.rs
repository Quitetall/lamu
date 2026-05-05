//! VRAM Budget Scheduler — bin-packing for GPU model management.
//!
//! Port of `lamu/core/scheduler.py`.
//! TODO: nvidia-ml-rs for GPU queries (replace nvidia-smi subprocess).
//! TODO: LRU eviction with pinned model support.

use crate::types::{LoadedModel, ModelEntry, VramBudget};
use crate::Result;
use std::collections::HashMap;

const VRAM_RESERVED_MB: u32 = 1500;

pub struct VramScheduler {
    reserved_mb: u32,
    loaded: HashMap<String, LoadedModel>,
    total_mb: u32,
}

impl VramScheduler {
    pub fn new() -> Self {
        Self {
            reserved_mb: VRAM_RESERVED_MB,
            loaded: HashMap::new(),
            total_mb: 0,
        }
    }

    pub fn total_mb(&self) -> u32 { self.total_mb }

    pub fn available_mb(&self) -> u32 {
        let used: u32 = self.loaded.values().map(|m| m.vram_actual_mb).sum();
        self.total_mb.saturating_sub(used).saturating_sub(self.reserved_mb)
    }

    pub fn budget(&self) -> VramBudget {
        todo!("port VramScheduler.budget()")
    }

    pub fn register_loaded(&mut self, _entry: ModelEntry, _pid: Option<u32>, _port: u16, _vram_mb: u32) {
        todo!("port register_loaded")
    }

    pub fn is_loaded(&self, name: &str) -> bool {
        self.loaded.contains_key(name)
    }

    pub fn plan_load(&self, _entry: &ModelEntry) -> Result<(bool, Vec<String>)> {
        todo!("port plan_load — returns (can_fit, models_to_evict)")
    }

    pub fn plan_eviction(&self, _needed_mb: u32) -> Vec<String> {
        todo!("port LRU eviction (skips pinned)")
    }

    pub fn mark_used(&mut self, _name: &str) {
        todo!("update last_used timestamp")
    }

    pub fn mark_unloaded(&mut self, _name: &str) {
        todo!("port mark_unloaded")
    }
}

impl Default for VramScheduler {
    fn default() -> Self { Self::new() }
}
