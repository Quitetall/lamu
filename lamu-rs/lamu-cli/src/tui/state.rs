//! Phase 3.1: AppState — UI state holder for the tui dashboard.
//!
//! Pure data + a few state-mutation methods (sort, filter, cursor
//! movement, save/restore favorites). Render and event loop live in
//! `render.rs` and `mod.rs` respectively.

use lamu_core::config::registry_path;
use lamu_core::registry::load_registry;
use lamu_core::scheduler::VramScheduler;
use lamu_core::types::{ModelEntry, VramBudget};
use ratatui::widgets::ListState;
use std::time::Instant;

use super::{probe_port, which_exists, Harness, InputMode, ModelRef, Mode, SettingAction, SettingFile, SortKey, SourceFilter, HARNESSES};
use crate::cloud_models::{self, CloudModel};
use crate::favorites::Favorites;
use crate::lamu_config::LamuConfig;
use crate::mcp_servers::{self, McpServerEntry};
use crate::theme::Theme;
use anyhow::Result;

#[derive(Debug)]
pub struct AppState {
    pub entries: Vec<ModelEntry>,
    pub list_state: ListState,
    pub vram: VramBudget,
    pub gpu_available: bool,
    pub gpu_reason: Option<String>,
    pub last_refresh: Instant,
    pub status_msg: String,
    pub serve_up: bool,
    pub bifrost_up: bool,
    pub mode: Mode,
    pub launcher_state: ListState,
    pub harness_installed: Vec<bool>,
    pub last_harness: Option<&'static str>,
    pub favorites: Favorites,
    pub model_sort: SortKey,
    pub model_filter: String,
    pub harness_sort: SortKey,
    pub harness_filter: String,
    pub input_mode: InputMode,
    pub quit_confirm: bool,
    pub api_key_input: String,
    /// (env_var_name, model_display_name) for the in-progress ApiKey input.
    pub api_key_for: Option<(String, String)>,
    /// Cached sorted+filtered indices into `entries` / `HARNESSES`.
    /// Recomputed on every state change. The `list_state.selected()`
    /// indexes INTO these vecs, not the underlying slices.
    pub harness_view: Vec<usize>,
    pub mcp_servers: Vec<McpServerEntry>,
    pub mcp_state: ListState,
    pub cloud_models: Vec<CloudModel>,
    pub source_filter: SourceFilter,
    /// Unified view: each row references either a local registry entry
    /// or a cloud entry. Replaces the previous `Vec<usize>` so cloud
    /// models can sit alongside local ones in the same scroll buffer.
    pub model_view: Vec<ModelRef>,
    pub config: LamuConfig,
    pub settings_state: ListState,
    /// Live snapshot of nvidia-smi compute-apps. (pid, mem_mb, name).
    /// Refreshed on every tick so the status pane can show what's
    /// actually eating VRAM — including processes lamu didn't spawn.
    pub gpu_procs: Vec<(u32, u32, String)>,
}

impl AppState {
    pub fn new() -> Result<Self> {
        let entries = load_registry(&registry_path()).unwrap_or_default();
        let scheduler = VramScheduler::new();
        let vram = scheduler.budget();
        let gpu_available = scheduler.gpu_available();
        let gpu_reason = scheduler.gpu_unavailable_reason().map(String::from);

        let mut list_state = ListState::default();
        if !entries.is_empty() {
            list_state.select(Some(0));
        }
        let mut launcher_state = ListState::default();
        launcher_state.select(Some(0));
        let harness_installed = HARNESSES.iter().map(|h| which_exists(h.bin)).collect();
        let favorites = Favorites::load();

        let mcp_servers = mcp_servers::load_servers();
        let mut mcp_state = ListState::default();
        if !mcp_servers.is_empty() {
            mcp_state.select(Some(0));
        }
        let cloud_models = cloud_models::load();
        let config = LamuConfig::load();
        let mut settings_state = ListState::default();
        settings_state.select(Some(0));

        let mut s = Self {
            entries,
            list_state,
            vram,
            gpu_available,
            gpu_reason,
            last_refresh: Instant::now(),
            status_msg: String::new(),
            serve_up: false,
            bifrost_up: false,
            mode: Mode::Dashboard,
            launcher_state,
            harness_installed,
            last_harness: None,
            favorites,
            model_sort: SortKey::Default,
            model_filter: String::new(),
            harness_sort: SortKey::Default,
            harness_filter: String::new(),
            input_mode: InputMode::Normal,
            quit_confirm: false,
            api_key_input: String::new(),
            api_key_for: None,
            harness_view: Vec::new(),
            mcp_servers,
            mcp_state,
            cloud_models,
            source_filter: SourceFilter::All,
            model_view: Vec::new(),
            config,
            settings_state,
            gpu_procs: Vec::new(),
        };
        s.recompute_views();
        s.refresh_gpu_procs();
        Ok(s)
    }

    /// Snapshot the GPU process list. Looks up each PID's command name
    /// from /proc/<pid>/comm so the user can identify the offender.
    pub fn refresh_gpu_procs(&mut self) {
        let scheduler = VramScheduler::new();
        let pairs = scheduler.query_gpu_pids();
        self.gpu_procs = pairs
            .into_iter()
            .map(|(pid, mb)| {
                let name = std::fs::read_to_string(format!("/proc/{pid}/comm"))
                    .map(|s| s.trim().to_string())
                    .unwrap_or_else(|_| "?".into());
                (pid, mb, name)
            })
            .collect();
    }

    pub fn settings_items(&self) -> Vec<(String, SettingAction)> {
        vec![
            (
                format!("Backend URL  [{}]  ({})", self.config.backend_label(), self.config.backend_url),
                SettingAction::CycleBackend,
            ),
            ("Edit lamu config (~/.config/lamu/config.toml)".into(), SettingAction::EditFile(SettingFile::LamuConfig)),
            ("Edit cloud models (~/.config/lamu/cloud-models.yaml)".into(), SettingAction::EditFile(SettingFile::CloudModels)),
            ("Edit local models registry (~/local-llm/config/models.yaml)".into(), SettingAction::EditFile(SettingFile::LocalModels)),
            ("Edit MCP servers (~/.claude.json)".into(), SettingAction::EditFile(SettingFile::McpServers)),
            ("Edit favorites (~/.config/lamu/favorites.json)".into(), SettingAction::EditFile(SettingFile::Favorites)),
            ("Open themes directory (~/.config/lamu/themes/)".into(), SettingAction::EditFile(SettingFile::ThemesDir)),
            ("Install bundled themes to user dir".into(), SettingAction::InstallBundledThemes),
            ("Reset cloud-models.yaml to bundled seed".into(), SettingAction::ResetCloudSeed),
        ]
    }

    pub fn move_settings(&mut self, delta: i32) {
        let n = self.settings_items().len() as i32;
        if n == 0 { return; }
        let cur = self.settings_state.selected().unwrap_or(0) as i32;
        let next = ((cur + delta).rem_euclid(n)) as usize;
        self.settings_state.select(Some(next));
    }

    pub fn move_mcp(&mut self, delta: i32) {
        if self.mcp_servers.is_empty() {
            return;
        }
        let n = self.mcp_servers.len() as i32;
        let cur = self.mcp_state.selected().unwrap_or(0) as i32;
        let next = ((cur + delta).rem_euclid(n)) as usize;
        self.mcp_state.select(Some(next));
    }

    pub fn selected_mcp_idx(&self) -> Option<usize> {
        self.mcp_state.selected()
    }

    /// Test: a model is "deployed" if the scheduler currently lists it
    /// among loaded_models.
    pub fn model_deployed(&self, name: &str) -> bool {
        self.vram.loaded_models.iter().any(|(n, _)| n == name)
    }

    /// Build sorted+filtered index vectors. Favorites pinned at the top
    /// regardless of sort key. Local models render before cloud (when
    /// source_filter = All) — both blocks individually sorted.
    pub fn recompute_views(&mut self) {
        // Local models filter
        let filter = self.model_filter.to_lowercase();
        let mut local_idx: Vec<usize> = (0..self.entries.len())
            .filter(|i| {
                if filter.is_empty() {
                    return true;
                }
                let e = &self.entries[*i];
                if e.name.to_lowercase().contains(&filter)
                    || e.quant.to_lowercase().contains(&filter)
                {
                    return true;
                }
                e.capabilities.iter().any(|c| {
                    let cs = match c {
                        lamu_core::types::Capability::Chat => "chat",
                        lamu_core::types::Capability::Code => "code",
                        lamu_core::types::Capability::Reasoning => "reasoning",
                        lamu_core::types::Capability::Routing => "routing",
                        lamu_core::types::Capability::Vision => "vision",
                        lamu_core::types::Capability::LongContext => "long",
                    };
                    cs.contains(&filter)
                })
            })
            .collect();

        let mut cloud_idx: Vec<usize> = (0..self.cloud_models.len())
            .filter(|i| {
                if filter.is_empty() {
                    return true;
                }
                let m = &self.cloud_models[*i];
                m.name.to_lowercase().contains(&filter)
                    || m.provider.to_lowercase().contains(&filter)
                    || m.notes.to_lowercase().contains(&filter)
            })
            .collect();

        // Source filter
        match self.source_filter {
            SourceFilter::All => {}
            SourceFilter::LocalOnly => cloud_idx.clear(),
            SourceFilter::CloudOnly => local_idx.clear(),
        }

        let sort = self.model_sort;
        let entries = &self.entries;
        let cloud_models = &self.cloud_models;
        let favs = &self.favorites;

        // Sort local
        local_idx.sort_by(|a, b| {
            let fa = favs.has_model(&entries[*a].name);
            let fb = favs.has_model(&entries[*b].name);
            if fa != fb {
                return fb.cmp(&fa);
            }
            match sort {
                // Default: params asc → vram asc → name (smallest/cheapest first)
                SortKey::Default => {
                    let ea = &entries[*a];
                    let eb = &entries[*b];
                    let params_ord = ea.params_b
                        .partial_cmp(&eb.params_b)
                        .unwrap_or(std::cmp::Ordering::Equal);
                    if params_ord != std::cmp::Ordering::Equal { return params_ord; }
                    let vram_ord = ea.vram_mb.cmp(&eb.vram_mb);
                    if vram_ord != std::cmp::Ordering::Equal { return vram_ord; }
                    ea.name.cmp(&eb.name)
                }
                SortKey::Name => entries[*a].name.cmp(&entries[*b].name),
                SortKey::Params => entries[*a]
                    .params_b
                    .partial_cmp(&entries[*b].params_b)
                    .unwrap_or(std::cmp::Ordering::Equal),
                SortKey::Vram => entries[*a].vram_mb.cmp(&entries[*b].vram_mb),
                SortKey::Ctx => entries[*a].context_max.cmp(&entries[*b].context_max),
            }
        });

        // Sort cloud (ctx asc by default — smaller ctx = lighter/cheaper first)
        cloud_idx.sort_by(|a, b| {
            let fa = favs.has_model(&cloud_models[*a].full_id());
            let fb = favs.has_model(&cloud_models[*b].full_id());
            if fa != fb {
                return fb.cmp(&fa);
            }
            match sort {
                SortKey::Default => cloud_models[*a]
                    .context_max
                    .cmp(&cloud_models[*b].context_max),
                SortKey::Name => cloud_models[*a].name.cmp(&cloud_models[*b].name),
                SortKey::Ctx => cloud_models[*a]
                    .context_max
                    .cmp(&cloud_models[*b].context_max),
                // Cloud has no params/vram — fall back to name.
                _ => cloud_models[*a].name.cmp(&cloud_models[*b].name),
            }
        });

        // Merge: local first, then cloud.
        let mut view: Vec<ModelRef> = Vec::with_capacity(local_idx.len() + cloud_idx.len());
        for i in local_idx {
            view.push(ModelRef::Local(i));
        }
        for i in cloud_idx {
            view.push(ModelRef::Cloud(i));
        }
        self.model_view = view;

        // Harnesses
        let h_filter = self.harness_filter.to_lowercase();
        let mut hidx: Vec<usize> = (0..HARNESSES.len())
            .filter(|i| {
                if h_filter.is_empty() {
                    return true;
                }
                let h = &HARNESSES[*i];
                h.name.to_lowercase().contains(&h_filter)
                    || h.bin.to_lowercase().contains(&h_filter)
            })
            .collect();

        let installed = &self.harness_installed;
        hidx.sort_by(|a, b| {
            let fa = favs.has_harness(HARNESSES[*a].name);
            let fb = favs.has_harness(HARNESSES[*b].name);
            if fa != fb {
                return fb.cmp(&fa);
            }
            // Then installed-before-missing.
            let ia = *installed.get(*a).unwrap_or(&false);
            let ib = *installed.get(*b).unwrap_or(&false);
            if ia != ib {
                return ib.cmp(&ia);
            }
            HARNESSES[*a].name.cmp(HARNESSES[*b].name)
        });
        self.harness_view = hidx;

        // Clamp selection to view length.
        if let Some(sel) = self.list_state.selected() {
            if sel >= self.model_view.len() {
                self.list_state.select(if self.model_view.is_empty() { None } else { Some(0) });
            }
        }
        if let Some(sel) = self.launcher_state.selected() {
            if sel >= self.harness_view.len() {
                self.launcher_state.select(if self.harness_view.is_empty() { None } else { Some(0) });
            }
        }
    }

    pub fn refresh(&mut self) {
        let scheduler = VramScheduler::new();
        self.vram = scheduler.budget();
        self.gpu_available = scheduler.gpu_available();
        self.gpu_reason = scheduler.gpu_unavailable_reason().map(String::from);
        self.serve_up = probe_port(8020);
        self.bifrost_up = probe_port(8080);
        self.harness_installed = HARNESSES.iter().map(|h| which_exists(h.bin)).collect();
        self.refresh_gpu_procs();
        self.last_refresh = Instant::now();
    }

    pub fn move_launcher(&mut self, delta: i32) {
        if self.harness_view.is_empty() {
            return;
        }
        let n = self.harness_view.len() as i32;
        let cur = self.launcher_state.selected().unwrap_or(0) as i32;
        let next = ((cur + delta).rem_euclid(n)) as usize;
        self.launcher_state.select(Some(next));
    }

    pub fn selected_harness(&self) -> Option<&'static Harness> {
        self.launcher_state
            .selected()
            .and_then(|i| self.harness_view.get(i).copied())
            .and_then(|orig| HARNESSES.get(orig))
    }

    pub fn selected_harness_orig_idx(&self) -> Option<usize> {
        self.launcher_state
            .selected()
            .and_then(|i| self.harness_view.get(i).copied())
    }

    pub fn move_cursor(&mut self, delta: i32) {
        if self.model_view.is_empty() {
            return;
        }
        let n = self.model_view.len() as i32;
        let cur = self.list_state.selected().unwrap_or(0) as i32;
        let next = ((cur + delta).rem_euclid(n)) as usize;
        self.list_state.select(Some(next));
    }

    pub fn selected_ref(&self) -> Option<ModelRef> {
        self.list_state
            .selected()
            .and_then(|i| self.model_view.get(i).copied())
    }

    pub fn selected_entry(&self) -> Option<&ModelEntry> {
        match self.selected_ref()? {
            ModelRef::Local(i) => self.entries.get(i),
            ModelRef::Cloud(_) => None,
        }
    }

    pub fn selected_cloud(&self) -> Option<&CloudModel> {
        match self.selected_ref()? {
            ModelRef::Cloud(i) => self.cloud_models.get(i),
            ModelRef::Local(_) => None,
        }
    }

    /// Display name for the selected row (local entry name OR cloud full_id).
    pub fn selected_name(&self) -> Option<String> {
        match self.selected_ref()? {
            ModelRef::Local(i) => self.entries.get(i).map(|e| e.name.clone()),
            ModelRef::Cloud(i) => self.cloud_models.get(i).map(|m| m.full_id()),
        }
    }
}
