//! Request router — capability-based model selection.
//! Direct port of `lamu/core/router.py`.

use crate::health::BackendHealth;
use crate::scheduler::VramScheduler;
use crate::types::{Capability, LoadedModel, ModelEntry, RouteDecision};
use std::collections::{HashMap, HashSet};

pub struct Router {
    registry: HashMap<String, ModelEntry>,
}

impl Router {
    pub fn new(_scheduler: &VramScheduler, registry: Vec<ModelEntry>) -> Self {
        let registry: HashMap<String, ModelEntry> = registry.into_iter()
            .map(|m| (m.name.clone(), m))
            .collect();
        Self { registry }
    }

    pub fn update_registry(&mut self, models: Vec<ModelEntry>) {
        self.registry = models.into_iter().map(|m| (m.name.clone(), m)).collect();
    }

    /// Select best model. Does NOT load — just decides.
    ///
    /// `health_map` lets the caller filter out DEAD/QUARANTINED backends —
    /// when present, an explicit-model request against an unhealthy backend
    /// is refused with `unhealthy:<state>` in the reason; capability-based
    /// routing skips them silently and falls through to the next candidate.
    pub fn route(
        &self,
        scheduler: &VramScheduler,
        model: Option<&str>,
        capabilities: Option<&[Capability]>,
        health_map: Option<&HashMap<String, BackendHealth>>,
    ) -> RouteDecision {
        let is_usable = |name: &str| -> bool {
            match health_map {
                None => true,
                Some(m) => m.get(name).map_or(true, |h| h.usable()),
            }
        };

        // Resolve aliases: "default", "main", "lamu" → the entry flagged
        // with `main: true` in models.yaml. Lets external harnesses pin
        // their config to "model: lamu" and stay agnostic of which model
        // is actually backing the call.
        let resolved_model: Option<String> = model.and_then(|n| {
            let lc = n.to_ascii_lowercase();
            if matches!(lc.as_str(), "default" | "main" | "lamu") {
                // Deterministic when >1 entry sets main:true: HashMap
                // iteration order is randomized per process, so `find`
                // resolved "lamu"/"main"/"default" to different models on
                // different starts. Pick the lowest name. (#22)
                self.registry.values()
                    .filter(|e| e.main)
                    .min_by(|a, b| a.name.cmp(&b.name))
                    .map(|e| e.name.clone())
            } else {
                None
            }
        });
        let model_eff: Option<&str> = resolved_model.as_deref().or(model);

        // Explicit model override
        if let Some(name) = model_eff {
            let Some(entry) = self.find_model(name) else {
                return RouteDecision {
                    model_name: name.to_string(),
                    reason: format!("model '{}' not found in registry", name),
                    loaded: false,
                    would_evict: vec![],
                };
            };

            if !is_usable(&entry.name) {
                let state = health_map
                    .and_then(|m| m.get(&entry.name))
                    .map(|h| format!("{:?}", h.state).to_lowercase())
                    .unwrap_or_else(|| "unhealthy".to_string());
                return RouteDecision {
                    model_name: entry.name.clone(),
                    reason: format!("model '{}' is unhealthy: {}", entry.name, state),
                    loaded: false,
                    would_evict: vec![],
                };
            }

            if scheduler.is_loaded(&entry.name) {
                return RouteDecision {
                    model_name: entry.name.clone(),
                    reason: "explicit model selection (loaded)".to_string(),
                    loaded: true,
                    would_evict: vec![],
                };
            }

            let (can_load, to_evict) = scheduler.plan_load(entry);
            if can_load {
                return RouteDecision {
                    model_name: entry.name.clone(),
                    reason: "explicit model selection (will load)".to_string(),
                    loaded: false,
                    would_evict: to_evict,
                };
            }
            return RouteDecision {
                model_name: entry.name.clone(),
                reason: "explicit model — cannot fit in VRAM even with eviction".to_string(),
                loaded: false,
                would_evict: vec![],
            };
        }

        // Capability-based routing
        let required: HashSet<Capability> = match capabilities {
            Some(c) if !c.is_empty() => c.iter().copied().collect(),
            _ => [Capability::Chat].into_iter().collect(),
        };

        // If model unspecified AND no specific capability set, prefer the
        // operator-designated `main: true` entry (if loaded + healthy).
        // Lets external harnesses (Claude Code, Codex, Cursor, etc.)
        // call /v1/chat/completions with no model field and land on the
        // current main provider deterministically.
        if model.is_none() {
            if let Some(main_entry) = self.registry.values().find(|e| e.main) {
                if is_usable(&main_entry.name) && scheduler.is_loaded(&main_entry.name) {
                    return RouteDecision {
                        model_name: main_entry.name.clone(),
                        reason: "operator-designated main model (loaded)".to_string(),
                        loaded: true,
                        would_evict: vec![],
                    };
                }
            }
        }

        // Try loaded models first (filter unhealthy)
        let loaded_matches: Vec<&LoadedModel> = self
            .find_loaded_matching(scheduler, &required)
            .into_iter()
            .filter(|m| is_usable(&m.entry.name))
            .collect();
        if let Some(best) = self.rank_loaded(loaded_matches) {
            return RouteDecision {
                model_name: best.entry.name.clone(),
                reason: format!("best loaded model matching {:?}", required_vec(&required)),
                loaded: true,
                would_evict: vec![],
            };
        }

        // Find best unloaded candidate (also filter unhealthy)
        let mut candidates: Vec<&ModelEntry> = self
            .find_registry_matching(&required)
            .into_iter()
            .filter(|e| is_usable(&e.name))
            .collect();
        if candidates.is_empty() {
            let suffix = if health_map.is_some() {
                " (after health filtering)"
            } else {
                ""
            };
            return RouteDecision {
                model_name: String::new(),
                reason: format!(
                    "no model in registry has capabilities {:?}{}",
                    required_vec(&required), suffix
                ),
                loaded: false,
                would_evict: vec![],
            };
        }

        // Sort: more capabilities first (negative len), then smaller VRAM
        candidates.sort_by_key(|m| (-(m.capabilities.len() as i32), m.vram_mb));
        let best_entry = candidates[0];

        let (can_load, to_evict) = scheduler.plan_load(best_entry);
        if can_load {
            return RouteDecision {
                model_name: best_entry.name.clone(),
                reason: format!(
                    "best unloaded model matching {:?} (will load)",
                    required_vec(&required)
                ),
                loaded: false,
                would_evict: to_evict,
            };
        }

        RouteDecision {
            model_name: best_entry.name.clone(),
            reason: "matching model found but cannot fit in VRAM".to_string(),
            loaded: false,
            would_evict: vec![],
        }
    }

    fn find_model(&self, name: &str) -> Option<&ModelEntry> {
        if let Some(e) = self.registry.get(name) {
            return Some(e);
        }
        // Partial match
        let matches: Vec<&ModelEntry> = self.registry.iter()
            .filter(|(n, _)| n.contains(name))
            .map(|(_, e)| e)
            .collect();
        if matches.len() == 1 {
            Some(matches[0])
        } else {
            None
        }
    }

    fn find_loaded_matching<'b>(
        &self,
        scheduler: &'b VramScheduler,
        required: &HashSet<Capability>,
    ) -> Vec<&'b LoadedModel> {
        scheduler.loaded_models().into_iter()
            .filter(|loaded| loaded.entry.modality.is_llm()) // never chat-route a tts/image model
            .filter(|loaded| {
                let model_caps: HashSet<Capability> = loaded.entry.capabilities.iter().copied().collect();
                required.is_subset(&model_caps)
            })
            .collect()
    }

    fn find_registry_matching(&self, required: &HashSet<Capability>) -> Vec<&ModelEntry> {
        self.registry.values().filter(|entry| {
            // Non-LLM modalities (tts/image) are never chat-routable — an
            // empty-capabilities tts entry would otherwise match a no-filter
            // request (empty `required` is a subset of everything).
            entry.modality.is_llm()
                && {
                    let model_caps: HashSet<Capability> =
                        entry.capabilities.iter().copied().collect();
                    required.is_subset(&model_caps)
                }
        }).collect()
    }

    fn rank_loaded<'b>(&self, models: Vec<&'b LoadedModel>) -> Option<&'b LoadedModel> {
        // Returns None instead of panicking on empty input. Caller checks
        // emptiness before this is reached, but defensive in case the
        // invariant ever changes.
        models.into_iter().max_by(|a, b| {
            a.entry.params_b.partial_cmp(&b.entry.params_b)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.last_used.cmp(&b.last_used))
        })
    }
}

fn required_vec(set: &HashSet<Capability>) -> Vec<&'static str> {
    let mut v: Vec<&'static str> = set.iter().map(|c| match c {
        Capability::Chat => "chat",
        Capability::Code => "code",
        Capability::Reasoning => "reasoning",
        Capability::Routing => "routing",
        Capability::Vision => "vision",
        Capability::LongContext => "long_context",
        Capability::Embedding => "embedding",
    }).collect();
    v.sort();
    v
}
