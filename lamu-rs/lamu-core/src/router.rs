//! Request router — capability-based model selection.
//! Direct port of `lamu/core/router.py`.

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
    pub fn route(
        &self,
        scheduler: &VramScheduler,
        model: Option<&str>,
        capabilities: Option<&[Capability]>,
    ) -> RouteDecision {
        // Explicit model override
        if let Some(name) = model {
            let Some(entry) = self.find_model(name) else {
                return RouteDecision {
                    model_name: name.to_string(),
                    reason: format!("model '{}' not found in registry", name),
                    loaded: false,
                    would_evict: vec![],
                };
            };

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

        // Try loaded models first
        let loaded_matches = self.find_loaded_matching(scheduler, &required);
        if !loaded_matches.is_empty() {
            let best = self.rank_loaded(loaded_matches);
            return RouteDecision {
                model_name: best.entry.name.clone(),
                reason: format!("best loaded model matching {:?}", required_vec(&required)),
                loaded: true,
                would_evict: vec![],
            };
        }

        // Find best unloaded candidate
        let mut candidates = self.find_registry_matching(&required);
        if candidates.is_empty() {
            return RouteDecision {
                model_name: String::new(),
                reason: format!("no model in registry has capabilities {:?}", required_vec(&required)),
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
            .filter(|loaded| {
                let model_caps: HashSet<Capability> = loaded.entry.capabilities.iter().copied().collect();
                required.is_subset(&model_caps)
            })
            .collect()
    }

    fn find_registry_matching(&self, required: &HashSet<Capability>) -> Vec<&ModelEntry> {
        self.registry.values().filter(|entry| {
            let model_caps: HashSet<Capability> = entry.capabilities.iter().copied().collect();
            required.is_subset(&model_caps)
        }).collect()
    }

    fn rank_loaded<'b>(&self, models: Vec<&'b LoadedModel>) -> &'b LoadedModel {
        models.into_iter()
            .max_by(|a, b| {
                a.entry.params_b.partial_cmp(&b.entry.params_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.last_used.cmp(&b.last_used))
            })
            .expect("non-empty")
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
    }).collect();
    v.sort();
    v
}
