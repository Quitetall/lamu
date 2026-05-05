//! Request router — capability-based model selection with dry-run.
//!
//! Port of `lamu/core/router.py`.
//! Routing semantics: capabilities are REQUIREMENTS, not preferences.
//! Never silently downgrade.

use crate::scheduler::VramScheduler;
use crate::types::{Capability, ModelEntry, RouteDecision};

pub struct Router<'a> {
    scheduler: &'a VramScheduler,
    // TODO: registry as HashMap<String, ModelEntry>
}

impl<'a> Router<'a> {
    pub fn new(scheduler: &'a VramScheduler, _registry: Vec<ModelEntry>) -> Self {
        Self { scheduler }
    }

    /// Select best model. Does NOT load — just decides.
    pub fn route(
        &self,
        _model: Option<&str>,
        _capabilities: Option<&[Capability]>,
    ) -> RouteDecision {
        todo!("port lamu/core/router.py::Router.route")
    }
}
