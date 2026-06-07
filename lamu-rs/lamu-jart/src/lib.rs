//! lamu-jart — research MODULE (ADR 0023).
//!
//! Wraps the standalone `jart` crate (Just Another Research Tool) as a backend
//! extension: a `research` MCP tool that fans out jart's scrapers (HuggingFace,
//! PubMed, bioRxiv, Semantic Scholar) and summarizes the results in-process via
//! lamu-core (`ToolCtx::generate`) — no self-HTTP round-trip. Depends on
//! `lamu-core` + `jart`; `lamu-core` does NOT depend on this crate.
//!
//! Frontends (a `lamu research` TUI + the bundled web SPA) drive this tool; they
//! reuse `jart::tui` / `jart::server` with an in-process summarizer.

mod deep_research;
mod research;

/// Register this module's MCP tools into lamu-core (ADR 0023). Call ONCE at the
/// composition root before serving. Idempotent.
pub fn register() {
    lamu_core::tools_ext::register_tool(lamu_core::tools_ext::ModuleTool {
        name: "research",
        description: "Aggregate recent research for a query across HuggingFace papers/models/spaces, PubMed, bioRxiv, and Semantic Scholar (jart's scraper fan-out), then optionally summarize the results in-process via a local or cloud model. Returns papers/repos/spaces + an optional summary.",
        schema_fn: research::schema_research,
        handler: research::dispatch_research,
        // Honors routing mode through ctx.generate; the scrape itself hits public
        // research APIs (no model keys), and a local model keeps it fully local.
        // Flag cloud so local-only refuses only when the chosen summary model is
        // cloud — handled by ctx.generate's own routing, so keep this false: the
        // tool is usable local-only (summarize=false or a local model).
        cloud: false,
    });
    lamu_core::tools_ext::register_tool(lamu_core::tools_ext::ModuleTool {
        name: "deep_research",
        description: "Multi-step research orchestrator: decompose a question into sub-questions, search HuggingFace/PubMed/bioRxiv/Semantic Scholar concurrently per sub-question, merge into an indexed corpus, then synthesize a cited answer (every [N] citation resolves to a real retrieved paper). Returns the corpus, the cited report, the citation→link map, and any failed sources.",
        schema_fn: deep_research::schema_deep_research,
        handler: deep_research::dispatch_deep_research,
        // Default models are cloud (mimo); ctx.generate enforces routing mode.
        cloud: true,
    });
}

#[cfg(test)]
mod tests {
    #[test]
    fn register_installs_research_tools() {
        super::register();
        for name in ["research", "deep_research"] {
            assert!(
                lamu_core::tools_ext::find_handler(name).is_some(),
                "register() must install the {name} tool"
            );
            assert!(
                lamu_core::tools_ext::list_entries().iter().any(|e| e["name"] == name),
                "{name} must appear in tools/list"
            );
        }
    }
}
