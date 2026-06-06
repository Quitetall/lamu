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

mod research;

/// Register this module's MCP tool into lamu-core (ADR 0023). Call ONCE at the
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
}

#[cfg(test)]
mod tests {
    #[test]
    fn register_installs_research_tool() {
        super::register();
        assert!(
            lamu_core::tools_ext::find_handler("research").is_some(),
            "register() must install the research tool"
        );
        assert!(
            lamu_core::tools_ext::list_entries().iter().any(|e| e["name"] == "research"),
            "research must appear in tools/list"
        );
    }
}
