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

mod answer;
mod chat;
mod deep_research;
mod research;
mod session;
mod web_search;

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
    lamu_core::tools_ext::register_tool(lamu_core::tools_ext::ModuleTool {
        name: "research_chat",
        description: "Ask a follow-up grounded in a deep_research session's studies: retrieves the most relevant papers from the session corpus (embeddings RAG) and answers with [N] citations that resolve to real retrieved papers. Requires a session_id from deep_research.",
        schema_fn: chat::schema_research_chat,
        handler: chat::dispatch_research_chat,
        cloud: true,
    });
    lamu_core::tools_ext::register_tool(lamu_core::tools_ext::ModuleTool {
        name: "web_search",
        description: "General keyless web search via a self-hosted SearXNG instance (metasearch over many engines). Pure retrieval — no model call — so a local model (or the agent) can look facts up instead of answering from memory. Returns title/url/snippet/engine per result. Configurable via $SEARXNG_URL (default http://127.0.0.1:8888).",
        schema_fn: web_search::schema_web_search,
        handler: web_search::dispatch_web_search,
        // Pure retrieval, no model + no provider key — safe under local-only.
        cloud: false,
    });
    lamu_core::tools_ext::register_tool(lamu_core::tools_ext::ModuleTool {
        name: "answer",
        description: "Answer a question with agentic web grounding: the model first decides whether facts need looking up, runs SearXNG searches if so, then answers using ONLY the retrieved sources with [N] citations that resolve to real URLs. Pure-reasoning questions skip the search. Honors routing mode via the chosen model.",
        schema_fn: answer::schema_answer,
        handler: answer::dispatch_answer,
        // May summarize with a cloud model by default (ctx.generate enforces routing).
        cloud: true,
    });
}

#[cfg(test)]
mod tests {
    #[test]
    fn register_installs_research_tools() {
        super::register();
        for name in ["research", "deep_research", "research_chat", "web_search", "answer"] {
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
