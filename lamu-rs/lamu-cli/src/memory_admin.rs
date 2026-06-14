//! `lamu memory` — lifetime-memory maintenance (ADR 0030).
//!
//! `lamu memory reembed` converges the unified store's embedded rows
//! onto the CURRENT embedder identity after a model switch: vector
//! recall is model-filtered, so rows embedded under a previous model
//! (or never embedded — NULL embedding) only surface via the FTS leg
//! until they are re-embedded.
//!
//! DRY-RUN BY DEFAULT (the `lamu clean` convention): without `--yes`
//! you get the per-store stale counts and nothing is written.
//!
//! Embedder chain for THIS command (the CLI process has no MCP server
//! or API state to register a local adapter): `LAMU_EMBED_PROVIDER`
//! override → a RUNNING `lamu serve` at 127.0.0.1:8020 (probed via
//! `/health` + a one-item embed; override the URL with
//! `LAMU_SERVE_URL`) → `OPENAI_API_KEY`. The heavy lifting lives in
//! `lamu_memory::reembed` so tests drive the lib fns, not this binary.

use anyhow::{Context, Result};
use lamu_memory::reembed::{plan, run, StoreSel};

/// Default `lamu serve` base URL probed by the CLI chain.
const DEFAULT_SERVE_URL: &str = "http://127.0.0.1:8020";

/// Batch size for the re-embed loop.
const REEMBED_BATCH: usize = 32;

fn serve_url() -> String {
    std::env::var("LAMU_SERVE_URL")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| DEFAULT_SERVE_URL.to_string())
}

pub async fn cmd_memory_reembed(store: Option<String>, yes: bool) -> Result<()> {
    let sel = StoreSel::parse(store.as_deref())?;

    let url = serve_url();
    let embedder = lamu_memory::embedder::resolve_for_cli(&url)
        .await
        .with_context(|| {
            format!(
                "no embedder reachable — start `lamu serve` with an embedding-capable \
                 registry model (probed {url}), set OPENAI_API_KEY, or pin \
                 LAMU_EMBED_PROVIDER=openai"
            )
        })?;
    let identity = embedder.identity();
    println!(
        "current embedder: {} ({} dims)",
        identity.model, identity.dims
    );

    // Get a pool connection for the dry-run plan query.
    let conn = lamu_memory::store::conn()?;
    let plans = plan(&conn, &identity, sel)?;
    // Release the pool connection before the embed loop.
    drop(conn);

    let mut total_stale = 0u64;
    for p in &plans {
        println!(
            "  {:<9} {} of {} row(s) need re-embedding (NULL embedding or embedding_model != '{}')",
            p.store, p.stale, p.total, identity.model
        );
        total_stale += p.stale;
    }
    if total_stale == 0 {
        println!("nothing to do — every row matches the current embedder.");
        return Ok(());
    }
    if !yes {
        println!("dry-run: nothing re-embedded. Pass --yes to re-embed {total_stale} row(s).");
        return Ok(());
    }

    // reembed::run() takes &Arc<Mutex<Connection>> for its internal
    // lock-release-lock pattern. Open a dedicated connection via the
    // pool-initialized path (pool() has already run migrate) and wrap it.
    let path = lamu_memory::store::lamu_db_path();
    let raw_conn = lamu_memory::store::open_at(&path)?;
    let arc = std::sync::Arc::new(parking_lot::Mutex::new(raw_conn));
    let reports = run(&arc, embedder.as_ref(), sel, REEMBED_BATCH).await?;
    for r in &reports {
        println!("  {:<9} re-embedded {} row(s)", r.store, r.reembedded);
    }
    println!("done — embedding_stores now points at '{}'.", identity.model);
    Ok(())
}
