//! Hybrid recall — reciprocal-rank fusion of the vector + FTS legs
//! (ADR 0030).
//!
//! The vector leg (model-filtered cosine over `embedding`-bearing rows)
//! and the FTS5 leg (`bm25` over `memories_fts`) each produce a RANKED
//! id list; [`rrf_merge`] fuses them. RRF uses only ranks — the carried
//! scores are ignored — so the two legs' incomparable score scales
//! (cosine vs bm25) never need calibrating.
//!
//! With no embedder available the recency list substitutes for the
//! vector leg, so recall degrades to FTS + recency (strictly better
//! than the pre-0030 recency-only fallback).

use std::collections::HashMap;

/// The standard RRF constant. 60 is the value from the original
/// Cormack/Clarke/Buettcher paper and what every off-the-shelf hybrid
/// search (OpenSearch, Vespa, LanceDB) defaults to.
const RRF_K: f32 = 60.0;

/// Reciprocal-rank fusion. `vector` and `fts` are RANK-ORDERED lists
/// (best first); the attached scores are carried by the callers for
/// their own use and ignored here — only positions matter. Each id's
/// fused score is `Σ 1/(60 + rank)` over the lists it appears in
/// (rank is 1-based). Returns the top-`k` `(id, fused_score)` sorted by
/// fused score descending, ties broken by ascending id (stable across
/// runs and HashMap iteration orders).
pub fn rrf_merge(vector: &[(i64, f32)], fts: &[(i64, f64)], k: usize) -> Vec<(i64, f32)> {
    let mut fused: HashMap<i64, f32> = HashMap::new();
    for (rank, (id, _)) in vector.iter().enumerate() {
        *fused.entry(*id).or_default() += 1.0 / (RRF_K + rank as f32 + 1.0);
    }
    for (rank, (id, _)) in fts.iter().enumerate() {
        *fused.entry(*id).or_default() += 1.0 / (RRF_K + rank as f32 + 1.0);
    }
    let mut out: Vec<(i64, f32)> = fused.into_iter().collect();
    out.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    out.truncate(k);
    out
}

/// Turn a free-text query into a syntax-safe FTS5 MATCH expression.
///
/// FTS5 MATCH parses its argument as a query language — bare quotes,
/// `-`, `*`, parens, and column filters all change meaning or raise
/// syntax errors. We neutralize that: split on whitespace, drop quote
/// characters from each token, drop empties, cap at 12 tokens, wrap
/// each survivor in double quotes (a quoted FTS5 string is literal),
/// and join with OR. `None` when nothing tokenizable remains.
pub(crate) fn sanitize_fts_query(query: &str) -> Option<String> {
    const MAX_TOKENS: usize = 12;
    let tokens: Vec<String> = query
        .split_whitespace()
        .map(|t| t.replace(['"', '\''], ""))
        .filter(|t| !t.is_empty())
        .take(MAX_TOKENS)
        .map(|t| format!("\"{t}\""))
        .collect();
    if tokens.is_empty() {
        None
    } else {
        Some(tokens.join(" OR "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rrf_disjoint_lists_interleave_by_rank() {
        let vector = vec![(1, 0.9_f32), (2, 0.8)];
        let fts = vec![(10, -5.0_f64), (20, -4.0)];
        let merged = rrf_merge(&vector, &fts, 10);
        assert_eq!(merged.len(), 4);
        // Rank-1 entries from both lists have the SAME fused score
        // (1/61) → tie broken by ascending id; same for rank-2 (1/62).
        let ids: Vec<i64> = merged.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids, vec![1, 10, 2, 20]);
        assert!((merged[0].1 - 1.0 / 61.0).abs() < 1e-6);
        assert_eq!(merged[0].1, merged[1].1);
    }

    #[test]
    fn rrf_overlap_boosts_shared_id_to_top() {
        // id 5 is rank-2 in BOTH lists; ids 1 and 9 are rank-1 in one
        // list each. 1/62 + 1/62 > 1/61, so 5 must win.
        let vector = vec![(1, 0.99_f32), (5, 0.5)];
        let fts = vec![(9, -9.0_f64), (5, -1.0)];
        let merged = rrf_merge(&vector, &fts, 10);
        assert_eq!(merged[0].0, 5);
        assert!(merged[0].1 > merged[1].1);
    }

    #[test]
    fn rrf_ties_stable_by_ascending_id() {
        // Two ids at the same rank in different lists → identical score.
        let vector = vec![(42, 1.0_f32)];
        let fts = vec![(7, -1.0_f64)];
        let merged = rrf_merge(&vector, &fts, 10);
        assert_eq!(merged[0].0, 7, "equal score → smaller id first");
        assert_eq!(merged[1].0, 42);
    }

    #[test]
    fn rrf_k_caps_output() {
        let vector: Vec<(i64, f32)> = (1..=8).map(|i| (i, 1.0 / i as f32)).collect();
        let merged = rrf_merge(&vector, &[], 3);
        assert_eq!(merged.len(), 3);
        assert_eq!(merged[0].0, 1);
    }

    #[test]
    fn rrf_empty_inputs() {
        assert!(rrf_merge(&[], &[], 5).is_empty());
        let only_fts = rrf_merge(&[], &[(3, -1.0)], 5);
        assert_eq!(only_fts.len(), 1);
        assert_eq!(only_fts[0].0, 3);
    }

    #[test]
    fn sanitize_wraps_tokens_and_strips_quotes() {
        assert_eq!(
            sanitize_fts_query("hello world").as_deref(),
            Some("\"hello\" OR \"world\"")
        );
        // Quote chars are dropped, not escaped.
        assert_eq!(
            sanitize_fts_query("say \"hi\" o'clock").as_deref(),
            Some("\"say\" OR \"hi\" OR \"oclock\"")
        );
        // FTS5 operators end up inside literal strings — no syntax error.
        assert_eq!(
            sanitize_fts_query("NEAR(foo) -bar col:x *").as_deref(),
            Some("\"NEAR(foo)\" OR \"-bar\" OR \"col:x\" OR \"*\"")
        );
    }

    #[test]
    fn sanitize_caps_at_12_tokens() {
        let q = (0..20).map(|i| format!("t{i}")).collect::<Vec<_>>().join(" ");
        let m = sanitize_fts_query(&q).unwrap();
        assert_eq!(m.matches(" OR ").count(), 11, "12 tokens → 11 ORs");
    }

    #[test]
    fn sanitize_empty_and_quote_only_is_none() {
        assert!(sanitize_fts_query("").is_none());
        assert!(sanitize_fts_query("   ").is_none());
        assert!(sanitize_fts_query("\"\" '' \"").is_none());
    }
}
