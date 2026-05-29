//! Vector similarity index — the SEAM between LAMU's retrieval layer and
//! whatever does the nearest-neighbor work underneath.
//!
//! Today the only impl is [`BruteForceCosine`]: exact cosine over every
//! stored vector. At LAMU's scale (~150 chunks/repo, per-conversation
//! turns) that's sub-millisecond and 100%-recall, so an ANN/quantized
//! index would be pure overhead + needless recall loss (see rag.rs's
//! "why brute-force" note).
//!
//! The point of the trait is the SWAP. When a corpus outgrows exact
//! brute-force — a lifetime cross-session memory, a multi-repo/fleet
//! index, ~100K–1M+ vectors — drop in a compressed/ANN impl (e.g. a
//! TurboQuant index via `turbovec`) behind this same interface, and the
//! call sites (`rag::semantic_search`, a future memory recall) don't
//! change. `add` / `search` deliberately mirror the turbovec & FAISS
//! shape so that swap is mechanical.

/// A scored search result: the similarity score plus the caller's opaque
/// payload (path+content for RAG, a turn id for memory, …).
#[derive(Debug, Clone, PartialEq)]
pub struct Scored<P> {
    pub score: f32,
    pub payload: P,
}

/// Build-then-query vector index. `P` is the per-vector payload returned
/// with each hit. Implementors decide exact-vs-approximate; callers only
/// see ranked payloads.
pub trait VectorIndex<P> {
    /// Insert one vector with its payload.
    fn add(&mut self, vector: Vec<f32>, payload: P);
    /// Top-`k` payloads by descending similarity to `query`.
    fn search(&self, query: &[f32], k: usize) -> Vec<Scored<P>>;
    /// Number of indexed vectors.
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Exact cosine similarity. Returns `0.0` for mismatched-length, empty,
/// or zero-norm inputs, so a degenerate vector never poisons ranking
/// with `NaN`.
pub(crate) fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Exact brute-force cosine index. O(n·d) per query — correct and
/// 100%-recall, ideal until the corpus is large enough that an ANN /
/// quantized index earns its recall loss. This is LAMU's default and the
/// reference the swap-in must match for accuracy.
pub struct BruteForceCosine<P> {
    rows: Vec<(Vec<f32>, P)>,
}

impl<P> BruteForceCosine<P> {
    pub fn new() -> Self {
        Self { rows: Vec::new() }
    }
    pub fn with_capacity(n: usize) -> Self {
        Self { rows: Vec::with_capacity(n) }
    }
}

impl<P> Default for BruteForceCosine<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: Clone> VectorIndex<P> for BruteForceCosine<P> {
    fn add(&mut self, vector: Vec<f32>, payload: P) {
        self.rows.push((vector, payload));
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<Scored<P>> {
        if k == 0 {
            return Vec::new();
        }
        // Score by ROW INDEX first, sort, then clone only the top-k
        // payloads — not all N. Irrelevant at repo scale, but this is the
        // seam's scale case (lifetime memory / large payloads), where
        // cloning every payload before truncating would spike allocation.
        let mut scored: Vec<(f32, usize)> = self
            .rows
            .iter()
            .enumerate()
            .map(|(i, (v, _))| (cosine(query, v), i))
            .collect();
        // Descending by score. cosine() never returns NaN, but the
        // partial_cmp fallback keeps the sort total regardless.
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored
            .into_iter()
            .take(k)
            .map(|(score, i)| Scored {
                score,
                payload: self.rows[i].1.clone(),
            })
            .collect()
    }

    fn len(&self) -> usize {
        self.rows.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_orthogonal_is_zero() {
        assert_eq!(cosine(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]), 0.0);
    }

    #[test]
    fn cosine_identical_is_one() {
        let c = cosine(&[0.5, 0.5, 0.5], &[0.5, 0.5, 0.5]);
        assert!((c - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_handles_zero_vector() {
        assert_eq!(cosine(&[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0]), 0.0);
    }

    #[test]
    fn cosine_mismatched_len_is_zero() {
        assert_eq!(cosine(&[1.0, 2.0], &[1.0]), 0.0);
    }

    #[test]
    fn brute_force_ranks_by_similarity_and_caps_k() {
        let mut idx: BruteForceCosine<&str> = BruteForceCosine::new();
        idx.add(vec![1.0, 0.0, 0.0], "exact"); // identical to query
        idx.add(vec![0.0, 1.0, 0.0], "ortho"); // orthogonal → score 0
        idx.add(vec![0.9, 0.1, 0.0], "near"); // close to query
        assert_eq!(idx.len(), 3);
        let hits = idx.search(&[1.0, 0.0, 0.0], 2);
        assert_eq!(hits.len(), 2, "k caps the result count");
        assert_eq!(hits[0].payload, "exact", "best match first");
        assert_eq!(hits[1].payload, "near", "second-best next");
        assert!(hits[0].score >= hits[1].score, "descending order");
    }

    #[test]
    fn brute_force_empty_and_k_zero() {
        let mut idx: BruteForceCosine<u32> = BruteForceCosine::new();
        assert!(idx.is_empty());
        assert!(idx.search(&[1.0], 5).is_empty(), "empty index → no hits");
        idx.add(vec![1.0], 1);
        assert!(idx.search(&[1.0], 0).is_empty(), "k=0 → no hits");
    }
}
