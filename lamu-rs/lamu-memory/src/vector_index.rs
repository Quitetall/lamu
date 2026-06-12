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

/// L2-normalize a vector to unit length. Returns the input unchanged
/// (a copy) when its norm is zero or non-finite, so a degenerate vector
/// never produces `NaN`/`Inf` directions. Used by [`TurboVecIndex`] to
/// make turbovec's inner-product score equal cosine similarity (turbovec
/// scores MIPS on directions; on unit inputs `<a,b>` == `cos(a,b)`).
//
// Only the `turbovec` backend (and the tests) call this, so a feature-off
// non-test build would otherwise flag it as dead.
#[cfg_attr(not(any(feature = "turbovec", test)), allow(dead_code))]
pub(crate) fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 || !norm.is_finite() {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
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

// ── Backend selector ────────────────────────────────────────────────

/// Which [`VectorIndex`] backend a recall site should build.
///
/// `Brute` is the default and the ONLY backend in a normal build. `TurboVec`
/// is reachable only when BOTH the `turbovec` feature is compiled in AND the
/// runtime env requests it — see [`vector_backend`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VectorBackend {
    Brute,
    // Only ever constructed under the `turbovec` feature; the variant + its
    // match arms exist unconditionally so the selector stays total, so the
    // feature-off build would otherwise flag it as never-constructed.
    #[cfg_attr(not(feature = "turbovec"), allow(dead_code))]
    TurboVec,
}

/// Resolve the active vector backend from the `LAMU_VECTOR_BACKEND` env var.
///
/// - `"brute"` → [`VectorBackend::Brute`] — the explicit escape hatch,
///   honored in every build.
/// - `"turbovec"` → [`VectorBackend::TurboVec`] **iff** the `turbovec`
///   feature was compiled in; otherwise [`VectorBackend::Brute`] plus a
///   one-time `warn!` that turbovec was requested but not built in.
/// - unset / anything unrecognized → the BUILD's default: TurboVec when
///   the `turbovec` feature is compiled in (ADR 0031 — compiling the
///   feature IS the opt-in; the persistent `.tv` lifecycle in
///   `crate::tv_store` then maintains the index), Brute otherwise
///   (feature-off behavior unchanged: brute is a full scan anyway, so a
///   per-query rebuild loses nothing — there is nothing to persist).
pub(crate) fn vector_backend() -> VectorBackend {
    match std::env::var("LAMU_VECTOR_BACKEND").ok().as_deref() {
        Some("brute") => VectorBackend::Brute,
        Some("turbovec") => {
            #[cfg(feature = "turbovec")]
            {
                VectorBackend::TurboVec
            }
            #[cfg(not(feature = "turbovec"))]
            {
                warn_turbovec_unavailable();
                VectorBackend::Brute
            }
        }
        // Unset or unrecognized → the build's default.
        _ => {
            #[cfg(feature = "turbovec")]
            {
                VectorBackend::TurboVec
            }
            #[cfg(not(feature = "turbovec"))]
            {
                VectorBackend::Brute
            }
        }
    }
}

/// Emit a single `warn!` the first time turbovec is requested in a build
/// that did not compile the feature. Subsequent calls are silent.
#[cfg(not(feature = "turbovec"))]
fn warn_turbovec_unavailable() {
    use std::sync::Once;
    static WARN_ONCE: Once = Once::new();
    WARN_ONCE.call_once(|| {
        tracing::warn!(
            "LAMU_VECTOR_BACKEND=turbovec requested but lamu-mcp was built \
             without the `turbovec` feature; falling back to brute-force \
             cosine. Rebuild with `--features turbovec` to enable it."
        );
    });
}

// ── TurboVec backend (opt-in) ───────────────────────────────────────

/// Compressed-search vector index backed by `turbovec`'s `TurboQuantIndex`
/// (TurboQuant 2-4 bit quantization + SIMD MIPS search).
///
/// Same `VectorIndex<P>` contract as [`BruteForceCosine`]. Cosine-equivalence:
/// turbovec scores inner product on unit *directions*, so we L2-[`normalize`]
/// every vector on `add` and the query on `search`; on unit inputs the
/// returned score == cosine similarity (the seam's contract).
///
/// This impl builds the underlying `TurboQuantIndex` lazily, from the
/// buffered rows, on each `search` — mirroring how the recall sites rebuild
/// a `BruteForceCosine` per query. It remains the FALLBACK for queries the
/// persistent path can't serve (dims not a multiple of 8); the persistent
/// on-disk `.tv` lifecycle (load-or-rebuild, catch-up, throttled atomic
/// persist) lives in `crate::tv_store` (ADR 0031).
#[cfg(feature = "turbovec")]
pub struct TurboVecIndex<P> {
    /// Buffered (already-normalized vector, payload) rows, parallel to the
    /// slot indices the built `TurboQuantIndex` returns.
    rows: Vec<(Vec<f32>, P)>,
    bit_width: usize,
}

#[cfg(feature = "turbovec")]
impl<P> TurboVecIndex<P> {
    /// Default quantization width: 4 bits (best recall per turbovec's README).
    pub const DEFAULT_BIT_WIDTH: usize = 4;

    pub fn new() -> Self {
        Self::with_bit_width(Self::DEFAULT_BIT_WIDTH)
    }

    pub fn with_capacity(n: usize) -> Self {
        Self {
            rows: Vec::with_capacity(n),
            bit_width: Self::DEFAULT_BIT_WIDTH,
        }
    }

    /// Construct with an explicit `bit_width` (must be 2, 3, or 4; turbovec
    /// rejects anything else at build time, falling back to brute results).
    pub fn with_bit_width(bit_width: usize) -> Self {
        Self {
            rows: Vec::new(),
            bit_width,
        }
    }
}

#[cfg(feature = "turbovec")]
impl<P> Default for TurboVecIndex<P> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "turbovec")]
impl<P: Clone> VectorIndex<P> for TurboVecIndex<P> {
    /// Store the L2-normalized vector + payload. Normalizing on add (and on
    /// the query in `search`) is what makes turbovec's inner-product score
    /// equal cosine.
    fn add(&mut self, vector: Vec<f32>, payload: P) {
        self.rows.push((normalize(&vector), payload));
    }

    /// Build a `TurboQuantIndex` from the buffered rows and run a compressed
    /// MIPS search of the normalized query, returning the top-`k` payloads
    /// sorted descending by score.
    ///
    /// Falls back to the exact brute-force path (over the SAME normalized
    /// rows, so scores stay comparable) when the inputs can't be handed to
    /// turbovec: `k == 0`, empty index, a query whose length doesn't match
    /// the row dim, or a dim that isn't a positive multiple of 8 (turbovec's
    /// hard constraint). This keeps the seam total — never a panic or NaN —
    /// while still exercising compressed search on real (8-multiple) dims.
    fn search(&self, query: &[f32], k: usize) -> Vec<Scored<P>> {
        if k == 0 || self.rows.is_empty() {
            return Vec::new();
        }
        let dim = self.rows[0].0.len();
        let turbovec_usable = dim != 0 && dim % 8 == 0 && query.len() == dim;
        if !turbovec_usable {
            return self.brute_fallback(query, k);
        }

        let qn = normalize(query);
        let mut index = match turbovec::TurboQuantIndex::new(dim, self.bit_width) {
            Ok(idx) => idx,
            // bit_width out of {2,3,4} or dim not a multiple of 8 — already
            // guarded above for dim, so this is a bad bit_width. Stay total.
            Err(_) => return self.brute_fallback(query, k),
        };
        // Flatten all rows into one batch add (length = n * dim).
        let mut flat: Vec<f32> = Vec::with_capacity(self.rows.len() * dim);
        for (v, _) in &self.rows {
            flat.extend_from_slice(v);
        }
        index.add(&flat);

        let cap = k.min(self.rows.len());
        let results = index.search(&qn, cap);
        // SearchResults is row-major flattened; single query (nq == 1) lives
        // in [0..k]. `indices` are slot positions parallel to `self.rows`.
        let scores = results.scores_for_query(0);
        let indices = results.indices_for_query(0);
        let mut hits = Vec::with_capacity(scores.len());
        for (score, &slot) in scores.iter().zip(indices.iter()) {
            // turbovec pads short result rows with sentinel ids; skip any
            // out-of-range / negative slot defensively.
            if slot < 0 {
                continue;
            }
            let slot = slot as usize;
            if let Some((_, payload)) = self.rows.get(slot) {
                hits.push(Scored {
                    score: *score,
                    payload: payload.clone(),
                });
            }
        }
        // turbovec already returns descending, but re-sort to guarantee the
        // seam's contract regardless of backend internals.
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(k);
        hits
    }

    fn len(&self) -> usize {
        self.rows.len()
    }
}

#[cfg(feature = "turbovec")]
impl<P: Clone> TurboVecIndex<P> {
    /// Exact cosine over the buffered (already-normalized) rows. Used when a
    /// query can't be handed to turbovec; identical ranking logic to
    /// [`BruteForceCosine::search`].
    fn brute_fallback(&self, query: &[f32], k: usize) -> Vec<Scored<P>> {
        let qn = normalize(query);
        let mut scored: Vec<(f32, usize)> = self
            .rows
            .iter()
            .enumerate()
            .map(|(i, (v, _))| (cosine(&qn, v), i))
            .collect();
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

    #[test]
    fn normalize_unit_vector_stays_unit() {
        let n = normalize(&[3.0, 4.0]); // norm 5 → (0.6, 0.8)
        let len: f32 = n.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((len - 1.0).abs() < 1e-6, "normalized → unit length");
        assert!((n[0] - 0.6).abs() < 1e-6);
        assert!((n[1] - 0.8).abs() < 1e-6);
        // Already-unit input is preserved.
        let u = normalize(&[1.0, 0.0, 0.0]);
        assert_eq!(u, vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn normalize_zero_vector_is_safe() {
        let z = normalize(&[0.0, 0.0, 0.0]);
        assert_eq!(z, vec![0.0, 0.0, 0.0], "zero vector returned unchanged");
        assert!(z.iter().all(|x| x.is_finite()), "never NaN/Inf");
    }

    /// Feature OFF: brute is the default for unset/unrecognized AND the
    /// forced value — unchanged pre-ADR-0031 behavior, byte-identical.
    #[cfg(not(feature = "turbovec"))]
    #[test]
    fn vector_backend_defaults_to_brute_when_feature_off() {
        // SAFETY: single-threaded test; we own the var for its scope.
        unsafe {
            std::env::remove_var("LAMU_VECTOR_BACKEND");
        }
        assert_eq!(vector_backend(), VectorBackend::Brute);
        unsafe {
            std::env::set_var("LAMU_VECTOR_BACKEND", "brute");
        }
        assert_eq!(vector_backend(), VectorBackend::Brute);
        unsafe {
            std::env::set_var("LAMU_VECTOR_BACKEND", "nonsense");
        }
        assert_eq!(
            vector_backend(),
            VectorBackend::Brute,
            "unrecognized value → brute"
        );
        unsafe {
            std::env::remove_var("LAMU_VECTOR_BACKEND");
        }
    }

    /// Feature ON: the default FLIPS to turbovec (ADR 0031 — compiling
    /// the feature is the opt-in), while `brute` stays an explicit
    /// escape hatch.
    #[cfg(feature = "turbovec")]
    #[test]
    fn vector_backend_defaults_to_turbovec_when_feature_on() {
        // SAFETY: single-threaded test; we own the var for its scope.
        unsafe {
            std::env::remove_var("LAMU_VECTOR_BACKEND");
        }
        assert_eq!(vector_backend(), VectorBackend::TurboVec, "unset → turbovec");
        unsafe {
            std::env::set_var("LAMU_VECTOR_BACKEND", "brute");
        }
        assert_eq!(
            vector_backend(),
            VectorBackend::Brute,
            "explicit brute still forces brute"
        );
        unsafe {
            std::env::set_var("LAMU_VECTOR_BACKEND", "nonsense");
        }
        assert_eq!(
            vector_backend(),
            VectorBackend::TurboVec,
            "unrecognized value → the build default"
        );
        unsafe {
            std::env::remove_var("LAMU_VECTOR_BACKEND");
        }
    }

    /// Without the `turbovec` feature, requesting it must still resolve to
    /// brute (the warn-and-fall-back path).
    #[cfg(not(feature = "turbovec"))]
    #[test]
    fn vector_backend_turbovec_requested_but_not_compiled_falls_back() {
        unsafe {
            std::env::set_var("LAMU_VECTOR_BACKEND", "turbovec");
        }
        assert_eq!(
            vector_backend(),
            VectorBackend::Brute,
            "turbovec requested but feature off → brute"
        );
        unsafe {
            std::env::remove_var("LAMU_VECTOR_BACKEND");
        }
    }

    #[cfg(feature = "turbovec")]
    #[test]
    fn vector_backend_turbovec_when_compiled_and_requested() {
        unsafe {
            std::env::set_var("LAMU_VECTOR_BACKEND", "turbovec");
        }
        assert_eq!(vector_backend(), VectorBackend::TurboVec);
        unsafe {
            std::env::remove_var("LAMU_VECTOR_BACKEND");
        }
    }

    /// turbovec is approximate, so we assert TOP-1 (and top-2 order)
    /// agreement with brute-force on the SAME normalized inputs over a
    /// clearly-separated tiny set — not exact score equality. dim is 8 (a
    /// positive multiple of 8, turbovec's hard constraint).
    #[cfg(feature = "turbovec")]
    #[test]
    fn turbovec_top_k_matches_brute_on_separated_set() {
        // Four well-separated 8-d directions; query is closest to "a", then "b".
        let a = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let c = vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let d = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        // Query leans hard toward a, a bit toward b.
        let query = vec![0.9, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let build = |mk: fn() -> Box<dyn VectorIndex<&'static str>>| {
            let mut idx = mk();
            idx.add(a.clone(), "a");
            idx.add(b.clone(), "b");
            idx.add(c.clone(), "c");
            idx.add(d.clone(), "d");
            idx
        };

        let brute = build(|| Box::new(BruteForceCosine::<&'static str>::new()));
        let turbo = build(|| Box::new(TurboVecIndex::<&'static str>::new()));

        let brute_hits = brute.search(&query, 2);
        let turbo_hits = turbo.search(&query, 2);

        assert_eq!(turbo_hits.len(), 2, "k caps result count");
        assert_eq!(brute_hits[0].payload, "a", "brute top-1 is a");
        assert_eq!(turbo_hits[0].payload, "a", "turbovec top-1 matches brute");
        assert_eq!(turbo_hits[1].payload, "b", "turbovec top-2 matches brute order");
        assert!(
            turbo_hits[0].score >= turbo_hits[1].score,
            "descending order"
        );
        // Scores are bounded similarities, never NaN.
        assert!(turbo_hits.iter().all(|h| h.score.is_finite()));
    }

    /// turbovec backend stays total on degenerate inputs: empty index, k=0,
    /// and a non-multiple-of-8 dim (falls back to exact cosine).
    #[cfg(feature = "turbovec")]
    #[test]
    fn turbovec_total_on_edge_cases() {
        let mut empty: TurboVecIndex<u32> = TurboVecIndex::new();
        assert!(empty.is_empty());
        assert!(empty.search(&[0.0; 8], 5).is_empty(), "empty → no hits");
        empty.add(vec![1.0; 8], 1);
        assert!(empty.search(&[1.0; 8], 0).is_empty(), "k=0 → no hits");

        // dim=3 (not a multiple of 8) → exact-cosine fallback, still ranked.
        let mut odd: TurboVecIndex<&str> = TurboVecIndex::new();
        odd.add(vec![1.0, 0.0, 0.0], "x");
        odd.add(vec![0.0, 1.0, 0.0], "y");
        let hits = odd.search(&[1.0, 0.0, 0.0], 1);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].payload, "x", "fallback ranks correctly");
    }
}
