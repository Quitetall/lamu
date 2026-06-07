//! In-process research session store (ADR 0023, R4).
//!
//! `deep_research` writes a session (the final corpus + one embedding per paper);
//! `research_chat` reads it to answer follow-ups grounded in the SAME studies.
//! Process-local (lost on restart — acceptable), capped, and TTL-evicted so the
//! corpus + 768-dim vectors don't accumulate unbounded.

use jart::core::model::Paper;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

const MAX_SESSIONS: usize = 32;
const TTL: Duration = Duration::from_secs(3600);

struct Session {
    corpus: Vec<Paper>,
    embeddings: Vec<Vec<f32>>, // one per corpus paper (empty if embedding failed)
    created: Instant,
}

static STORE: OnceLock<Mutex<HashMap<String, Session>>> = OnceLock::new();
static COUNTER: AtomicU64 = AtomicU64::new(1);

fn store() -> &'static Mutex<HashMap<String, Session>> {
    STORE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Store a session; returns its id. Evicts expired entries, then the oldest
/// while over the cap.
pub(crate) fn put(corpus: Vec<Paper>, embeddings: Vec<Vec<f32>>) -> String {
    let id = format!("dr-{:x}", COUNTER.fetch_add(1, Ordering::Relaxed));
    let mut m = store().lock().expect("session store poisoned");
    m.retain(|_, s| s.created.elapsed() < TTL);
    while m.len() >= MAX_SESSIONS {
        match m.iter().min_by_key(|(_, s)| s.created).map(|(k, _)| k.clone()) {
            Some(oldest) => {
                m.remove(&oldest);
            }
            None => break,
        }
    }
    m.insert(id.clone(), Session { corpus, embeddings, created: Instant::now() });
    id
}

/// Fetch a live session's corpus + embeddings (cloned) by id.
pub(crate) fn get(id: &str) -> Option<(Vec<Paper>, Vec<Vec<f32>>)> {
    let m = store().lock().expect("session store poisoned");
    m.get(id)
        .filter(|s| s.created.elapsed() < TTL)
        .map(|s| (s.corpus.clone(), s.embeddings.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paper(t: &str) -> Paper {
        Paper {
            kind: "paper".into(), source: "HF".into(), topic: "t".into(), title: t.into(),
            link: "l".into(), date_label: "d".into(), ts: 1, summary: "s".into(), grounding: "g".into(),
        }
    }

    #[test]
    fn put_then_get_roundtrips() {
        let id = put(vec![paper("A")], vec![vec![1.0, 2.0]]);
        let (corpus, embs) = get(&id).expect("session present");
        assert_eq!(corpus.len(), 1);
        assert_eq!(corpus[0].title, "A");
        assert_eq!(embs, vec![vec![1.0, 2.0]]);
        assert!(get("dr-nonexistent").is_none());
    }
}
