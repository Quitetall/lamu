//! Scale-test P0 (ADR 0017 roadmap): the request queue under concurrent load.
//!
//! The per-model queue (`lamu_core::queue`) serializes concurrent agent calls
//! behind a semaphore. These tests hammer it with 1000 concurrent enqueues per
//! strategy and assert the three invariants that matter under load:
//!   1. the concurrency cap is never exceeded (no semaphore over-issue),
//!   2. every task completes (no permit leak / lost wake-up),
//!   3. the whole run finishes inside a timeout (no deadlock).
//! Pure-unit, GPU-free — runs in normal CI. Ordering-per-strategy is covered by
//! the unit tests in queue.rs; this is the safety net the multi-GPU + multi-user
//! tracks lean on when they add concurrency.

use lamu_core::queue::{QueueRequest, RequestQueue, Strategy};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

async fn run_load(strategy: Strategy, concurrency: usize, n: usize) {
    let q: Arc<RequestQueue<usize>> = Arc::new(RequestQueue::new(strategy, concurrency));
    let live = Arc::new(AtomicUsize::new(0));
    let max_live = Arc::new(AtomicUsize::new(0));
    let done = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::with_capacity(n);
    for i in 0..n {
        let (q, live, max_live, done) = (q.clone(), live.clone(), max_live.clone(), done.clone());
        handles.push(tokio::spawn(async move {
            let guard = q
                .enqueue(QueueRequest {
                    payload: i,
                    priority: (i % 7) as i32,
                    enqueued_at: Instant::now(),
                    origin: "load-test".into(),
                })
                .await;
            // Between acquiring the guard and dropping it we hold one of the
            // `concurrency` permits — so `live` can never exceed the cap.
            let now = live.fetch_add(1, Ordering::SeqCst) + 1;
            max_live.fetch_max(now, Ordering::SeqCst);
            tokio::task::yield_now().await; // force interleaving
            live.fetch_sub(1, Ordering::SeqCst);
            done.fetch_add(1, Ordering::SeqCst);
            drop(guard);
        }));
    }

    let all = futures_util::future::join_all(handles);
    tokio::time::timeout(Duration::from_secs(30), all)
        .await
        .expect("queue load deadlocked or leaked permits (timed out)");

    assert_eq!(done.load(Ordering::SeqCst), n, "every enqueued task completed");
    let peak = max_live.load(Ordering::SeqCst);
    assert!(peak >= 1, "at least one task ran");
    assert!(peak <= concurrency, "concurrency cap respected: peak {peak} <= {concurrency}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn queue_load_fifo() {
    run_load(Strategy::Fifo, 4, 1000).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn queue_load_lifo() {
    run_load(Strategy::Lifo, 4, 1000).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn queue_load_priority() {
    run_load(Strategy::Priority, 8, 1000).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn queue_load_concurrency_one_serializes() {
    // concurrency=1 → strict serialization; peak live must be exactly 1.
    run_load(Strategy::Fifo, 1, 500).await;
}
