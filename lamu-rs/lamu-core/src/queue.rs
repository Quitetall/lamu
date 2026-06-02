//! Request queue — modular strategies for concurrent agent calls.
//!
//! Multiple agents calling the same backend concurrently would interleave
//! requests in unpredictable ways. The queue serializes per-model requests
//! and applies a configurable scheduling strategy.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{oneshot, Mutex, Notify, OwnedSemaphorePermit, Semaphore};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Strategy {
    /// First-in, first-out. Default.
    #[default]
    Fifo,
    /// Last-in, first-out.
    Lifo,
    /// Higher priority served first; ties FIFO.
    Priority,
}

#[derive(Debug)]
pub struct QueueRequest<T> {
    pub payload: T,
    pub priority: i32,
    pub enqueued_at: Instant,
    pub origin: String,
}

struct PendingTicket<T> {
    req: QueueRequest<T>,
    notify: oneshot::Sender<OwnedSemaphorePermit>,
}

struct Inner<T> {
    strategy: Strategy,
    pending: Mutex<VecDeque<PendingTicket<T>>>,
    sem: Arc<Semaphore>,
    wake: Arc<Notify>,
}

pub struct RequestQueue<T> {
    inner: Arc<Inner<T>>,
}

impl<T: Send + 'static> RequestQueue<T> {
    pub fn new(strategy: Strategy, concurrency: usize) -> Self {
        // Defense-in-depth: 0 permits is never valid for a serializing queue —
        // the dispatcher could never hand out a permit and every enqueue would
        // block forever. Clamp so a 0 from any caller can't silently wedge.
        let concurrency = concurrency.max(1);
        let inner = Arc::new(Inner {
            strategy,
            pending: Mutex::new(VecDeque::new()),
            sem: Arc::new(Semaphore::new(concurrency)),
            wake: Arc::new(Notify::new()),
        });

        let dispatch_inner = inner.clone();
        tokio::spawn(async move {
            loop {
                dispatch_inner.wake.notified().await;
                loop {
                    let permit = match dispatch_inner.sem.clone().try_acquire_owned() {
                        Ok(p) => p,
                        Err(_) => break,
                    };
                    let mut q = dispatch_inner.pending.lock().await;
                    let Some(ticket) = q.pop_front() else {
                        drop(permit);
                        break;
                    };
                    drop(q);
                    if ticket.notify.send(permit).is_err() {
                        // waiter cancelled; permit dropped; loop continues
                    }
                }
            }
        });

        Self { inner }
    }

    pub fn strategy(&self) -> Strategy { self.inner.strategy }

    pub async fn enqueue(&self, req: QueueRequest<T>) -> QueueGuard {
        let (tx, rx) = oneshot::channel();
        {
            let mut q = self.inner.pending.lock().await;
            let ticket = PendingTicket { req, notify: tx };
            match self.inner.strategy {
                Strategy::Fifo => q.push_back(ticket),
                Strategy::Lifo => q.push_front(ticket),
                Strategy::Priority => {
                    let pos = q.iter().position(|t| t.req.priority < ticket.req.priority);
                    match pos {
                        Some(i) => q.insert(i, ticket),
                        None => q.push_back(ticket),
                    }
                }
            }
        }
        self.inner.wake.notify_one();

        let permit = rx.await.expect("dispatcher dropped sender");

        QueueGuard {
            _permit: Some(permit),
            wake: self.inner.wake.clone(),
        }
    }

    pub async fn depth(&self) -> usize {
        self.inner.pending.lock().await.len()
    }
}

pub struct QueueGuard {
    _permit: Option<OwnedSemaphorePermit>,
    wake: Arc<Notify>,
}

impl Drop for QueueGuard {
    fn drop(&mut self) {
        // Drop permit first (frees concurrency slot), then notify dispatcher
        self._permit.take();
        self.wake.notify_one();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[tokio::test]
    async fn zero_concurrency_is_clamped_not_wedged() {
        // concurrency 0 must NOT build a 0-permit semaphore (would block every
        // enqueue forever). Clamped to 1 → enqueue completes promptly.
        let q: Arc<RequestQueue<u32>> = Arc::new(RequestQueue::new(Strategy::Fifo, 0));
        let guard = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            q.enqueue(QueueRequest {
                payload: 1,
                priority: 0,
                enqueued_at: Instant::now(),
                origin: "test".into(),
            }),
        )
        .await;
        assert!(guard.is_ok(), "0-concurrency queue must clamp to 1, not deadlock the enqueue");
    }

    #[tokio::test]
    async fn fifo_serial() {
        let q: Arc<RequestQueue<u32>> = Arc::new(RequestQueue::new(Strategy::Fifo, 1));
        let counter = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();
        for i in 0..5u32 {
            let q = q.clone();
            let counter = counter.clone();
            handles.push(tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_millis(i as u64 * 5)).await;
                let _g = q.enqueue(QueueRequest {
                    payload: i,
                    priority: 0,
                    enqueued_at: Instant::now(),
                    origin: "test".into(),
                }).await;
                counter.fetch_add(1, Ordering::SeqCst);
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                i
            }));
        }

        let results: Vec<u32> = futures_util::future::join_all(handles).await
            .into_iter().map(|r| r.unwrap()).collect();
        assert_eq!(results, vec![0, 1, 2, 3, 4]);
        assert_eq!(counter.load(Ordering::SeqCst), 5);
    }

    #[tokio::test]
    async fn priority_first() {
        let q: Arc<RequestQueue<&'static str>> = Arc::new(RequestQueue::new(Strategy::Priority, 1));
        let order = Arc::new(Mutex::new(Vec::<&str>::new()));

        let q1 = q.clone();
        let o1 = order.clone();
        let blocker = tokio::spawn(async move {
            let _g = q1.enqueue(QueueRequest {
                payload: "block", priority: 0,
                enqueued_at: Instant::now(), origin: "x".into(),
            }).await;
            o1.lock().await.push("block");
            tokio::time::sleep(std::time::Duration::from_millis(80)).await;
        });

        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        let q2 = q.clone();
        let o2 = order.clone();
        let low = tokio::spawn(async move {
            let _g = q2.enqueue(QueueRequest {
                payload: "low", priority: 1,
                enqueued_at: Instant::now(), origin: "x".into(),
            }).await;
            o2.lock().await.push("low");
        });

        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        let q3 = q.clone();
        let o3 = order.clone();
        let high = tokio::spawn(async move {
            let _g = q3.enqueue(QueueRequest {
                payload: "high", priority: 10,
                enqueued_at: Instant::now(), origin: "x".into(),
            }).await;
            o3.lock().await.push("high");
        });

        let _ = tokio::join!(blocker, low, high);
        let final_order = order.lock().await.clone();
        assert_eq!(final_order, vec!["block", "high", "low"]);
    }

    #[tokio::test]
    async fn concurrency_2() {
        let q: Arc<RequestQueue<u32>> = Arc::new(RequestQueue::new(Strategy::Fifo, 2));
        let in_flight = Arc::new(AtomicUsize::new(0));
        let max_seen = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();
        for i in 0..6u32 {
            let q = q.clone();
            let inf = in_flight.clone();
            let max = max_seen.clone();
            handles.push(tokio::spawn(async move {
                let _g = q.enqueue(QueueRequest {
                    payload: i, priority: 0,
                    enqueued_at: Instant::now(), origin: "x".into(),
                }).await;
                let cur = inf.fetch_add(1, Ordering::SeqCst) + 1;
                let prev = max.load(Ordering::SeqCst);
                if cur > prev { max.store(cur, Ordering::SeqCst); }
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                inf.fetch_sub(1, Ordering::SeqCst);
            }));
        }

        let _ = futures_util::future::join_all(handles).await;
        // Concurrency=2 → up to 2 simultaneously
        assert!(max_seen.load(Ordering::SeqCst) <= 2);
        assert!(max_seen.load(Ordering::SeqCst) >= 2, "should hit concurrency 2 with 6 tasks");
    }
}
