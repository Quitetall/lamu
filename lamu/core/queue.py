"""Request queue — modular strategies for concurrent agent calls.

Mirrors `lamu-rs/lamu-core/src/queue.rs`.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, Optional, TypeVar


class Strategy(Enum):
    FIFO = "fifo"
    LIFO = "lifo"
    PRIORITY = "priority"


T = TypeVar("T")


@dataclass(frozen=True)
class QueueRequest(Generic[T]):
    payload: T
    priority: int = 0
    enqueued_at: float = field(default_factory=time.monotonic)
    origin: str = "anonymous"


class QueueGuard:
    """Releases queue slot on exit. Use `async with`."""

    def __init__(self, queue: "RequestQueue") -> None:
        self._queue = queue
        self._released = False

    async def __aenter__(self) -> "QueueGuard":
        return self

    async def __aexit__(self, *_args: object) -> None:
        self.release()

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._queue._release_slot()


class RequestQueue(Generic[T]):
    """Bounded-concurrency request queue with pluggable scheduling."""

    def __init__(self, strategy: Strategy = Strategy.FIFO, concurrency: int = 1) -> None:
        self._strategy = strategy
        self._max_concurrency = concurrency
        self._in_flight = 0
        self._pending: deque[tuple[QueueRequest, asyncio.Future[QueueGuard]]] = deque()
        self._lock = asyncio.Lock()

    @property
    def strategy(self) -> Strategy:
        return self._strategy

    @property
    def concurrency(self) -> int:
        return self._max_concurrency

    async def enqueue(self, req: QueueRequest[T]) -> QueueGuard:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[QueueGuard] = loop.create_future()

        async with self._lock:
            # If a slot is free and queue is empty, hand out immediately
            if self._in_flight < self._max_concurrency and not self._pending:
                self._in_flight += 1
                fut.set_result(QueueGuard(self))
                return await fut

            entry = (req, fut)
            if self._strategy is Strategy.FIFO:
                self._pending.append(entry)
            elif self._strategy is Strategy.LIFO:
                self._pending.appendleft(entry)
            else:  # PRIORITY: insert before first lower-priority entry
                pos = next(
                    (i for i, (r, _) in enumerate(self._pending) if r.priority < req.priority),
                    None,
                )
                if pos is None:
                    self._pending.append(entry)
                else:
                    self._pending.insert(pos, entry)

        return await fut

    def _release_slot(self) -> None:
        # Schedule the wake-up; we may not be inside the lock context here.
        asyncio.create_task(self._on_release())

    async def _on_release(self) -> None:
        async with self._lock:
            self._in_flight -= 1
            while self._in_flight < self._max_concurrency and self._pending:
                _, fut = self._pending.popleft()
                if fut.done():
                    continue  # caller cancelled
                self._in_flight += 1
                fut.set_result(QueueGuard(self))

    async def depth(self) -> int:
        async with self._lock:
            return len(self._pending)
