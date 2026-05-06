"""Backend supervisor — restart-with-backoff and quarantine policy.

Sits between the daemon and `BackendHealth`. When a backend dies the
supervisor:
1. Logs a structured JSON event to stderr.
2. Calls `health.record_error()` (state advances).
3. If the backend is now `DEAD` and has restart attempts left, schedules
   an exponential-backoff restart (1s, 2s, 4s).
4. After exhausting retries, calls `health.force_quarantine()` and emits
   a final stderr event so operators know to intervene.

The supervisor never silently fails. Every state transition produces a
log line, every restart attempt is countable, and quarantine is sticky.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional

from lamu.core.health import BackendHealth, HealthState
from lamu.core.observability import emit


_log = logging.getLogger(__name__)


@dataclass
class RestartPolicy:
    """How aggressively the supervisor retries a dead backend."""
    max_attempts: int = 3
    backoff_seconds: tuple[int, ...] = (1, 2, 4)


def _emit_event(event: str, **fields: object) -> None:
    """Backwards-compatible shim. Routes through the observability sink so
    the file/OTLP exporters see every event the supervisor produces."""
    emit(event, **fields)


class Supervisor:
    """Coordinates restart attempts for a single backend."""

    def __init__(
        self,
        health: BackendHealth,
        restart_fn: Callable[[], None],
        policy: Optional[RestartPolicy] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._health = health
        self._restart_fn = restart_fn
        self._policy = policy or RestartPolicy()
        self._sleep = sleep_fn

    def report_failure(self, exc: BaseException) -> None:
        """Record an error and possibly trigger restart-with-backoff.

        Returns when:
        - the backend has recovered (after a successful restart), OR
        - the backend has been quarantined (max attempts exhausted).
        """
        self._health.record_error(exc)
        _emit_event(
            "backend_failure",
            backend_id=self._health.backend_id,
            state=self._health.state.value,
            consecutive_errors=self._health.consecutive_errors,
            error=self._health.last_error,
        )
        if self._health.state is HealthState.DEAD:
            self._attempt_restart()
        elif self._health.state is HealthState.QUARANTINED:
            _emit_event(
                "backend_quarantined",
                backend_id=self._health.backend_id,
                reason=self._health.last_error,
            )

    def _attempt_restart(self) -> None:
        """Walk through the backoff schedule. Quarantines on exhaustion."""
        for attempt_idx, delay in enumerate(self._policy.backoff_seconds, 1):
            if attempt_idx > self._policy.max_attempts:
                break
            self._health.restart_attempts = attempt_idx
            _emit_event(
                "backend_restart_attempt",
                backend_id=self._health.backend_id,
                attempt=attempt_idx,
                delay_s=delay,
            )
            self._sleep(delay)
            try:
                self._restart_fn()
            except Exception as exc:  # noqa: BLE001
                _log.warning(
                    "supervisor_restart_failed backend=%s attempt=%d err=%s",
                    self._health.backend_id, attempt_idx, exc,
                )
                continue
            else:
                # Restart returned without raising — declare victory.
                self._health.record_success()
                _emit_event(
                    "backend_restarted",
                    backend_id=self._health.backend_id,
                    attempt=attempt_idx,
                )
                return

        # All attempts exhausted.
        self._health.force_quarantine(reason="max restart attempts exhausted")
        _emit_event(
            "backend_quarantined",
            backend_id=self._health.backend_id,
            reason="max restart attempts exhausted",
        )
