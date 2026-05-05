"""Backend health state machine.

The whole point: if a backend dies, NEVER let users get stuck on it. The
router consults `BackendHealth.usable` for every routing decision; the API
surfaces a 503 the moment no healthy backend exists.

State transitions (driven entirely by `record_*` methods):

    HEALTHY ──record_error──> DEGRADED ──record_error×2──> DEAD
       ▲                                                     │
       └───────── record_success ─────────────┐              │
                                              │              ▼
                                              │      record_error×2
                                              │              │
                                              │              ▼
                                              └─ ... ── QUARANTINED  (terminal)

QUARANTINED is sticky — only a human (or supervisor explicit reset) can
clear it. DEAD can recover via `record_success()` if a probe succeeds.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Optional


class HealthState(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DEAD = "dead"
    QUARANTINED = "quarantined"


@dataclass
class BackendHealth:
    """Health state for a single backend identified by `backend_id`."""

    backend_id: str
    state: HealthState = HealthState.HEALTHY
    consecutive_errors: int = 0
    last_error: Optional[str] = None
    last_error_ts: float = 0.0
    restart_attempts: int = 0
    quarantined_at: Optional[float] = None

    # Thresholds — same constants live in lamu-core/src/health.rs (Rust mirror).
    DEAD_THRESHOLD: ClassVar[int] = 3
    QUARANTINE_THRESHOLD: ClassVar[int] = 5

    def record_success(self) -> None:
        """Reset error counter; do NOT clear QUARANTINED — that is sticky."""
        if self.state is HealthState.QUARANTINED:
            return
        self.consecutive_errors = 0
        self.state = HealthState.HEALTHY
        self.last_error = None

    def record_error(self, exc: BaseException) -> None:
        """Record a failure. Promotes state per thresholds.

        QUARANTINED is terminal — once there, stays there until manual reset.
        """
        if self.state is HealthState.QUARANTINED:
            return
        self.consecutive_errors += 1
        self.last_error = f"{type(exc).__name__}: {exc}"
        self.last_error_ts = time.time()

        if self.consecutive_errors >= self.QUARANTINE_THRESHOLD:
            self.state = HealthState.QUARANTINED
            self.quarantined_at = self.last_error_ts
        elif self.consecutive_errors >= self.DEAD_THRESHOLD:
            self.state = HealthState.DEAD
        else:
            self.state = HealthState.DEGRADED

    def force_quarantine(self, reason: str = "manual") -> None:
        """Hard-quarantine a backend (e.g. supervisor exhausted all restarts)."""
        self.state = HealthState.QUARANTINED
        self.quarantined_at = time.time()
        self.last_error = f"force_quarantine: {reason}"

    @property
    def usable(self) -> bool:
        """True only if state is HEALTHY or DEGRADED.

        DEGRADED still routes — gives the backend a chance to recover. DEAD
        and QUARANTINED never route until manual intervention.
        """
        return self.state in (HealthState.HEALTHY, HealthState.DEGRADED)

    def to_dict(self) -> dict:
        return {
            "backend_id": self.backend_id,
            "state": self.state.value,
            "consecutive_errors": self.consecutive_errors,
            "last_error": self.last_error,
            "last_error_ts": self.last_error_ts,
            "restart_attempts": self.restart_attempts,
        }


@dataclass
class HealthRegistry:
    """Process-wide map of backend_id → BackendHealth."""

    _by_id: dict[str, BackendHealth] = field(default_factory=dict)

    def get_or_create(self, backend_id: str) -> BackendHealth:
        if backend_id not in self._by_id:
            self._by_id[backend_id] = BackendHealth(backend_id=backend_id)
        return self._by_id[backend_id]

    def get(self, backend_id: str) -> Optional[BackendHealth]:
        return self._by_id.get(backend_id)

    def all(self) -> dict[str, BackendHealth]:
        return dict(self._by_id)

    def usable_ids(self) -> set[str]:
        return {bid for bid, h in self._by_id.items() if h.usable}

    def snapshot(self) -> dict[str, dict]:
        return {bid: h.to_dict() for bid, h in self._by_id.items()}
