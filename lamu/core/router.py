"""Request router — capability-based model selection with dry-run support."""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

from lamu.core.health import BackendHealth
from lamu.core.scheduler import VramScheduler
from lamu.core.types import Capability, LoadedModel, ModelEntry, RouteDecision


class Router:
    """Routes requests to the best available model based on capabilities.

    Routing semantics:
    - `capabilities` is a REQUIREMENT, not a preference
    - If no loaded model matches, scheduler MUST load one (evicting LRU if needed)
    - `model` parameter overrides capability routing (explicit > smart)
    - Never silently downgrade
    """

    def __init__(
        self,
        scheduler: VramScheduler,
        registry: Sequence[ModelEntry],
    ) -> None:
        self._scheduler = scheduler
        self._registry: dict[str, ModelEntry] = {m.name: m for m in registry}

    def update_registry(self, models: Sequence[ModelEntry]) -> None:
        self._registry = {m.name: m for m in models}

    def route(
        self,
        model: Optional[str] = None,
        capabilities: Optional[Sequence[Capability]] = None,
        health_map: Optional[Mapping[str, BackendHealth]] = None,
    ) -> RouteDecision:
        """Select the best model for a request. Does NOT load — just decides.

        Args:
            model: explicit model name override (highest priority).
            capabilities: required capabilities; default = {CHAT}.
            health_map: optional {model_name -> BackendHealth}. Models whose
                health is not `usable` (DEAD/QUARANTINED) are filtered out.
                Backwards compatible: if absent, no health filtering.

        Priority:
        1. Explicit model name (if given) — refuses if model is unhealthy
        2. Best loaded model matching capabilities
        3. Best unloaded model matching capabilities (requires scheduler load)
        """
        def _is_usable(name: str) -> bool:
            if not health_map:
                return True
            h = health_map.get(name)
            return h is None or h.usable

        # Explicit model override
        if model:
            entry = self._find_model(model)
            if entry is None:
                return RouteDecision(
                    model_name=model,
                    reason=f"model '{model}' not found in registry",
                    loaded=False,
                )
            if not _is_usable(entry.name):
                return RouteDecision(
                    model_name=entry.name,
                    reason=f"model '{entry.name}' is unhealthy: {health_map[entry.name].state.value}",
                    loaded=False,
                )
            loaded = self._scheduler.is_loaded(entry.name)
            if loaded:
                return RouteDecision(
                    model_name=entry.name,
                    reason="explicit model selection (loaded)",
                    loaded=True,
                )
            # Need to load
            can_load, to_evict = self._scheduler.plan_load(entry)
            if can_load:
                return RouteDecision(
                    model_name=entry.name,
                    reason="explicit model selection (will load)",
                    loaded=False,
                    would_evict=tuple(to_evict),
                )
            return RouteDecision(
                model_name=entry.name,
                reason="explicit model — cannot fit in VRAM even with eviction",
                loaded=False,
            )

        # Capability-based routing
        required = set(capabilities) if capabilities else {Capability.CHAT}

        # Try loaded models first (prefer already-running for speed)
        loaded_matches = [
            m for m in self._find_loaded_matching(required)
            if _is_usable(m.entry.name)
        ]
        if loaded_matches:
            best = self._rank_loaded(loaded_matches)
            return RouteDecision(
                model_name=best.entry.name,
                reason=f"best loaded model matching {[c.value for c in required]}",
                loaded=True,
            )

        # No loaded model matches — find best unloaded candidate
        candidates = [
            e for e in self._find_registry_matching(required)
            if _is_usable(e.name)
        ]
        if not candidates:
            base_reason = (
                f"no model in registry has capabilities "
                f"{[c.value for c in required]}"
            )
            if health_map:
                base_reason += " (after health filtering)"
            return RouteDecision(
                model_name="",
                reason=base_reason,
                loaded=False,
            )

        # Pick best candidate (prefer smaller VRAM for faster load)
        candidates.sort(key=lambda m: (-len(m.capabilities), m.vram_mb))
        best_entry = candidates[0]

        can_load, to_evict = self._scheduler.plan_load(best_entry)
        if can_load:
            return RouteDecision(
                model_name=best_entry.name,
                reason=f"best unloaded model matching {[c.value for c in required]} (will load)",
                loaded=False,
                would_evict=tuple(to_evict),
            )

        return RouteDecision(
            model_name=best_entry.name,
            reason="matching model found but cannot fit in VRAM",
            loaded=False,
        )

    def _find_model(self, name: str) -> Optional[ModelEntry]:
        """Find model by name (exact or substring match)."""
        if name in self._registry:
            return self._registry[name]
        # Partial match
        matches = [m for n, m in self._registry.items() if name in n]
        return matches[0] if len(matches) == 1 else None

    def _find_loaded_matching(
        self, required: set[Capability]
    ) -> list[LoadedModel]:
        """Find loaded models that have ALL required capabilities."""
        matches: list[LoadedModel] = []
        for loaded in self._scheduler.loaded_models():
            model_caps = set(loaded.entry.capabilities)
            if required.issubset(model_caps):
                matches.append(loaded)
        return matches

    def _find_registry_matching(
        self, required: set[Capability]
    ) -> list[ModelEntry]:
        """Find registry models that have ALL required capabilities."""
        matches: list[ModelEntry] = []
        for entry in self._registry.values():
            model_caps = set(entry.capabilities)
            if required.issubset(model_caps):
                matches.append(entry)
        return matches

    def _rank_loaded(self, models: list[LoadedModel]) -> LoadedModel:
        """Rank loaded models: prefer largest (smartest), then most recently used."""
        return max(models, key=lambda m: (m.entry.params_b, m.last_used_ts))
