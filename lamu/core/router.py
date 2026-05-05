"""Request router — capability-based model selection with dry-run support."""
from __future__ import annotations

from typing import Optional, Sequence

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
    ) -> RouteDecision:
        """Select the best model for a request. Does NOT load — just decides.

        Priority:
        1. Explicit model name (if given)
        2. Best loaded model matching capabilities
        3. Best unloaded model matching capabilities (requires scheduler load)
        """
        # Explicit model override
        if model:
            entry = self._find_model(model)
            if entry is None:
                return RouteDecision(
                    model_name=model,
                    reason=f"model '{model}' not found in registry",
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
        loaded_matches = self._find_loaded_matching(required)
        if loaded_matches:
            best = self._rank_loaded(loaded_matches)
            return RouteDecision(
                model_name=best.entry.name,
                reason=f"best loaded model matching {[c.value for c in required]}",
                loaded=True,
            )

        # No loaded model matches — find best unloaded candidate
        candidates = self._find_registry_matching(required)
        if not candidates:
            return RouteDecision(
                model_name="",
                reason=f"no model in registry has capabilities {[c.value for c in required]}",
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
