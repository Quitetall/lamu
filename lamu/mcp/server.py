"""LAMU MCP Server — primary interface for model management.

Tools exposed:
  query        — generate with smart routing
  plan_query   — dry-run: see what would be routed without generating
  list_models  — registry + load status
  load_model   — explicit load
  unload_model — explicit unload
  vram_status  — current VRAM allocation
  scan_models  — re-discover models on disk
"""
from __future__ import annotations

import json as _json
import logging
import os
import urllib.error
from pathlib import Path
from typing import Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from lamu.core.errors import BackendError, BackendUnavailable
from lamu.core.health import HealthRegistry
from lamu.core.queue import QueueRequest, RequestQueue, Strategy as QueueStrategy
from lamu.core.reasoning import get_extractor
from lamu.core.registry import load_registry, scan_directory, write_registry
from lamu.core.router import Router
from lamu.core.scheduler import VramScheduler
from lamu.core.supervisor import Supervisor
from lamu.core.types import (
    Capability,
    ModelEntry,
)


_log = logging.getLogger(__name__)


# Errors expected when talking to a (potentially dying) backend over HTTP.
# Anything outside this set is a real bug and must propagate.
_BACKEND_HTTP_ERRORS: tuple[type[BaseException], ...] = (
    urllib.error.URLError,
    ConnectionError,
    TimeoutError,
    OSError,
    _json.JSONDecodeError,
    KeyError,
    IndexError,
)


def _urlopen_read(req: object) -> bytes:
    """Sync helper for offloading urlopen via to_thread."""
    import urllib.request as _u
    with _u.urlopen(req, timeout=300) as resp:  # type: ignore[arg-type]
        return resp.read()


class LamuMcpServer:
    """MCP server that manages local models with budget-aware routing.

    Shares ``scheduler`` and ``health`` with the rest of the daemon — the
    OpenAI-compat layer reads the same VRAM map and the same health states,
    so a backend marked DEAD by an HTTP request also fails MCP routing.
    Pass ``None`` only in tests; production daemons construct one of each
    in ``daemon.cmd_start`` and hand them in here.
    """

    def __init__(
        self,
        models_dir: Path,
        registry_path: Path,
        scheduler: VramScheduler,
        health: Optional[HealthRegistry] = None,
    ) -> None:
        self._models_dir = models_dir
        self._registry_path = registry_path
        self._scheduler = scheduler

        # Load or scan registry
        entries = load_registry(registry_path)
        if not entries:
            entries = scan_directory(models_dir)
            write_registry(entries, registry_path)

        self._entries: dict[str, ModelEntry] = {e.name: e for e in entries}
        self._router = Router(scheduler, entries)
        self._server = Server("lamu")

        # TODO: backend pool (lazy init on first load)
        self._backends: dict[str, object] = {}
        # Shared with daemon + OpenAI compat. Default-construct only when not
        # supplied (test convenience); production always passes the daemon's.
        self._health = health if health is not None else HealthRegistry()
        # One Supervisor per loaded backend. Backend death runs through
        # supervisor.report_failure → restart-with-backoff → quarantine.
        # Keyed by model name; populated when load_model succeeds.
        self._supervisors: dict[str, Supervisor] = {}

        # Per-model request queues (concurrent agent serialization)
        strategy_str = os.environ.get("LAMU_QUEUE_STRATEGY", "fifo").lower()
        try:
            self._queue_strategy = QueueStrategy(strategy_str)
        except ValueError:
            self._queue_strategy = QueueStrategy.FIFO
        try:
            self._queue_concurrency = int(os.environ.get("LAMU_QUEUE_CONCURRENCY", "1"))
        except ValueError:
            self._queue_concurrency = 1
        self._queues: dict[str, RequestQueue] = {}

        self._register_tools()

    # ── Failure / success funnels ──────────────────────────────────────────
    # All backend success/failure observations route through these so we
    # have a single place to (a) update health, (b) drive supervisor
    # restart, (c) emit structured events. Direct `self._health.record_*`
    # calls would bypass supervisor — keep them out of the hot path.

    def _report_failure(self, model_name: str, exc: BaseException) -> None:
        """Funnel for backend failures.

        If a Supervisor exists for this model, the failure routes through it
        (which advances health state and triggers restart-with-backoff once
        the DEAD threshold is hit). Otherwise we fall back to a direct
        health update — useful in tests and during the load handshake before
        the supervisor is registered.
        """
        sup = self._supervisors.get(model_name)
        if sup is not None:
            sup.report_failure(exc)
        else:
            self._health.get_or_create(model_name).record_error(exc)

    def _report_success(self, model_name: str) -> None:
        """Funnel for backend successes — clears DEGRADED, never QUARANTINED."""
        self._health.get_or_create(model_name).record_success()

    def _register_tools(self) -> None:
        @self._server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="query",
                    description=(
                        "Send a prompt to a local LLM. Routes to the best model "
                        "based on capabilities or explicit model name. "
                        "Fast, free, uncensored, runs on your GPU."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "The prompt to send",
                            },
                            "model": {
                                "type": "string",
                                "description": (
                                    "Explicit model name (overrides capability routing). "
                                    "Use list_models to see available names."
                                ),
                            },
                            "capabilities": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Required capabilities: chat, code, reasoning, "
                                    "routing, long_context. Router loads matching model."
                                ),
                            },
                            "system": {
                                "type": "string",
                                "description": "Optional system prompt",
                                "default": "",
                            },
                            "max_tokens": {
                                "type": "integer",
                                "description": "Max tokens (default 16384)",
                                "default": 16384,
                            },
                            "temperature": {
                                "type": "number",
                                "description": "Sampling temperature 0-2 (default 0.7)",
                                "default": 0.7,
                            },
                            "include_reasoning": {
                                "type": "boolean",
                                "description": (
                                    "Return reasoning/thinking as separate field. "
                                    "False = stripped (default). True = structured blocks."
                                ),
                                "default": False,
                            },
                            "priority": {
                                "type": "integer",
                                "description": (
                                    "Queue priority for PRIORITY strategy. "
                                    "Higher = served first. Default 0."
                                ),
                                "default": 0,
                            },
                            "origin": {
                                "type": "string",
                                "description": (
                                    "Caller identifier (agent name, tool, etc.) "
                                    "for queue diagnostics. Default 'anonymous'."
                                ),
                                "default": "anonymous",
                            },
                        },
                        "required": ["prompt"],
                    },
                ),
                Tool(
                    name="plan_query",
                    description=(
                        "Dry-run: see which model WOULD handle a request without generating. "
                        "Returns routing decision, eviction plan, and reason. "
                        "Invaluable for debugging agent loops."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "The prompt (used for context, not sent to model)",
                            },
                            "model": {"type": "string"},
                            "capabilities": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["prompt"],
                    },
                ),
                Tool(
                    name="list_models",
                    description="List all known models with load status and capabilities.",
                    inputSchema={"type": "object", "properties": {}},
                ),
                Tool(
                    name="load_model",
                    description="Explicitly load a model onto GPU.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Model name from registry",
                            },
                        },
                        "required": ["name"],
                    },
                ),
                Tool(
                    name="unload_model",
                    description="Unload a model from GPU to free VRAM.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                        },
                        "required": ["name"],
                    },
                ),
                Tool(
                    name="vram_status",
                    description="Show current VRAM allocation and budget.",
                    inputSchema={"type": "object", "properties": {}},
                ),
                Tool(
                    name="scan_models",
                    description="Re-scan disk for new models and update registry.",
                    inputSchema={"type": "object", "properties": {}},
                ),
                Tool(
                    name="queue_status",
                    description="Show per-model queue depth and scheduling strategy.",
                    inputSchema={"type": "object", "properties": {}},
                ),
            ]

        @self._server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            if name == "query":
                return await self._handle_query(arguments)
            elif name == "plan_query":
                return self._handle_plan_query(arguments)
            elif name == "list_models":
                return self._handle_list_models()
            elif name == "load_model":
                return self._handle_load_model(arguments)
            elif name == "unload_model":
                return self._handle_unload_model(arguments)
            elif name == "vram_status":
                return self._handle_vram_status()
            elif name == "scan_models":
                return self._handle_scan()
            elif name == "queue_status":
                return await self._handle_queue_status()
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    def _get_or_create_queue(self, model_name: str) -> RequestQueue:
        q = self._queues.get(model_name)
        if q is None:
            q = RequestQueue(strategy=self._queue_strategy, concurrency=self._queue_concurrency)
            self._queues[model_name] = q
        return q

    async def _handle_query(self, args: dict) -> list[TextContent]:
        """Route and generate."""
        prompt = args["prompt"]
        model = args.get("model")
        caps_raw = args.get("capabilities")
        system = args.get("system", "")
        max_tokens = args.get("max_tokens", 16384)
        temperature = args.get("temperature", 0.7)
        include_reasoning = args.get("include_reasoning", False)
        priority = args.get("priority", 0)
        origin = args.get("origin", "anonymous")

        # Parse capabilities
        capabilities: Optional[list[Capability]] = None
        if caps_raw:
            capabilities = [Capability(c) for c in caps_raw]

        # Route — health_map filters out DEAD/QUARANTINED so a dying backend
        # never gets picked for a query. The router refuses with an explicit
        # reason instead of silently downgrading.
        decision = self._router.route(
            model=model,
            capabilities=capabilities,
            health_map=self._health.all() or None,
        )

        if not decision.model_name:
            return [TextContent(type="text", text=f"No model available: {decision.reason}")]

        # Router refuses unhealthy backends explicitly via decision.reason —
        # surface that to the caller as "no model available" rather than the
        # generic "not loaded" message, so an agent loop can stop retrying.
        if "unhealthy" in decision.reason:
            return [TextContent(
                type="text",
                text=f"No model available: {decision.reason}",
            )]

        if not decision.loaded:
            # TODO: trigger scheduler to load model
            return [TextContent(
                type="text",
                text=f"Model '{decision.model_name}' not loaded. Would need to load (evicting: {decision.would_evict}). "
                     f"Use load_model tool first, or use a loaded model.",
            )]

        # Get backend and generate
        loaded = self._scheduler.get_loaded(decision.model_name)
        if not loaded:
            return [TextContent(type="text", text="Internal error: model reported loaded but not found in scheduler")]

        # Mark used for LRU
        self._scheduler.mark_used(decision.model_name)

        # Generate via HTTP to the loaded backend
        import json
        import urllib.request

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        # Acquire queue slot — concurrent agents serialized per model
        queue = self._get_or_create_queue(decision.model_name)
        queue_req = QueueRequest(payload=None, priority=priority, origin=origin)

        try:
            async with await queue.enqueue(queue_req):
                req = urllib.request.Request(
                    f"http://localhost:{loaded.port}/v1/chat/completions",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                )
                import asyncio as _asyncio
                resp_data = await _asyncio.to_thread(_urlopen_read, req)
                data = json.loads(resp_data)
        except _BACKEND_HTTP_ERRORS as exc:
            self._report_failure(decision.model_name, exc)
            _log.warning(
                "mcp_query_backend_error model=%s err=%s",
                decision.model_name, exc,
            )
            raise BackendError(
                f"backend '{decision.model_name}' failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        # Mark backend healthy on success — routes through the funnel so any
        # supervisor wired up for this backend sees the success too.
        self._report_success(decision.model_name)

        msg = data["choices"][0]["message"]
        content = msg.get("content") or ""
        reasoning_content = msg.get("reasoning_content", "")

        entry = self._entries.get(decision.model_name)
        extractor = get_extractor(entry.reasoning_marker if entry else None)

        if reasoning_content:
            reasoning = reasoning_content
        else:
            reasoning, content = extractor.split(content)

        if include_reasoning and reasoning:
            result_text = f"**Reasoning:**\n{reasoning}\n\n**Answer:**\n{content}"
        else:
            result_text = content

        if not result_text.strip():
            result_text = (
                f"[Model thinking truncated — reasoning: {len(reasoning)} chars]"
            )

        return [TextContent(type="text", text=result_text)]

    def _handle_plan_query(self, args: dict) -> list[TextContent]:
        """Dry-run routing."""
        model = args.get("model")
        caps_raw = args.get("capabilities")

        capabilities: Optional[list[Capability]] = None
        if caps_raw:
            capabilities = [Capability(c) for c in caps_raw]

        decision = self._router.route(
            model=model,
            capabilities=capabilities,
            health_map=self._health.all() or None,
        )

        import json
        result = {
            "would_route_to": decision.model_name,
            "reason": decision.reason,
            "loaded": decision.loaded,
            "would_evict": list(decision.would_evict),
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    def _handle_list_models(self) -> list[TextContent]:
        """List all models with status."""
        lines: list[str] = []
        for name, entry in self._entries.items():
            loaded = self._scheduler.is_loaded(name)
            status = "🟢 loaded" if loaded else "⚪ available"
            caps = ", ".join(c.value for c in entry.capabilities)
            lines.append(
                f"{status} {name} ({entry.params_b}B {entry.quant}, "
                f"{entry.vram_mb}MB, [{caps}])"
            )
        return [TextContent(type="text", text="\n".join(lines))]

    def _spawn_backend(self, entry: ModelEntry, port: int) -> int:
        """Spawn llama-server for ``entry`` on ``port``. Return the PID.

        Pure spawn — no eviction, no scheduler accounting. The caller is
        responsible for scheduler.mark_loading + confirm_loaded. Used by both
        ``_handle_load_model`` (initial spawn) and ``Supervisor.restart_fn``
        (post-failure restart).

        Raises:
            BackendUnavailable: llama-server binary missing, or backend
                failed to come up healthy within 45s.
        """
        import subprocess
        import time as _time
        import urllib.request
        import json

        from lamu.core.config import LLAMA_BIN

        if not LLAMA_BIN.exists():
            raise BackendUnavailable(f"llama-server not found at {LLAMA_BIN}")

        cmd = [
            str(LLAMA_BIN), "-m", str(entry.path),
            "--host", "0.0.0.0", "--port", str(port),
            "--ctx-size", str(min(entry.context_max, 32768)),
            "-ngl", "99", "--flash-attn", "on",
            "--cache-type-k", "q4_0", "--cache-type-v", "q4_0",
            "--parallel", "1",
        ]

        try:
            help_out = subprocess.run(
                [str(LLAMA_BIN), "--help"], capture_output=True, text=True, timeout=5,
            )
            has_ngram = "--spec-ngram-mod-n-match" in help_out.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            _log.debug("ngram_probe_failed err=%s", exc)
            has_ngram = False

        if has_ngram and entry.arch in ("qwen35", "qwen3"):
            cmd.extend(["--spec-type", "ngram-mod",
                       "--spec-ngram-mod-n-match", "24",
                       "--spec-ngram-mod-n-min", "12",
                       "--spec-ngram-mod-n-max", "48"])

        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                stderr=open("/tmp/lamu-load.log", "w"))

        for _ in range(45):
            _time.sleep(1)
            try:
                req = urllib.request.Request(f"http://localhost:{port}/health")
                with urllib.request.urlopen(req, timeout=2) as resp:
                    health_payload = json.loads(resp.read())
            except _BACKEND_HTTP_ERRORS as exc:
                _log.debug("spawn_health_probe_failed port=%d err=%s", port, exc)
                continue
            if health_payload.get("status") == "ok":
                return proc.pid

        # Timeout — kill the orphan and surface a typed error.
        proc.kill()
        raise BackendUnavailable(
            f"backend '{entry.name}' did not come up healthy on :{port} "
            "within 45s; check /tmp/lamu-load.log"
        )

    def _restart_backend(self, name: str) -> None:
        """Supervisor restart hook. Re-spawns a dead backend with same args.

        Looks up the (entry, port) the backend was originally loaded with,
        re-runs ``_spawn_backend``, and updates the scheduler with the new
        PID. Raises whatever ``_spawn_backend`` raises so the supervisor's
        backoff sees the failure and either retries or quarantines.
        """
        loaded = self._scheduler.get_loaded(name)
        if loaded is None:
            raise BackendUnavailable(f"'{name}' is not registered in scheduler")
        entry = loaded.entry
        port = loaded.port
        new_pid = self._spawn_backend(entry, port)

        # Refresh actual VRAM from nvidia-smi if possible, else keep the prior
        # estimate. Don't let a transient nvidia-smi miss reset us to 0.
        pids = self._scheduler.query_gpu_pids()
        vram = next((m for p, m in pids if p == new_pid), 0) or loaded.vram_actual_mb
        self._scheduler.confirm_loaded(name, new_pid, port, vram)

    def _handle_load_model(self, args: dict) -> list[TextContent]:
        """Load a model onto GPU."""
        import os
        import signal
        import time as _time

        from lamu.core.errors import GpuUnavailableError

        name = args["name"]

        # No silent CPU fallback: refuse load if the GPU is in unavailable
        # state. The scheduler tracks this — see VramScheduler.require_gpu().
        try:
            self._scheduler.require_gpu()
        except GpuUnavailableError as exc:
            return [TextContent(
                type="text",
                text=f"GPU unavailable: {exc}. Cannot load '{name}'.",
            )]

        # Find in registry (partial match)
        entry: Optional[ModelEntry] = None
        for n, e in self._entries.items():
            if name in n or n in name:
                entry = e
                break

        if not entry:
            return [TextContent(type="text", text=f"Model '{name}' not found in registry. Use scan_models first.")]

        if self._scheduler.is_loaded(entry.name):
            return [TextContent(type="text", text=f"Model '{entry.name}' already loaded.")]

        # Check VRAM budget
        can_load, to_evict = self._scheduler.plan_load(entry)
        if not can_load:
            return [TextContent(type="text", text=f"Cannot fit '{entry.name}' ({entry.vram_mb}MB) in VRAM. Not enough space even after eviction.")]

        # Evict if needed
        for evict_name in to_evict:
            loaded = self._scheduler.get_loaded(evict_name)
            if loaded and loaded.pid:
                try:
                    os.kill(loaded.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self._scheduler.mark_unloaded(evict_name)
            # Tear down the supervisor and quarantine bookkeeping for the
            # evicted backend — its lifetime ends here.
            self._supervisors.pop(evict_name, None)

        if to_evict:
            _time.sleep(3)  # wait for VRAM to free

        # Pick a port
        from lamu.core.config import PORT_MAIN, PORT_SIDECAR
        port = PORT_SIDECAR
        if not self._scheduler.loaded_models():
            port = PORT_MAIN

        self._scheduler.mark_loading(entry)
        try:
            pid = self._spawn_backend(entry, port)
        except BackendUnavailable as exc:
            self._scheduler.mark_unloaded(entry.name)
            return [TextContent(type="text", text=f"Failed to load '{entry.name}': {exc}")]

        pids = self._scheduler.query_gpu_pids()
        vram = next((m for p, m in pids if p == pid), 0) or entry.vram_mb
        self._scheduler.confirm_loaded(entry.name, pid, port, vram)

        # Loading clears any prior quarantine (manual recovery path) and
        # registers a fresh Supervisor whose restart hook re-spawns this
        # exact backend with the same (entry, port).
        h = self._health.get_or_create(entry.name)
        if h.state.value == "quarantined":
            # Reset by replacing the BackendHealth wholesale.
            self._health._by_id[entry.name] = type(h)(backend_id=entry.name)
            h = self._health.get_or_create(entry.name)
        h.record_success()
        self._supervisors[entry.name] = Supervisor(
            health=h,
            restart_fn=lambda name=entry.name: self._restart_backend(name),
        )

        evict_msg = f" (evicted: {to_evict})" if to_evict else ""
        return [TextContent(
            type="text",
            text=f"Loaded '{entry.name}' on :{port} ({vram}MB VRAM){evict_msg}",
        )]

    def _handle_unload_model(self, args: dict) -> list[TextContent]:
        """Unload a model from GPU."""
        import os
        import signal

        name = args["name"]

        # Find loaded model (partial match)
        target: Optional[str] = None
        for n in list(self._scheduler._loaded.keys()):
            if name in n or n in name:
                target = n
                break

        if not target:
            return [TextContent(type="text", text=f"Model '{name}' not loaded. Nothing to unload.")]

        loaded = self._scheduler.get_loaded(target)
        if loaded and loaded.pid and loaded.pid > 0:
            try:
                os.kill(loaded.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        self._scheduler.mark_unloaded(target)
        # Stop watching this backend — its lifetime ends here.
        self._supervisors.pop(target, None)
        return [TextContent(type="text", text=f"Unloaded '{target}'. VRAM freed.")]

    def _handle_vram_status(self) -> list[TextContent]:
        """VRAM budget snapshot. Surfaces GPU unavailability as a typed reason."""
        budget = self._scheduler.budget()
        if not self._scheduler.gpu_available:
            return [TextContent(
                type="text",
                text=(
                    f"GPU unavailable: {self._scheduler.gpu_unavailable_reason}\n"
                    f"VRAM data is stale (last known total: {budget.total_mb} MB)."
                ),
            )]
        lines = [
            f"VRAM: {budget.used_mb}/{budget.total_mb} MB ({budget.free_mb} MB free)",
            f"Available for models: {budget.available_mb} MB",
            "Loaded:",
        ]
        for name, vram in budget.loaded_models:
            lines.append(f"  {name}: {vram} MB")
        if not budget.loaded_models:
            lines.append("  (none)")
        return [TextContent(type="text", text="\n".join(lines))]

    def _handle_scan(self) -> list[TextContent]:
        """Re-scan disk for models."""
        entries = scan_directory(self._models_dir)
        write_registry(entries, self._registry_path)
        self._entries = {e.name: e for e in entries}
        self._router.update_registry(entries)
        return [TextContent(
            type="text",
            text=f"Scanned {self._models_dir}: {len(entries)} models found. Registry updated.",
        )]

    async def _handle_queue_status(self) -> list[TextContent]:
        """Per-model queue depth + scheduling strategy."""
        lines = [
            f"Strategy: {self._queue_strategy.value} (concurrency={self._queue_concurrency})",
            "Per-model queue depth:",
        ]
        if not self._queues:
            lines.append("  (no queues active)")
        else:
            for name, q in self._queues.items():
                depth = await q.depth()
                lines.append(f"  {name}: {depth} pending")
        return [TextContent(type="text", text="\n".join(lines))]

    async def run(self) -> None:
        """Start the MCP server on stdio."""
        async with stdio_server() as (read, write):
            await self._server.run(read, write, self._server.create_initialization_options())
