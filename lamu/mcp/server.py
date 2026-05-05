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

import time
from pathlib import Path
from typing import Optional, Sequence

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from lamu.core.reasoning import get_extractor
from lamu.core.registry import load_registry, scan_directory, write_registry
from lamu.core.router import Router
from lamu.core.scheduler import VramScheduler
from lamu.core.types import (
    Capability,
    ModelEntry,
    QueryResult,
    QueryStats,
    RouteDecision,
    VramBudget,
)


class LamuMcpServer:
    """MCP server that manages local models with budget-aware routing."""

    def __init__(
        self,
        models_dir: Path,
        registry_path: Path,
        scheduler: VramScheduler,
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

        self._register_tools()

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
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    async def _handle_query(self, args: dict) -> list[TextContent]:
        """Route and generate."""
        prompt = args["prompt"]
        model = args.get("model")
        caps_raw = args.get("capabilities")
        system = args.get("system", "")
        max_tokens = args.get("max_tokens", 16384)
        temperature = args.get("temperature", 0.7)
        include_reasoning = args.get("include_reasoning", False)

        # Parse capabilities
        capabilities: Optional[list[Capability]] = None
        if caps_raw:
            capabilities = [Capability(c) for c in caps_raw]

        # Route
        decision = self._router.route(model=model, capabilities=capabilities)

        if not decision.model_name:
            return [TextContent(type="text", text=f"No model available: {decision.reason}")]

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

        try:
            req = urllib.request.Request(
                f"http://localhost:{loaded.port}/v1/chat/completions",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            t0 = time.monotonic()
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read())
            elapsed = time.monotonic() - t0

            msg = data["choices"][0]["message"]
            content = msg.get("content") or ""
            reasoning_content = msg.get("reasoning_content", "")

            # Apply reasoning extraction
            entry = self._entries.get(decision.model_name)
            extractor = get_extractor(entry.reasoning_marker if entry else None)

            if reasoning_content:
                # Server already separated them
                reasoning = reasoning_content
            else:
                reasoning, content = extractor.split(content)

            # Build response
            if include_reasoning and reasoning:
                result_text = f"**Reasoning:**\n{reasoning}\n\n**Answer:**\n{content}"
            else:
                result_text = content

            if not result_text.strip():
                result_text = f"[Model thinking truncated — reasoning: {len(reasoning)} chars]"

            return [TextContent(type="text", text=result_text)]

        except Exception as e:
            return [TextContent(type="text", text=f"Generation error: {e}")]

    def _handle_plan_query(self, args: dict) -> list[TextContent]:
        """Dry-run routing."""
        model = args.get("model")
        caps_raw = args.get("capabilities")

        capabilities: Optional[list[Capability]] = None
        if caps_raw:
            capabilities = [Capability(c) for c in caps_raw]

        decision = self._router.route(model=model, capabilities=capabilities)

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

    def _handle_load_model(self, args: dict) -> list[TextContent]:
        """Load a model onto GPU."""
        import subprocess
        import time as _time

        name = args["name"]

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
                import signal
                try:
                    import os
                    os.kill(loaded.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self._scheduler.mark_unloaded(evict_name)

        if to_evict:
            _time.sleep(3)  # wait for VRAM to free

        # Pick a port
        from lamu.core.config import PORT_MAIN, PORT_SIDECAR
        port = PORT_SIDECAR  # default to sidecar port
        if not self._scheduler.loaded_models():
            port = PORT_MAIN  # if nothing loaded, use main port

        # Start llama-server
        from lamu.core.config import LLAMA_BIN
        if not LLAMA_BIN.exists():
            return [TextContent(type="text", text=f"llama-server not found at {LLAMA_BIN}")]

        cmd = [
            str(LLAMA_BIN), "-m", str(entry.path),
            "--host", "0.0.0.0", "--port", str(port),
            "--ctx-size", str(min(entry.context_max, 32768)),
            "-ngl", "99", "--flash-attn", "on",
            "--cache-type-k", "q4_0", "--cache-type-v", "q4_0",
            "--parallel", "1",
        ]

        # Detect ngram-mod support (not available on DFlash PR branch)
        try:
            help_out = subprocess.run(
                [str(LLAMA_BIN), "--help"], capture_output=True, text=True, timeout=5
            )
            has_ngram = "--spec-ngram-mod-n-match" in help_out.stdout
        except Exception:
            has_ngram = False

        if has_ngram and entry.arch in ("qwen35", "qwen3"):
            cmd.extend(["--spec-type", "ngram-mod",
                       "--spec-ngram-mod-n-match", "24",
                       "--spec-ngram-mod-n-min", "12",
                       "--spec-ngram-mod-n-max", "48"])

        self._scheduler.mark_loading(entry)
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                stderr=open("/tmp/lamu-load.log", "w"))

        # Wait for health
        import urllib.request
        import json
        for _ in range(45):
            _time.sleep(1)
            try:
                req = urllib.request.Request(f"http://localhost:{port}/health")
                with urllib.request.urlopen(req, timeout=2) as resp:
                    if json.loads(resp.read()).get("status") == "ok":
                        # Get actual VRAM
                        from lamu.core.scheduler import _query_gpu_pids
                        pids = _query_gpu_pids()
                        vram = 0
                        for pid, mem in pids:
                            if pid == proc.pid:
                                vram = mem
                                break
                        if vram == 0:
                            vram = entry.vram_mb  # fallback to estimate

                        self._scheduler.confirm_loaded(entry.name, proc.pid, port, vram)
                        evict_msg = f" (evicted: {to_evict})" if to_evict else ""
                        return [TextContent(type="text",
                            text=f"Loaded '{entry.name}' on :{port} ({vram}MB VRAM){evict_msg}")]
            except Exception:
                pass

        # Timeout
        proc.kill()
        self._scheduler.mark_unloaded(entry.name)
        return [TextContent(type="text", text=f"Failed to load '{entry.name}' (timeout after 45s). Check /tmp/lamu-load.log")]

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
        return [TextContent(type="text", text=f"Unloaded '{target}'. VRAM freed.")]

    def _handle_vram_status(self) -> list[TextContent]:
        """VRAM budget snapshot."""
        budget = self._scheduler.budget()
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

    async def run(self) -> None:
        """Start the MCP server on stdio."""
        async with stdio_server() as (read, write):
            await self._server.run(read, write, self._server.create_initialization_options())
