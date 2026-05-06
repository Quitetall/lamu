"""LAMU Daemon — main entry point.

Usage:
  python -m lamu start              # start daemon (MCP server on stdio)
  python -m lamu scan               # scan models, write registry
  python -m lamu status             # show loaded models + VRAM
  python -m lamu serve [port]       # boot OpenAI-compat HTTP (default 8020)
  python -m lamu repl  [api_url]    # interactive REPL
"""
from __future__ import annotations

import asyncio
import json as _json
import logging
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from lamu.core.health import HealthRegistry
from lamu.core.registry import load_registry, scan_directory, write_registry
from lamu.core.scheduler import VramScheduler


_log = logging.getLogger(__name__)

MODELS_DIR = Path.home() / "models"
REGISTRY_PATH = Path.home() / "local-llm" / "config" / "models.yaml"

# Errors we expect from probing un-running services. Anything else is a real
# bug and must surface — never swallow.
_PROBE_EXPECTED_ERRORS: tuple[type[BaseException], ...] = (
    urllib.error.URLError,
    ConnectionError,
    TimeoutError,
    OSError,
    _json.JSONDecodeError,
    KeyError,
    IndexError,
)


def cmd_scan() -> None:
    """Scan ~/models/ and write registry."""
    entries = scan_directory(MODELS_DIR)
    write_registry(entries, REGISTRY_PATH)
    print(f"Discovered {len(entries)} models → {REGISTRY_PATH}")
    for e in entries:
        caps = ", ".join(c.value for c in e.capabilities)
        print(f"  {e.name}: {e.params_b}B {e.quant} ({e.vram_mb}MB) [{caps}]")


def cmd_status() -> None:
    """Show registry + VRAM status. Typed catches per probe so any unexpected
    exception surfaces instead of being silently swallowed."""
    entries = load_registry(REGISTRY_PATH)
    sched = VramScheduler()
    budget = sched.budget()

    print(f"VRAM: {budget.used_mb}/{budget.total_mb} MB ({budget.free_mb} MB free)")
    print(f"Models in registry: {len(entries)}")
    print()

    ports_to_check = [8020, 8001, 8000]
    for port in ports_to_check:
        try:
            req = urllib.request.Request(f"http://localhost:{port}/health")
            with urllib.request.urlopen(req, timeout=1) as resp:
                data = _json.loads(resp.read())
        except _PROBE_EXPECTED_ERRORS as exc:
            _log.debug("probe_failed port=%d err=%s", port, exc)
            print(f"  ⚪ :{port} — not running")
            continue

        if data.get("status") != "ok":
            print(f"  ⚪ :{port} — health!=ok")
            continue

        try:
            req2 = urllib.request.Request(f"http://localhost:{port}/v1/models")
            with urllib.request.urlopen(req2, timeout=1) as resp2:
                mdata = _json.loads(resp2.read())
                model_id = mdata["data"][0]["id"]
            print(f"  🟢 :{port} — {model_id}")
        except _PROBE_EXPECTED_ERRORS as exc:
            _log.debug("models_probe_failed port=%d err=%s", port, exc)
            print(f"  🟢 :{port} — running (unknown model)")


def cmd_start() -> None:
    """Start the LAMU MCP server.

    Constructs the daemon-wide singletons — one VramScheduler, one
    HealthRegistry — and hands them to every surface that follows. MCP
    today, OpenAI-compat tomorrow. Two surfaces seeing the same backend
    DEAD/QUARANTINED state is the whole point of the v3 wiring.
    """
    from lamu.mcp.server import LamuMcpServer

    sched = VramScheduler()
    health = HealthRegistry()

    # Auto-register already-running models. Typed catch — anything outside
    # _PROBE_EXPECTED_ERRORS is a real bug and must surface.
    for port in [8020, 8001]:
        try:
            req = urllib.request.Request(f"http://localhost:{port}/v1/models")
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = _json.loads(resp.read())
            model_id = data["data"][0]["id"]
        except _PROBE_EXPECTED_ERRORS as exc:
            _log.debug("auto_register_skip port=%d err=%s", port, exc)
            continue

        entries = load_registry(REGISTRY_PATH)
        for entry in entries:
            if entry.name in model_id.lower() or model_id.lower() in entry.name:
                pids = sched.query_gpu_pids()
                vram = sum(v for _, v in pids) // max(len(pids), 1)
                sched.register_loaded(entry, pid=0, port=port, vram_actual_mb=vram)
                # Adopted backend starts HEALTHY by virtue of probing OK.
                health.get_or_create(entry.name).record_success()
                break

    server = LamuMcpServer(
        models_dir=MODELS_DIR,
        registry_path=REGISTRY_PATH,
        scheduler=sched,
        health=health,
    )

    print("LAMU daemon starting (MCP stdio)...", file=sys.stderr)
    asyncio.run(server.run())


def cmd_serve(port: int = 8020) -> None:
    """Boot OpenAI-compatible HTTP server on the given port."""
    from lamu.api.openai_compat import serve

    serve(port=port)


def cmd_repl(api_url: Optional[str] = None) -> None:
    """Launch interactive REPL talking to the daemon."""
    from lamu.cli.repl import ReplState, run_repl

    state = ReplState()
    if api_url:
        state.api_url = api_url
    run_repl(state)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m lamu [scan|status|start|serve|repl]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "scan":
        cmd_scan()
    elif cmd == "status":
        cmd_status()
    elif cmd == "start":
        cmd_start()
    elif cmd == "serve":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8020
        cmd_serve(port=port)
    elif cmd == "repl":
        api_url = sys.argv[2] if len(sys.argv) > 2 else None
        cmd_repl(api_url=api_url)
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python -m lamu [scan|status|start|serve|repl]")
        sys.exit(1)


if __name__ == "__main__":
    main()
