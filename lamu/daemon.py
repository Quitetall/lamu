"""LAMU Daemon — main entry point.

Usage:
  python -m lamu start       # start daemon (MCP server on stdio)
  python -m lamu scan        # scan models, write registry
  python -m lamu status      # show loaded models + VRAM
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from lamu.core.registry import load_registry, scan_directory, write_registry
from lamu.core.scheduler import VramScheduler


MODELS_DIR = Path.home() / "models"
REGISTRY_PATH = Path.home() / "local-llm" / "config" / "models.yaml"


def cmd_scan() -> None:
    """Scan ~/models/ and write registry."""
    entries = scan_directory(MODELS_DIR)
    write_registry(entries, REGISTRY_PATH)
    print(f"Discovered {len(entries)} models → {REGISTRY_PATH}")
    for e in entries:
        caps = ", ".join(c.value for c in e.capabilities)
        print(f"  {e.name}: {e.params_b}B {e.quant} ({e.vram_mb}MB) [{caps}]")


def cmd_status() -> None:
    """Show registry + VRAM status."""
    entries = load_registry(REGISTRY_PATH)
    sched = VramScheduler()
    budget = sched.budget()

    print(f"VRAM: {budget.used_mb}/{budget.total_mb} MB ({budget.free_mb} MB free)")
    print(f"Models in registry: {len(entries)}")
    print()

    # Check which are actually running by probing ports
    import urllib.request
    import json

    ports_to_check = [8020, 8001, 8000]
    for port in ports_to_check:
        try:
            req = urllib.request.Request(f"http://localhost:{port}/health")
            with urllib.request.urlopen(req, timeout=1) as resp:
                data = json.loads(resp.read())
                if data.get("status") == "ok":
                    # Try to get model name
                    try:
                        req2 = urllib.request.Request(f"http://localhost:{port}/v1/models")
                        with urllib.request.urlopen(req2, timeout=1) as resp2:
                            mdata = json.loads(resp2.read())
                            model_id = mdata["data"][0]["id"]
                            print(f"  🟢 :{port} — {model_id}")
                    except Exception:
                        print(f"  🟢 :{port} — running (unknown model)")
        except Exception:
            print(f"  ⚪ :{port} — not running")


def cmd_start() -> None:
    """Start the LAMU MCP server."""
    from lamu.mcp.server import LamuMcpServer

    sched = VramScheduler()

    # Auto-register already-running models
    import urllib.request
    import json

    for port in [8020, 8001]:
        try:
            req = urllib.request.Request(f"http://localhost:{port}/v1/models")
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read())
                model_id = data["data"][0]["id"]

                # Find in registry
                entries = load_registry(REGISTRY_PATH)
                for entry in entries:
                    if entry.name in model_id.lower() or model_id.lower() in entry.name:
                        # Estimate VRAM from nvidia-smi
                        from lamu.core.scheduler import _query_gpu_pids
                        pids = _query_gpu_pids()
                        vram = sum(v for _, v in pids) // max(len(pids), 1)
                        sched.register_loaded(entry, pid=0, port=port, vram_actual_mb=vram)
                        break
        except Exception:
            pass

    server = LamuMcpServer(
        models_dir=MODELS_DIR,
        registry_path=REGISTRY_PATH,
        scheduler=sched,
    )

    print("LAMU daemon starting (MCP stdio)...", file=sys.stderr)
    asyncio.run(server.run())


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m lamu [scan|status|start]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "scan":
        cmd_scan()
    elif cmd == "status":
        cmd_status()
    elif cmd == "start":
        cmd_start()
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python -m lamu [scan|status|start]")
        sys.exit(1)


if __name__ == "__main__":
    main()
