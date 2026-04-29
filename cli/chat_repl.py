#!/usr/bin/env python3
"""
llm — local LLM terminal chat.

Auto-discovers running models. Auto-starts the stack if nothing is up.
Streams responses with think-block collapsing and rich markdown rendering.

Usage:
  llm                     # chat with default model
  llm --model dflash/luce-dflash  # pick a specific model
  llm --direct 8020       # bypass Bifrost, hit port directly
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.request
import urllib.error

from rich.console import Console
from rich.markdown import Markdown
from rich.rule import Rule
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table

ROOT = os.path.expanduser("~/local-llm")
BIFROST_URL = "http://localhost:8080/v1/chat/completions"
BIFROST_KEY = "sk-local"
DEFAULT_MODEL = "qwen/qwen3.6-35b-uncensored"

# Endpoints to probe for model discovery
ENDPOINTS = {
    "bifrost":  ("http://localhost:8080/v1/models", "gateway"),
    "qwen36":   ("http://localhost:8020/v1/models", "Qwen3.6 uncensored"),
    "dflash":   ("http://localhost:8000/v1/models", "Qwen3.5 DFlash"),
    "sglang":   ("http://localhost:8001/v1/models", "SGLang"),
    "gpt2proxy": ("http://localhost:9001/v1/models", "GPT-2 presets"),
}

console = Console()


# ── Discovery ────────────────────────────────────────────────────────────

def probe_endpoint(url: str, timeout: int = 2) -> list[str]:
    try:
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {BIFROST_KEY}"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return [m["id"] for m in data.get("data", [])]
    except Exception:
        return []


def discover_models() -> dict[str, list[str]]:
    """Probe all endpoints, return {endpoint_name: [model_ids]}."""
    found = {}
    for name, (url, _) in ENDPOINTS.items():
        models = probe_endpoint(url)
        if models:
            found[name] = models
    return found


def get_available_models() -> list[str]:
    """Return flat list of Bifrost-routable model names."""
    # If Bifrost is up, it knows all routes
    bifrost_models = probe_endpoint("http://localhost:8080/v1/models")
    if bifrost_models:
        # Bifrost returns models as provider/name already
        return bifrost_models

    # Fallback: direct model names from individual endpoints
    models = []
    for name, (url, _) in ENDPOINTS.items():
        if name == "bifrost":
            continue
        models.extend(probe_endpoint(url))
    return models


def auto_start():
    """Start the LLM stack if nothing is running."""
    console.print("[dim]No models detected. Starting stack...[/dim]")

    # Try Qwen3.6 first (primary), then DFlash (fallback)
    scripts = [
        ("Qwen3.6", f"{ROOT}/scripts/serve-qwen36.sh"),
        ("Bifrost", f"{ROOT}/scripts/serve-bifrost.sh"),
    ]

    for label, script in scripts:
        if os.path.exists(script):
            console.print(f"  [dim]Starting {label}...[/dim]")
            subprocess.run(["bash", script], capture_output=True)

    # Verify something came up
    import time
    for _ in range(5):
        if probe_endpoint("http://localhost:8020/v1/models") or probe_endpoint("http://localhost:8000/v1/models"):
            return True
        time.sleep(1)
    return False


# ── Streaming ────────────────────────────────────────────────────────────

def stream_response(messages: list[dict], model: str, url: str = BIFROST_URL) -> tuple[str, str]:
    """Stream from backend, buffer silently. Returns (reply, think)."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": 0.7,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {BIFROST_KEY}",
        },
    )

    think_buf, reply_buf = [], []
    think_done = False
    pending = ""

    def _iter_tokens(resp):
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            chunk = line[6:]
            if chunk == "[DONE]":
                break
            try:
                delta = json.loads(chunk)["choices"][0]["delta"].get("content", "")
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
            if delta:
                yield delta

    with urllib.request.urlopen(req, timeout=180) as resp:
        with Live(console=console, refresh_per_second=8) as live:
            live.update(Spinner("dots", style="dim", text=" thinking..."))

            for token in _iter_tokens(resp):
                pending += token

                if not think_done:
                    if "</think>" in pending:
                        think_part, _, rest = pending.partition("</think>")
                        think_buf.append(think_part)
                        think_done = True
                        pending = rest
                        live.update(Spinner("dots", style="dim", text=" generating..."))
                    else:
                        think_buf.append(pending)
                        pending = ""

                if think_done and pending:
                    reply_buf.append(pending)
                    pending = ""

            if pending:
                reply_buf.append(pending)

    return "".join(reply_buf), "".join(think_buf)


def render_reply(reply: str, think: str):
    console.print()
    if think:
        tok_count = len(think.split())
        console.print(
            Text.assemble(("  o ", "dim"), (f"thought for ~{tok_count} tokens", "dim italic"))
        )
    console.rule(style="dim")
    console.print(Markdown(reply))
    console.rule(style="dim")


# ── Commands ─────────────────────────────────────────────────────────────

def cmd_status():
    """Show what's running."""
    table = Table(show_header=False, padding=(0, 2), box=None)
    table.add_column(style="bold")
    table.add_column(style="dim")
    table.add_column()

    for name, (url, label) in ENDPOINTS.items():
        models = probe_endpoint(url)
        if models:
            table.add_row(f"  {label}", f":{url.split(':')[2].split('/')[0]}", f"[green]{', '.join(models)}[/green]")
        else:
            table.add_row(f"  {label}", f":{url.split(':')[2].split('/')[0]}", "[dim]down[/dim]")

    console.print(table)


def cmd_models():
    """List available models."""
    models = get_available_models()
    if not models:
        console.print("[yellow]No models available.[/yellow] Run: [bold]just start[/bold]")
        return
    for m in models:
        console.print(f"  [cyan]{m}[/cyan]")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Local LLM chat", add_help=False)
    parser.add_argument("--model", "-m", default=None, help="Model to use")
    parser.add_argument("--direct", type=int, default=None, help="Bypass Bifrost, hit port directly")
    parser.add_argument("prompt", nargs="*", help="One-shot prompt (skip REPL)")
    args = parser.parse_args()

    # Determine API URL
    if args.direct:
        api_url = f"http://localhost:{args.direct}/v1/chat/completions"
    else:
        api_url = BIFROST_URL

    # One-shot mode: llm "what is quicksort"
    if args.prompt:
        prompt = " ".join(args.prompt)
        model = args.model or DEFAULT_MODEL
        try:
            reply, think = stream_response(
                [{"role": "user", "content": prompt}], model, api_url
            )
            if reply.strip():
                render_reply(reply.strip(), think)
        except urllib.error.URLError:
            console.print("[red]error:[/red] can't reach LLM. Run [bold]just start[/bold]")
        return

    # Check if anything is running
    available = get_available_models()
    if not available:
        if not auto_start():
            console.print("[red]Could not start any models.[/red] Run [bold]just start[/bold] manually.")
            return
        available = get_available_models()

    # Pick model
    model = args.model or DEFAULT_MODEL
    if model not in available and available:
        # Default not available, pick first available
        model = available[0]

    history = []

    console.print()
    console.print(Rule("[bold]llm[/bold]", style="bright_blue"))
    console.print(
        Text.assemble(
            ("  Model: ", "dim"),
            (model, "cyan bold"),
        )
    )
    console.print(
        Text.assemble(
            ("  ", ""),
            ("/model", "dim bold"), (" switch  ", "dim"),
            ("/models", "dim bold"), (" list  ", "dim"),
            ("/status", "dim bold"), ("  ", "dim"),
            ("/clear", "dim bold"), ("  ", "dim"),
            ("/quit", "dim bold"),
        )
    )
    console.print()

    while True:
        try:
            console.print("[bold bright_blue]>[/bold bright_blue] ", end="")
            user = input().strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye.[/dim]")
            break

        if not user:
            continue

        if user in ("/quit", "/exit", "/q"):
            console.print("[dim]bye.[/dim]")
            break

        if user == "/clear":
            history.clear()
            console.print("[dim]history cleared.[/dim]")
            continue

        if user == "/status":
            cmd_status()
            continue

        if user in ("/models", "/model list"):
            cmd_models()
            continue

        if user.startswith("/model"):
            parts = user.split(maxsplit=1)
            if len(parts) < 2 or parts[1] == "list":
                cmd_models()
                continue
            req_model = parts[1].strip()
            current_available = get_available_models()
            # Allow partial match
            matches = [m for m in current_available if req_model in m]
            if len(matches) == 1:
                model = matches[0]
                console.print(f"[green]->[/green] [cyan bold]{model}[/cyan bold]")
            elif len(matches) > 1:
                console.print(f"[yellow]ambiguous:[/yellow] {', '.join(matches)}")
            elif req_model in current_available:
                model = req_model
                console.print(f"[green]->[/green] [cyan bold]{model}[/cyan bold]")
            else:
                console.print(f"[yellow]not found:[/yellow] {req_model}")
                console.print(f"[dim]available:[/dim] {', '.join(current_available)}")
            continue

        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        messages += history
        messages.append({"role": "user", "content": user})

        console.print()
        try:
            reply, think = stream_response(messages, model, api_url)
        except urllib.error.URLError as e:
            console.print(f"[red]error:[/red] {e}")
            console.print("[dim]stack not running? try: just start[/dim]")
            continue

        if reply.strip():
            render_reply(reply.strip(), think)

        history.append({"role": "user", "content": user})
        history.append({"role": "assistant", "content": reply})
        if len(history) > 40:
            history = history[-40:]
        console.print()


if __name__ == "__main__":
    main()
