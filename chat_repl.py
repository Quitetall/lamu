#!/usr/bin/env python3
"""
llm-chat — terminal REPL via Bifrost.
Buffers response, renders markdown with rich,
collapses Qwen3.5 <think> blocks into a subtle indicator.
"""

import json
import urllib.request
import urllib.error

from rich.console import Console
from rich.markdown import Markdown
from rich.rule import Rule
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner

BIFROST_URL = "http://localhost:8080/v1/chat/completions"
BIFROST_KEY = "sk-local"
DEFAULT_MODEL = "dflash/luce-dflash"
AVAILABLE_MODELS = [
    "dflash/luce-dflash",
    "gpt2/shitty-best2021",
    "gpt2/shitty-inferkit",
    "gpt2/shitty-coherent",
    "gpt2/shitty-repetitive",
    "gpt2/shitty-terrible",
    "gpt2/shitty-incoherent",
    "gpt2/shitty-bloatware",
    "gpt2/shitty-default",
]

console = Console()


def stream_response(messages, model):
    """
    Streams from Bifrost, buffers silently.
    Returns (reply_text, think_text).
    Handles Qwen3.5 <think>...</think> blocks:
      - spinner shows "thinking..." during think phase
      - spinner shows "generating..." during reply phase
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": 0.7,
    }
    req = urllib.request.Request(
        BIFROST_URL,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {BIFROST_KEY}",
        },
    )

    think_buf = []
    reply_buf = []
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

    with urllib.request.urlopen(req, timeout=120) as resp:
        with Live(console=console, refresh_per_second=8) as live:
            live.update(Spinner("dots", style="dim", text=" thinking..."))

            for token in _iter_tokens(resp):
                pending += token

                if not think_done:
                    # <think> is injected by the chat template, not the model.
                    # The model outputs think content first, then </think>, then reply.
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
                # No </think> found — entire output is the reply (model skipped thinking)
                reply_buf.append(pending)

    return "".join(reply_buf), "".join(think_buf)


def render_reply(reply: str, think: str):
    """Render the reply as rich markdown, with optional think summary."""
    console.print()
    if think:
        tok_count = len(think.split())
        console.print(
            Text.assemble(("  o ", "dim"), (f"thought for ~{tok_count} tokens", "dim italic"))
        )
    console.rule(style="dim")
    console.print(Markdown(reply))
    console.rule(style="dim")


def main():
    model = DEFAULT_MODEL
    history = []

    console.print()
    console.print(Rule("[bold]llm-chat[/bold]", style="bright_blue"))
    console.print(
        Text.assemble(
            ("  Model: ", "dim"),
            (model, "cyan bold"),
            ("  |  /model <name>  /clear  /quit", "dim"),
        )
    )
    console.print(
        Text.assemble(("  Models: ", "dim"), (", ".join(AVAILABLE_MODELS), "dim italic"))
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

        if user in ("/quit", "/exit"):
            console.print("[dim]bye.[/dim]")
            break

        if user == "/clear":
            history.clear()
            console.print("[dim]history cleared.[/dim]")
            continue

        if user.startswith("/model"):
            parts = user.split(maxsplit=1)
            if len(parts) < 2:
                console.print(
                    f"[dim]usage:[/dim] /model <name>\n[dim]available:[/dim] {', '.join(AVAILABLE_MODELS)}"
                )
                continue
            req_model = parts[1].strip()
            if req_model not in AVAILABLE_MODELS:
                console.print(f"[yellow]unknown:[/yellow] {req_model}")
                continue
            model = req_model
            console.print(f"[green]->[/green] [cyan bold]{model}[/cyan bold]")
            continue

        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        messages += history
        messages.append({"role": "user", "content": user})

        console.print()
        try:
            reply, think = stream_response(messages, model)
        except urllib.error.URLError as e:
            console.print(f"[red]error:[/red] {e}\n[dim]stack not running? try: llm[/dim]")
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
