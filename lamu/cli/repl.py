"""Interactive REPL targeting LAMU daemon's OpenAI compat endpoint.

Mirrors `lamu-rs/lamu-cli` style. Talks HTTP to localhost:8020/v1/chat/completions.
Slash commands hit the daemon's MCP/admin surface where useful, otherwise
operate locally on REPL state.
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, Optional

from rich.console import Console
from rich.rule import Rule


DEFAULT_API_URL = "http://localhost:8020/v1/chat/completions"
DEFAULT_LIST_URL = "http://localhost:8020/v1/models"
API_KEY = "sk-local"
DEFAULT_MODEL = "default"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class Message:
    role: Role
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role.value, "content": self.content}


@dataclass
class ReplState:
    """REPL state. Mutable container — NOT frozen."""
    api_url: str = DEFAULT_API_URL
    model: str = DEFAULT_MODEL
    history: list[Message] = field(default_factory=list)
    show_thinking: bool = False
    max_tokens: int = 16384
    temperature: float = 0.7


class Command(Enum):
    QUIT = "quit"
    MODEL = "model"
    MODELS = "models"
    LOAD = "load"
    UNLOAD = "unload"
    VRAM = "vram"
    THINK = "think"
    CLEAR = "clear"
    HELP = "help"


def parse_command(line: str) -> Optional[tuple[Command, str]]:
    """Parse '/cmd args...' into (Command, rest). None if not a command."""
    if not line.startswith("/"):
        return None
    parts = line[1:].strip().split(maxsplit=1)
    if not parts:
        return None
    name = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""
    for cmd in Command:
        if cmd.value == name:
            return (cmd, rest)
    return None


_HTTP_PROBE_ERRORS: tuple[type[BaseException], ...] = (
    urllib.error.URLError,
    ConnectionError,
    TimeoutError,
    OSError,
    json.JSONDecodeError,
)


def http_get_json(url: str, timeout: float = 3.0) -> Optional[dict]:
    """Fetch + parse JSON. Returns None on expected probe failures.

    Anything outside `_HTTP_PROBE_ERRORS` is a real bug and propagates.
    """
    try:
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {API_KEY}"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except _HTTP_PROBE_ERRORS:
        return None


def stream_chat(
    state: ReplState,
    user_msg: Message,
    console: Console,
) -> Optional[Message]:
    """Send chat completion request, stream response, return assistant message.

    Returns None on transport failure.
    """
    payload = {
        "model": state.model,
        "messages": [m.to_dict() for m in state.history] + [user_msg.to_dict()],
        "stream": True,
        "max_tokens": state.max_tokens,
        "temperature": state.temperature,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        state.api_url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=300)
    except urllib.error.URLError as e:
        console.print(f"[red]Connection error:[/red] {e}")
        return None

    full_content = ""
    in_think = False
    think_indicator_shown = False

    try:
        for chunk_text in iter_sse_deltas(resp):
            full_content += chunk_text
            visible = filter_think(chunk_text, in_think_ref=[in_think], state=state)
            # Detect think open/close transitions for indicator
            if THINK_OPEN in chunk_text:
                in_think = True
                if not state.show_thinking and not think_indicator_shown:
                    console.print("[dim italic](thinking…)[/]", end="")
                    think_indicator_shown = True
            if THINK_CLOSE in chunk_text:
                in_think = False
                if not state.show_thinking and think_indicator_shown:
                    console.print()
                    think_indicator_shown = False
            if visible:
                console.print(visible, end="", soft_wrap=True, highlight=False)
    except KeyboardInterrupt:
        console.print("\n[yellow](interrupted)[/]")
        return None

    console.print()
    # Strip think blocks from history-stored content
    stripped = strip_think_blocks(full_content)
    return Message(role=Role.ASSISTANT, content=stripped)


def iter_sse_deltas(resp) -> Iterator[str]:
    """Yield delta content strings from an OpenAI SSE stream."""
    for raw_line in resp:
        if not raw_line:
            continue
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        choices = obj.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        if content:
            yield content


def filter_think(chunk: str, *, in_think_ref: list[bool], state: ReplState) -> str:
    """Strip <think>...</think> from a chunk if not show_thinking.

    `in_think_ref` is a one-element list used as a mutable boolean across calls.
    """
    if state.show_thinking:
        return chunk
    out = []
    i = 0
    in_think = in_think_ref[0]
    while i < len(chunk):
        if not in_think:
            open_idx = chunk.find(THINK_OPEN, i)
            if open_idx == -1:
                out.append(chunk[i:])
                break
            out.append(chunk[i:open_idx])
            in_think = True
            i = open_idx + len(THINK_OPEN)
        else:
            close_idx = chunk.find(THINK_CLOSE, i)
            if close_idx == -1:
                break  # rest is inside think
            in_think = False
            i = close_idx + len(THINK_CLOSE)
    in_think_ref[0] = in_think
    return "".join(out)


def strip_think_blocks(text: str) -> str:
    """Remove all <think>...</think> blocks from text."""
    out = []
    i = 0
    while i < len(text):
        open_idx = text.find(THINK_OPEN, i)
        if open_idx == -1:
            out.append(text[i:])
            break
        out.append(text[i:open_idx])
        close_idx = text.find(THINK_CLOSE, open_idx)
        if close_idx == -1:
            break
        i = close_idx + len(THINK_CLOSE)
    return "".join(out).strip()


def handle_command(cmd: Command, args: str, state: ReplState, console: Console) -> bool:
    """Run slash command. Return False if REPL should exit."""
    if cmd is Command.QUIT:
        return False

    if cmd is Command.HELP:
        _print_help(console)
        return True

    if cmd is Command.MODEL:
        if not args:
            console.print(f"current model: [cyan]{state.model}[/]")
        else:
            state.model = args.strip()
            console.print(f"model → [cyan]{state.model}[/]")
        return True

    if cmd is Command.MODELS:
        data = http_get_json(DEFAULT_LIST_URL)
        if not data:
            console.print("[red]Could not reach daemon /v1/models[/]")
            return True
        for m in data.get("data") or []:
            console.print(f"  {m.get('id')}")
        return True

    if cmd is Command.LOAD or cmd is Command.UNLOAD or cmd is Command.VRAM:
        console.print(
            f"[yellow]/{cmd.value}[/] requires MCP transport "
            "(stdio). Use the daemon's MCP tools or `python -m lamu status`."
        )
        return True

    if cmd is Command.THINK:
        state.show_thinking = not state.show_thinking
        on = "ON" if state.show_thinking else "OFF"
        console.print(f"thinking display: [cyan]{on}[/]")
        return True

    if cmd is Command.CLEAR:
        state.history.clear()
        console.print("[dim]history cleared[/]")
        return True

    return True


def _print_help(console: Console) -> None:
    console.print(Rule("commands"))
    rows = [
        ("/quit", "exit"),
        ("/model [name]", "show or set model"),
        ("/models", "list models from daemon"),
        ("/load <name>", "(MCP only) load a model"),
        ("/unload <name>", "(MCP only) unload a model"),
        ("/vram", "(MCP only) show VRAM"),
        ("/think", "toggle reasoning visibility"),
        ("/clear", "clear conversation history"),
        ("/help", "this list"),
    ]
    for k, v in rows:
        console.print(f"  [cyan]{k:18}[/] {v}")


def run_repl(state: Optional[ReplState] = None) -> None:
    state = state or ReplState()
    console = Console()
    console.print(Rule("LAMU REPL — talking to daemon"))
    console.print(f"[dim]endpoint: {state.api_url} | model: {state.model} | /help[/]")

    while True:
        try:
            line = console.input("[bold green]>[/] ")
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not line.strip():
            continue

        cmd_parsed = parse_command(line)
        if cmd_parsed is not None:
            if not handle_command(cmd_parsed[0], cmd_parsed[1], state, console):
                break
            continue

        user_msg = Message(role=Role.USER, content=line)
        reply = stream_chat(state, user_msg, console)
        if reply is None:
            continue
        state.history.append(user_msg)
        state.history.append(reply)


def main() -> None:
    state = ReplState()
    if len(sys.argv) > 1:
        state.api_url = sys.argv[1]
    run_repl(state)


if __name__ == "__main__":
    main()
