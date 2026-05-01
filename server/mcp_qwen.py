"""
MCP server — exposes all local LLMs as tools for Claude Code (or any MCP client).

Tools:
  query_local_llm    Send a prompt to any local model. Fast, free, uncensored.
  list_local_models   Discover what's running and available.

Configures in ~/.claude.json:
  "mcpServers": {
    "local-llm": {
      "type": "stdio",
      "command": "/home/brianklam/local-llm/.venv/bin/python",
      "args": ["/home/brianklam/local-llm/server/mcp_qwen.py"]
    }
  }
"""

import json
import os
import urllib.request
import urllib.error

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

BIFROST_URL = os.getenv("BIFROST_URL", "http://localhost:8080/v1")
API_KEY = os.getenv("LLM_API_KEY", "sk-local")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3.6-27b-uncensored")

# Known endpoints to probe for model discovery
ENDPOINTS = {
    "bifrost": "http://localhost:8080/v1",
    "qwen36": "http://localhost:8020/v1",
    "dflash": "http://localhost:8000/v1",
}

server = Server("lamu")


def _discover_models() -> dict[str, list[str]]:
    """Probe all known endpoints and return available models."""
    result = {}
    for name, base in ENDPOINTS.items():
        try:
            req = urllib.request.Request(
                f"{base}/models",
                headers={"Authorization": f"Bearer {API_KEY}"},
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read())
                models = [m["id"] for m in data.get("data", [])]
                if models:
                    result[name] = models
        except Exception:
            pass
    return result


def _chat(
    prompt: str,
    system: str = "",
    model: str = None,
    max_tokens: int = 4096,
    temperature: float = 0.3,
) -> str:
    """Send a chat completion to Bifrost (routes to the right backend)."""
    model = model or DEFAULT_MODEL

    # RAG: prepend relevant wiki context to the prompt
    try:
        from server.rag import WikiRAG
        rag = WikiRAG()
        wiki_context = rag.retrieve(prompt, max_pages=2)
        if wiki_context:
            prompt = f"Relevant context from knowledge base:\n{wiki_context}\n\nUser question:\n{prompt}"
    except Exception:
        pass

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    req = urllib.request.Request(
        f"{BIFROST_URL}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
    )

    # Try Bifrost first, fall back to direct endpoint
    endpoints = [
        f"{BIFROST_URL}/chat/completions",
        "http://localhost:8020/v1/chat/completions",
    ]

    for url in endpoints:
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {API_KEY}",
                },
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read())
                content = data["choices"][0]["message"]["content"]
                # Strip think blocks (belt and suspenders — server also strips)
                if "</think>" in content:
                    _, _, content = content.partition("</think>")
                return content.strip()
        except urllib.error.URLError:
            continue
        except Exception as e:
            return f"Error: {e}"

    return "Error: local LLM unreachable. Is the stack running? Start with: just start"


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="query_local_llm",
            description=(
                "Send a prompt to a local LLM running on this machine. "
                "Fast, free, and uncensored — never refuses implementation tasks. "
                "Use for: bulk code generation, drafting implementations, "
                "getting a second opinion, refactoring, or any task that "
                "doesn't require cloud-tier reasoning. "
                "Supports multiple models via the 'model' parameter."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The prompt to send to the local model",
                    },
                    "model": {
                        "type": "string",
                        "description": (
                            "Which model to use (provider/name format). "
                            "Options: qwen/qwen3.6-27b-uncensored (default, best), "
                            "dflash/luce-dflash (Qwen3.5-27B). "
                            "Use list_local_models to see what's running."
                        ),
                    },
                    "system": {
                        "type": "string",
                        "description": "Optional system prompt",
                        "default": "",
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Max tokens (default 4096)",
                        "default": 4096,
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Sampling temperature 0-2 (default 0.3)",
                        "default": 0.3,
                    },
                },
                "required": ["prompt"],
            },
        ),
        Tool(
            name="list_local_models",
            description=(
                "List all local LLM models currently running on this machine. "
                "Shows which endpoints are up and which models are available."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "query_local_llm":
        result = _chat(
            prompt=arguments["prompt"],
            model=arguments.get("model"),
            system=arguments.get("system", ""),
            max_tokens=arguments.get("max_tokens", 4096),
            temperature=arguments.get("temperature", 0.3),
        )
        return [TextContent(type="text", text=result)]

    elif name == "list_local_models":
        available = _discover_models()
        if not available:
            return [TextContent(
                type="text",
                text="No local models are running. Start with: just start",
            )]
        lines = []
        for endpoint, model_list in available.items():
            lines.append(f"{endpoint}: {', '.join(model_list)}")
        return [TextContent(type="text", text="\n".join(lines))]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
