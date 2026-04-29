"""
MCP server exposing local Qwen (DFlash) as a tool for Claude Code.

Gives Claude Code access to a `query_local_llm` tool that sends prompts
to your local Qwen3.5-27B running on DFlash. Claude can use this for:
  - Offloading bulk code generation to the free local model
  - Getting a second opinion on implementation approaches
  - Parallel drafting — Claude plans, Qwen implements
  - Running prompts that don't need cloud-tier reasoning

Transport: stdio (launched as subprocess by Claude Code)

Config (.mcp.json in project root):
  {
    "mcpServers": {
      "local-llm": {
        "type": "stdio",
        "command": "/home/YOU/local-llm/.venv/bin/python",
        "args": ["/home/YOU/local-llm/server/mcp_qwen.py"]
      }
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

DFLASH_URL = os.getenv("DFLASH_URL", "http://localhost:8000/v1/chat/completions")
BIFROST_URL = os.getenv("BIFROST_URL", "http://localhost:8080/v1/chat/completions")
API_KEY = os.getenv("LLM_API_KEY", "sk-local")

# Use Bifrost if available (routes by model name), fall back to DFlash direct
USE_BIFROST = os.getenv("MCP_USE_BIFROST", "1") == "1"

server = Server("local-llm")


def _chat(
    prompt: str,
    system: str = "",
    model: str = "qwen/qwen3.6-35b-uncensored",
    max_tokens: int = 2048,
    temperature: float = 0.3,
) -> str:
    """Send a chat completion request to the local LLM."""
    url = BIFROST_URL if USE_BIFROST else DFLASH_URL

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    # If going direct to DFlash (not through Bifrost), strip the provider prefix
    actual_model = model
    if not USE_BIFROST and "/" in model:
        actual_model = model.split("/", 1)[1]

    payload = {
        "model": actual_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            content = data["choices"][0]["message"]["content"]

            # Strip think blocks (Qwen3.5 template injects <think>)
            if "</think>" in content:
                _, _, content = content.partition("</think>")
            return content.strip()
    except urllib.error.URLError as e:
        return f"Error connecting to local LLM: {e}"
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        return f"Error parsing LLM response: {e}"


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="query_local_llm",
            description=(
                "Send a prompt to the local Qwen3.6-35B-A3B uncensored model (MoE, 3B active). "
                "Fast, free, never refuses. Use for bulk code generation, drafting "
                "implementations, second opinions, or tasks that don't need cloud reasoning. "
                "Best for well-specified implementation tasks with clear instructions."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The prompt to send to the local model",
                    },
                    "system": {
                        "type": "string",
                        "description": "Optional system prompt to set the model's behavior",
                        "default": "",
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum tokens to generate (default 2048)",
                        "default": 2048,
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Sampling temperature 0-2 (default 0.3, lower = more deterministic)",
                        "default": 0.3,
                    },
                },
                "required": ["prompt"],
            },
        ),
        Tool(
            name="local_llm_status",
            description="Check if the local LLM (DFlash/Qwen) is running and responsive.",
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
            system=arguments.get("system", ""),
            max_tokens=arguments.get("max_tokens", 2048),
            temperature=arguments.get("temperature", 0.3),
        )
        return [TextContent(type="text", text=result)]

    elif name == "local_llm_status":
        try:
            url = BIFROST_URL.replace("/chat/completions", "/models")
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {API_KEY}"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                models = [m["id"] for m in data.get("data", [])]
                return [TextContent(type="text", text=f"Local LLM is running. Available models: {models}")]
        except Exception as e:
            return [TextContent(type="text", text=f"Local LLM is not reachable: {e}")]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
