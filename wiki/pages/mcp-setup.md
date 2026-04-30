# MCP Server Setup for Claude Code

## Configuration

In `~/.claude.json`:
```json
{
  "mcpServers": {
    "lamu": {
      "type": "stdio",
      "command": "/home/brianklam/local-llm/.venv/bin/python",
      "args": ["/home/brianklam/local-llm/server/mcp_qwen.py"]
    }
  }
}
```

## Available Tools

### query_local_llm
Send prompts to the local Qwen3.6 model.
- **model**: `qwen/qwen3.6-27b-uncensored` (default)
- **system**: optional system prompt
- **max_tokens**: default 4096
- **temperature**: default 0.3
- Falls back to direct :8020 if Bifrost is down

### list_local_models
Discovers running endpoints and shows available models.

## Reliability
- MCP server tries Bifrost (:8080) first, falls back to direct (:8020)
- Think blocks stripped at server level (middleware in serve.py)
- No client-side think-block handling needed

## Token Savings Pattern
Claude plans → query_local_llm implements → Claude reviews.
70-80% of generation tokens are free (local model).
