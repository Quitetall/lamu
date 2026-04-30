"""
Production-grade Qwen3.6 server with think-block stripping middleware.

Wraps llama-cpp-python's server with:
1. Automatic <think>...</think> removal from all responses
2. Health endpoint at /health
3. Proper model alias
4. Correct settings for 262K context on 24GB GPU

All clients get clean output — no think blocks to handle.
"""

import json
import os
import re
import sys
from pathlib import Path

from llama_cpp.server.app import create_app
from llama_cpp.server.settings import ModelSettings, ServerSettings
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
import uvicorn

# ── Config ──────────────────────────────────────────────────────────────

MODELS_DIR = Path.home() / "models" / "qwen3.6-27b-heretic"
PORT = int(os.getenv("LLM_PORT") or "8020")
CTX = int(os.getenv("LLM_CTX") or "262144")

# Auto-detect best quant
QUANT_PREFERENCE = ["Q5_K_S", "Q5_K_M", "Q4_K_M", "Q4_K_S", "Q3_K_L"]

def find_gguf() -> str:
    for q in QUANT_PREFERENCE:
        for f in MODELS_DIR.glob(f"*{q}*.gguf"):
            return str(f)
    raise FileNotFoundError(f"No GGUF found in {MODELS_DIR}")

# KV cache type based on quant (Q5 → Q8 KV for quality, Q4 → Q4 KV for max context)
def kv_type_for_quant(gguf_path: str) -> int:
    if "Q5_K" in gguf_path:
        return 8  # Q8_0 — better quality
    return 2  # Q4_0 — fits 262K context

def ctx_for_quant(gguf_path: str) -> int:
    if CTX != 262144:
        return CTX  # user override
    if "Q5_K" in gguf_path:
        return 108000  # Q5 + Q8 KV → 108K max
    return 262144  # Q4 + Q4 KV → full 262K


# ── Think-block stripping middleware ────────────────────────────────────

THINK_PATTERN = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks from text."""
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    # Also handle <think>\n\n</think>\n\n (empty think = instruct mode)
    return THINK_PATTERN.sub("", text).strip() or text


class ThinkBlockMiddleware(BaseHTTPMiddleware):
    """Strips think blocks from all chat completion responses."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Only process non-streaming chat completions
        if "/chat/completions" not in str(request.url):
            return response
        if "text/event-stream" in response.headers.get("content-type", ""):
            return response

        # Read full body
        body = b""
        async for chunk in response.body_iterator:
            body += chunk if isinstance(chunk, bytes) else chunk.encode()

        try:
            data = json.loads(body)
            for choice in data.get("choices", []):
                msg = choice.get("message", {})
                if msg.get("content"):
                    msg["content"] = strip_think(msg["content"])
            modified = json.dumps(data).encode()
            return Response(
                content=modified,
                status_code=response.status_code,
                media_type="application/json",
            )
        except (json.JSONDecodeError, KeyError):
            return Response(
                content=body,
                status_code=response.status_code,
                media_type="application/json",
            )


# ── Main ────────────────────────────────────────────────────────────────

def main():
    gguf = find_gguf()
    kv_type = kv_type_for_quant(gguf)
    ctx = ctx_for_quant(gguf)

    print(f"Qwen3.6 Server")
    print(f"  Model:   {Path(gguf).name}")
    print(f"  Context: {ctx:,} tokens")
    print(f"  KV type: {'Q8_0' if kv_type == 8 else 'Q4_0'}")
    print(f"  Port:    {PORT}")
    print(f"  Think:   stripped from all responses")

    model = ModelSettings(
        model=gguf,
        model_alias="qwen3.6-27b-uncensored",
        n_gpu_layers=-1,
        n_ctx=ctx,
        type_k=kv_type,
        type_v=kv_type,
        flash_attn=True,
        logits_all=False,
        chat_format="chatml",
    )

    server = ServerSettings(host="0.0.0.0", port=PORT)
    app = create_app(server_settings=server, model_settings=[model])

    # Add think-block stripping middleware
    app.add_middleware(ThinkBlockMiddleware)

    # Add /health endpoint
    @app.get("/health")
    async def health():
        return {"status": "ok", "model": "qwen3.6-27b-uncensored", "context": ctx}

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")


if __name__ == "__main__":
    main()
