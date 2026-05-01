"""
Production Qwen3.6 server — native chat template, think-block middleware, health.

Key features:
1. Uses the model's native Jinja2 chat template (NOT chatml) — enables tool calling
2. Strips <think>...</think> from ALL responses (streaming and non-streaming)
3. Health endpoint at /health
4. Auto-selects best quant + optimal KV cache config
5. Correct settings for 262K context on 24GB GPU
"""

import json
import os
import re
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

QUANT_PREFERENCE = ["Q5_K_S", "Q5_K_M", "Q4_K_M", "Q4_K_S", "Q3_K_L"]


def find_gguf() -> str:
    for q in QUANT_PREFERENCE:
        for f in MODELS_DIR.glob(f"*{q}*.gguf"):
            return str(f)
    raise FileNotFoundError(f"No GGUF found in {MODELS_DIR}")


def kv_type_for_quant(gguf_path: str) -> int:
    if "Q5_K" in gguf_path:
        return 8  # Q8_0
    return 2  # Q4_0


def ctx_for_quant(gguf_path: str) -> int:
    if CTX != 262144:
        return CTX
    if "Q5_K" in gguf_path:
        return 108000
    return 262144


# ── Think-block stripping ───────────────────────────────────────────────

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_think(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return THINK_RE.sub("", text).strip() or text


class ThinkBlockMiddleware(BaseHTTPMiddleware):
    """Strips think blocks from ALL chat completion responses (streaming + non-streaming)."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        if "/chat/completions" not in str(request.url):
            return response

        content_type = response.headers.get("content-type", "")

        # ── Streaming (SSE) ──────────────────────────────────────────
        if "text/event-stream" in content_type:
            async def filter_stream():
                think_buf = []
                think_done = False

                async for chunk in response.body_iterator:
                    text = chunk if isinstance(chunk, str) else chunk.decode()

                    for line in text.split("\n"):
                        line = line.strip()
                        if not line or not line.startswith("data: "):
                            if line:
                                yield line + "\n"
                            else:
                                yield "\n"
                            continue

                        payload = line[6:]
                        if payload == "[DONE]":
                            yield "data: [DONE]\n\n"
                            continue

                        try:
                            data = json.loads(payload)
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")

                            if not content:
                                yield f"data: {payload}\n\n"
                                continue

                            if not think_done:
                                if "</think>" in content:
                                    _, _, after = content.partition("</think>")
                                    think_done = True
                                    after = after.lstrip("\n")
                                    if after:
                                        delta["content"] = after
                                        data["choices"][0]["delta"] = delta
                                        yield f"data: {json.dumps(data)}\n\n"
                                    continue
                                else:
                                    # Still in think block — suppress
                                    continue

                            yield f"data: {json.dumps(data)}\n\n"
                        except (json.JSONDecodeError, KeyError, IndexError):
                            yield f"data: {payload}\n\n"

            return StreamingResponse(
                filter_stream(),
                media_type="text/event-stream",
                headers={k: v for k, v in response.headers.items() if k.lower() != "content-length"},
            )

        # ── Non-streaming (JSON) ─────────────────────────────────────
        body = b""
        async for chunk in response.body_iterator:
            body += chunk if isinstance(chunk, bytes) else chunk.encode()

        try:
            data = json.loads(body)
            for choice in data.get("choices", []):
                msg = choice.get("message", {})
                if msg.get("content"):
                    msg["content"] = strip_think(msg["content"])
            return Response(
                content=json.dumps(data).encode(),
                status_code=response.status_code,
                media_type="application/json",
            )
        except (json.JSONDecodeError, KeyError):
            return Response(content=body, status_code=response.status_code, media_type="application/json")


# ── Main ────────────────────────────────────────────────────────────────

def main():
    gguf = find_gguf()
    kv_type = kv_type_for_quant(gguf)
    ctx = ctx_for_quant(gguf)

    print(f"LAMU Server")
    print(f"  Model:    {Path(gguf).name}")
    print(f"  Context:  {ctx:,} tokens")
    print(f"  KV type:  {'Q8_0' if kv_type == 8 else 'Q4_0'}")
    print(f"  Template: native Qwen3.6 (from GGUF, tool calling enabled)")
    print(f"  Think:    stripped from all responses (streaming + non-streaming)")
    print(f"  Port:     {PORT}")

    model = ModelSettings(
        model=gguf,
        model_alias="qwen3.6-27b-uncensored",
        n_gpu_layers=-1,
        n_ctx=ctx,
        type_k=kv_type,
        type_v=kv_type,
        flash_attn=True,
        logits_all=False,
        # chat_format=None → uses model's native Jinja2 template from GGUF
        # Enables Qwen3.6's tool calling syntax:
        #   <tool_call><function=name><parameter=p>value</parameter></function></tool_call>
    )

    server = ServerSettings(host="0.0.0.0", port=PORT)
    app = create_app(server_settings=server, model_settings=[model])

    app.add_middleware(ThinkBlockMiddleware)

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": "qwen3.6-27b-uncensored", "context": ctx}

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")


if __name__ == "__main__":
    main()
