"""
Production Qwen3.6 server — think-block stripping via ASGI middleware.
"""

import json
import os
import re
import sys
from pathlib import Path

import uvicorn

# ── Config ──────────────────────────────────────────────────────────────

MODELS_DIR = Path.home() / "models" / "qwen3.6-27b-heretic"
PORT = int(os.getenv("LLM_PORT") or "8020")
CTX = int(os.getenv("LLM_CTX") or "262144")
QUANT_PREFERENCE = ["Q5_K_S", "Q5_K_M", "Q4_K_M", "Q4_K_S", "Q3_K_L"]


def find_gguf(quant: str | None = None) -> str:
    quant = quant or os.getenv("LLM_QUANT")
    order = [quant] if quant else QUANT_PREFERENCE
    for q in order:
        for f in MODELS_DIR.glob(f"*{q}*.gguf"):
            return str(f)
    raise FileNotFoundError(f"No GGUF matching {order} in {MODELS_DIR}")


def kv_type_for_quant(p: str) -> int:
    return 8 if "Q5_K" in p else 2


def ctx_for_quant(p: str) -> int:
    if CTX != 262144:
        return CTX
    return 108000 if "Q5_K" in p else 262144


def strip_think(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip() or text


# ── ASGI Middleware ─────────────────────────────────────────────────────

class ThinkStripASGI:
    """Wraps the llama-cpp-python ASGI app and strips think blocks from responses."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or b"/chat/completions" not in scope.get("path", b"").encode() if isinstance(scope.get("path", ""), str) else b"/chat/completions" not in scope.get("path", b""):
            # Check path properly
            path = scope.get("path", "")
            if "/chat/completions" not in path:
                return await self.app(scope, receive, send)

        # Collect response to modify it
        response_started = False
        response_headers = []
        response_status = 200
        body_parts = []
        is_streaming = False

        async def intercept_send(message):
            nonlocal response_started, response_headers, response_status, is_streaming

            if message["type"] == "http.response.start":
                response_started = True
                response_status = message.get("status", 200)
                response_headers = dict(message.get("headers", []))

                # Check if streaming
                for k, v in message.get("headers", []):
                    if k == b"content-type" and b"text/event-stream" in v:
                        is_streaming = True
                        break

                if is_streaming:
                    # For streaming, pass through start and filter body chunks
                    await send(message)
                return

            if message["type"] == "http.response.body":
                chunk = message.get("body", b"")
                more = message.get("more_body", False)

                if is_streaming:
                    # Filter SSE stream in real-time
                    text = chunk.decode("utf-8", errors="replace")
                    filtered = self._filter_sse(text)
                    await send({"type": "http.response.body", "body": filtered.encode(), "more_body": more})
                    return

                body_parts.append(chunk)
                if not more:
                    # Complete body — strip thinks from JSON
                    full = b"".join(body_parts)
                    try:
                        data = json.loads(full)
                        for choice in data.get("choices", []):
                            msg = choice.get("message", {})
                            if msg.get("content"):
                                msg["content"] = strip_think(msg["content"])
                        modified = json.dumps(data).encode()
                    except (json.JSONDecodeError, KeyError):
                        modified = full

                    await send({
                        "type": "http.response.start",
                        "status": response_status,
                        "headers": [
                            [b"content-type", b"application/json"],
                            [b"content-length", str(len(modified)).encode()],
                        ],
                    })
                    await send({"type": "http.response.body", "body": modified, "more_body": False})

        self._think_done = False
        await self.app(scope, receive, intercept_send)

    def _filter_sse(self, text: str) -> str:
        lines = text.split("\n")
        out = []
        for line in lines:
            stripped = line.strip()
            if not stripped.startswith("data: "):
                out.append(line)
                continue
            if stripped == "data: [DONE]":
                out.append(line)
                continue
            payload = stripped[6:]
            try:
                data = json.loads(payload)
                content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if not content:
                    out.append(line)
                    continue
                if not self._think_done:
                    if "</think>" in content:
                        _, _, after = content.partition("</think>")
                        self._think_done = True
                        if after.strip():
                            data["choices"][0]["delta"]["content"] = after.lstrip("\n")
                            out.append(f"data: {json.dumps(data)}")
                    # suppress think content
                    continue
                out.append(line)
            except (json.JSONDecodeError, KeyError, IndexError):
                out.append(line)
        return "\n".join(out)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    from llama_cpp.server.app import create_app
    from llama_cpp.server.settings import ModelSettings, ServerSettings
    from starlette.requests import Request
    from starlette.responses import Response

    gguf = find_gguf()
    kv_type = kv_type_for_quant(gguf)
    ctx = ctx_for_quant(gguf)

    print(f"LAMU Server")
    print(f"  Model:    {Path(gguf).name}")
    print(f"  Context:  {ctx:,} tokens")
    print(f"  KV:       {'Q8_0' if kv_type == 8 else 'Q4_0'}")
    print(f"  Think:    stripped (ASGI middleware)")
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
        chat_format="chatml",
    )

    server_settings = ServerSettings(host="0.0.0.0", port=PORT)
    inner_app = create_app(server_settings=server_settings, model_settings=[model])

    # Add /health endpoint
    @inner_app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": "qwen3.6-27b-uncensored",
            "quant": Path(gguf).name,
            "context": ctx,
            "kv_type": "Q8_0" if kv_type == 8 else "Q4_0",
        }

    # Add /reload endpoint
    @inner_app.post("/reload")
    async def reload(request: Request):
        body = await request.json() if request.headers.get("content-type") == "application/json" else {}
        target_quant = body.get("quant")
        try:
            target_gguf = find_gguf(target_quant)
        except FileNotFoundError as e:
            return Response(content=json.dumps({"error": str(e)}), status_code=404, media_type="application/json")

        import threading
        def _restart():
            import time; time.sleep(0.5)
            env = os.environ.copy()
            if target_quant:
                env["LLM_QUANT"] = target_quant
            os.execve(sys.executable, [sys.executable] + sys.argv, env)

        threading.Thread(target=_restart, daemon=True).start()
        return {"status": "reloading", "to": Path(target_gguf).name, "context": ctx_for_quant(target_gguf)}

    # Wrap with think-stripping ASGI middleware
    app = ThinkStripASGI(inner_app)

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")


if __name__ == "__main__":
    main()
