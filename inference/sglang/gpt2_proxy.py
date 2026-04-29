#!/usr/bin/env python3
"""
GPT-2 preset proxy — maps shitty-* model names to sampling params,
forwards to SGLang on :8001. Runs on :9001.
"""

import json
import httpx
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

SGLANG_URL = "http://localhost:8001"
GPT2_MODEL = "gpt2-xl"

PRESETS = {
    "shitty-default":    {"temperature": 0.7,  "top_p": 1.0,  "max_tokens": 150},
    "shitty-inferkit":   {"temperature": 0.85, "top_p": 1.0,  "max_tokens": 120},
    "shitty-terrible":   {"temperature": 2.0,  "top_p": 1.0,  "max_tokens": 25},
    "shitty-incoherent": {"temperature": 3.0,  "top_p": 1.0,  "max_tokens": 15},
    "shitty-repetitive": {"temperature": 0.1,  "top_p": 1.0,  "max_tokens": 80},
    "shitty-coherent":   {"temperature": 0.5,  "top_p": 1.0,  "max_tokens": 200},
    "shitty-bloatware":  {"temperature": 2.2,  "top_p": 0.5,  "max_tokens": 28},
    "shitty-best2021":   {"temperature": 0.72, "top_p": 0.95, "max_tokens": 300},
}

app = FastAPI()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": name, "object": "model", "owned_by": "local"}
            for name in PRESETS
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model", "shitty-default")

    preset = PRESETS.get(model)
    if preset is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {model}. Available: {list(PRESETS)}",
        )

    # Apply preset defaults; caller can still override individual params
    fwd = dict(body)
    fwd["model"] = GPT2_MODEL
    for k, v in preset.items():
        fwd.setdefault(k, v)

    stream = fwd.get("stream", False)

    async with httpx.AsyncClient(timeout=120) as client:
        if stream:
            async def generate():
                async with client.stream(
                    "POST",
                    f"{SGLANG_URL}/v1/chat/completions",
                    json=fwd,
                    headers={"Authorization": "Bearer sk-local"},
                ) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk

            return StreamingResponse(generate(), media_type="text/event-stream")

        resp = await client.post(
            f"{SGLANG_URL}/v1/chat/completions",
            json=fwd,
            headers={"Authorization": "Bearer sk-local"},
        )
        return JSONResponse(content=resp.json(), status_code=resp.status_code)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9001, log_level="warning")
