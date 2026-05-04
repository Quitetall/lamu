"""
OpenAI-compatible server for Qwen3.5-0.8B megakernel.
462+ tok/s on RTX 4090. Runs alongside 27B model (~1.5 GB VRAM).

    python server/megakernel_server.py --port 8001

Endpoints:
    GET  /health
    GET  /v1/models
    POST /v1/chat/completions
"""
import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional

# Add megakernel dir to path
MEGA_DIR = Path.home() / "local-llm" / "lucebox-hub" / "megakernel"
sys.path.insert(0, str(MEGA_DIR))

app = FastAPI()
decoder = None
tokenizer = None
MODEL_NAME = "megakernel/qwen3.5-0.8b"


def get_decoder():
    global decoder, tokenizer
    if decoder is None:
        from model import Decoder
        from transformers import AutoTokenizer
        print("Loading Qwen3.5-0.8B megakernel...", flush=True)
        decoder = Decoder(verbose=True)
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
        # Warmup
        decoder.reset()
        for t in tokenizer.encode("warmup", add_special_tokens=False):
            decoder.step(t)
        print(f"Ready. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)
    return decoder, tokenizer


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def models():
    return {
        "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "local"}],
        "object": "list",
    }


class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = MODEL_NAME
    messages: list[Message]
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.0
    stream: Optional[bool] = False


def generate(prompt_ids: list[int], max_tokens: int) -> tuple[list[int], float]:
    dec, tok = get_decoder()
    dec.reset()

    # Prefill
    for t in prompt_ids[:-1]:
        dec.step(t)

    # Decode
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = []
    next_id = prompt_ids[-1]
    for _ in range(max_tokens):
        next_id = dec.step(next_id)
        if next_id == tok.eos_token_id:
            break
        out.append(next_id)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return out, elapsed


def format_prompt(messages: list[Message]) -> str:
    """Simple chat template — Qwen3.5 format."""
    parts = []
    for msg in messages:
        if msg.role == "system":
            parts.append(f"<|im_start|>system\n{msg.content}<|im_end|>")
        elif msg.role == "user":
            parts.append(f"<|im_start|>user\n{msg.content}<|im_end|>")
        elif msg.role == "assistant":
            parts.append(f"<|im_start|>assistant\n{msg.content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    dec, tok = get_decoder()

    prompt_text = format_prompt(req.messages)
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)

    if req.stream:
        async def stream_gen():
            dec.reset()
            for t in prompt_ids[:-1]:
                dec.step(t)

            rid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            next_id = prompt_ids[-1]
            for i in range(req.max_tokens):
                next_id = dec.step(next_id)
                if next_id == tok.eos_token_id:
                    break
                token_text = tok.decode([next_id])
                chunk = {
                    "id": rid,
                    "object": "chat.completion.chunk",
                    "model": MODEL_NAME,
                    "choices": [{"index": 0, "delta": {"content": token_text}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            done_chunk = {
                "id": rid,
                "object": "chat.completion.chunk",
                "model": MODEL_NAME,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(done_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_gen(), media_type="text/event-stream")

    # Non-streaming
    out_ids, elapsed = generate(prompt_ids, req.max_tokens)
    content = tok.decode(out_ids, skip_special_tokens=True)

    # Strip think blocks
    if "</think>" in content:
        _, _, content = content.partition("</think>")
        content = content.strip()

    tps = len(out_ids) / elapsed if elapsed > 0 else 0

    return JSONResponse({
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "model": MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop" if len(out_ids) < req.max_tokens else "length",
        }],
        "usage": {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(out_ids),
            "total_tokens": len(prompt_ids) + len(out_ids),
        },
        "timings": {
            "predicted_n": len(out_ids),
            "predicted_ms": elapsed * 1000,
            "predicted_per_token_ms": (elapsed * 1000) / max(len(out_ids), 1),
        },
        "_meta": {"tok_per_sec": round(tps, 1)},
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    # Pre-load model
    get_decoder()

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
