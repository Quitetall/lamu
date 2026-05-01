"""
Custom speculative decoding server using the trained EAGLE head.

Loads both the main model (4-bit) and EAGLE head on the same GPU.
EAGLE head predicts draft tokens from hidden states, main model verifies.

Usage:
    python -m server.eagle_server
    python -m server.eagle_server --draft-tokens 4 --port 8020
"""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── Config ──────────────────────────────────────────────────────────────

MODEL_ID = "llmfan46/Qwen3.6-27B-uncensored-heretic-v2"
EAGLE_DIR = Path.home() / "models" / "qwen3.6-27b-heretic-eagle" / "eagle_head"
INNER_DIM = 1024

app = FastAPI(title="LAMU EAGLE Server")


# ── EAGLE Head ──────────────────────────────────────────────────────────

class EagleHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, inner_dim=INNER_DIM):
        super().__init__()
        self.down = nn.Linear(hidden_size, inner_dim)
        self.norm = nn.LayerNorm(inner_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=inner_dim, nhead=8, dim_feedforward=inner_dim * 4,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=2)
        self.lm = nn.Linear(inner_dim, vocab_size, bias=False)

    def forward(self, x):
        return self.lm(self.enc(self.norm(self.down(x))))

    @torch.no_grad()
    def predict_next(self, hidden_state: torch.Tensor, n_draft: int = 4) -> list[int]:
        """Predict next n_draft tokens from a single hidden state vector."""
        # hidden_state: [hidden_size] — cast to match EAGLE head dtype
        hidden_state = hidden_state.to(dtype=next(self.parameters()).dtype)
        x = hidden_state.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_size]
        logits = self.forward(x)  # [1, 1, vocab_size]
        token = logits[0, 0].argmax().item()

        # For multi-token draft: autoregressively feed back through EAGLE
        # (simplified — uses embedding lookup instead of true hidden states)
        drafts = [token]
        # Single-token draft for now — multi-token needs embedding table
        return drafts


# ── Speculative Decoding Loop ───────────────────────────────────────────

class SpeculativeDecoder:
    def __init__(self, model, tokenizer, eagle_head, n_draft=4, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.eagle = eagle_head
        self.n_draft = n_draft
        self.device = device

        # Stats
        self.total_accepted = 0
        self.total_drafted = 0
        self.total_tokens = 0

    @torch.no_grad()
    def generate(self, input_ids, max_tokens=512, temperature=0.7, top_p=0.95):
        """Generate tokens with speculative decoding."""
        generated = []
        current_ids = input_ids.to(self.device)

        for _ in range(max_tokens):
            # Step 1: Main model forward pass — get hidden states + logits
            outputs = self.model(
                current_ids,
                output_hidden_states=True,
                use_cache=False,
            )
            main_logits = outputs.logits[0, -1]  # last token logits
            hidden = outputs.hidden_states[-2][0, -1]  # 2nd-to-last layer, last token

            # Sample from main model
            if temperature > 0:
                probs = torch.softmax(main_logits / temperature, dim=-1)
                token = torch.multinomial(probs, 1).item()
            else:
                token = main_logits.argmax().item()

            # Check for EOS
            if token in (self.tokenizer.eos_token_id, 248046):  # qwen3.6 EOS
                break

            generated.append(token)
            self.total_tokens += 1

            # Step 2: EAGLE head predicts draft tokens
            draft_tokens = self.eagle.predict_next(hidden, self.n_draft)
            self.total_drafted += len(draft_tokens)

            # Step 3: Verify drafts with main model
            # Append main token + drafts to input
            verify_ids = torch.cat([
                current_ids,
                torch.tensor([[token] + draft_tokens], device=self.device),
            ], dim=1)

            verify_out = self.model(verify_ids, output_hidden_states=True, use_cache=False)
            verify_logits = verify_out.logits[0]  # [seq_len, vocab]

            # Check which draft tokens match main model's predictions
            accepted = 0
            pos = current_ids.shape[1]  # position after original input
            for i, draft_t in enumerate(draft_tokens):
                main_pred = verify_logits[pos + i].argmax().item()
                if main_pred == draft_t:
                    accepted += 1
                    generated.append(draft_t)
                    self.total_tokens += 1
                    self.total_accepted += 1
                else:
                    # Rejection — use main model's prediction instead
                    generated.append(main_pred)
                    self.total_tokens += 1
                    break

            # Update input for next iteration
            n_new = 1 + accepted + (1 if accepted < len(draft_tokens) else 0)
            current_ids = torch.cat([
                current_ids,
                torch.tensor([generated[-n_new:]], device=self.device),
            ], dim=1)

            # Trim context if too long
            if current_ids.shape[1] > 2048:
                current_ids = current_ids[:, -1536:]

            yield from generated[-(1 + accepted + (1 if accepted < len(draft_tokens) else 0)):]
            generated_tail = generated  # keep for final return

        return

    @property
    def acceptance_rate(self):
        if self.total_drafted == 0:
            return 0.0
        return self.total_accepted / self.total_drafted


# ── API ─────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "qwen3.6-27b-uncensored"
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = False

decoder: SpeculativeDecoder = None


def strip_think(text):
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text


@app.get("/health")
async def health():
    rate = decoder.acceptance_rate if decoder else 0
    return {
        "status": "ok",
        "model": "qwen3.6-27b-uncensored",
        "engine": "eagle_speculative",
        "acceptance_rate": f"{rate:.1%}",
        "total_tokens": decoder.total_tokens if decoder else 0,
    }


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": "qwen3.6-27b-uncensored", "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    # Format messages
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    text = decoder.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    input_ids = decoder.tokenizer.encode(text, return_tensors="pt")

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if req.stream:
        async def sse():
            head = {"id": completion_id, "object": "chat.completion.chunk", "created": created,
                    "model": req.model, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
            yield f"data: {json.dumps(head)}\n\n"

            full_text = []
            think_done = False
            for token_id in decoder.generate(input_ids, req.max_tokens, req.temperature):
                tok_text = decoder.tokenizer.decode([token_id])
                full_text.append(tok_text)
                joined = "".join(full_text)

                if not think_done:
                    if "</think>" in joined:
                        think_done = True
                        after = joined.split("</think>", 1)[1].lstrip("\n")
                        if after:
                            chunk = {"id": completion_id, "object": "chat.completion.chunk", "created": created,
                                     "model": req.model, "choices": [{"index": 0, "delta": {"content": after}, "finish_reason": None}]}
                            yield f"data: {json.dumps(chunk)}\n\n"
                    continue

                chunk = {"id": completion_id, "object": "chat.completion.chunk", "created": created,
                         "model": req.model, "choices": [{"index": 0, "delta": {"content": tok_text}, "finish_reason": None}]}
                yield f"data: {json.dumps(chunk)}\n\n"

            tail = {"id": completion_id, "object": "chat.completion.chunk", "created": created,
                    "model": req.model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            yield f"data: {json.dumps(tail)}\n\ndata: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    # Non-streaming
    tokens = []
    for token_id in decoder.generate(input_ids, req.max_tokens, req.temperature):
        tokens.append(token_id)

    text = decoder.tokenizer.decode(tokens, skip_special_tokens=True)
    text = strip_think(text)

    return {
        "id": completion_id, "object": "chat.completion", "created": created,
        "model": req.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": input_ids.shape[1], "completion_tokens": len(tokens), "total_tokens": input_ids.shape[1] + len(tokens)},
        "eagle_stats": {"acceptance_rate": f"{decoder.acceptance_rate:.1%}", "total_accepted": decoder.total_accepted, "total_drafted": decoder.total_drafted},
    }


# ── Main ────────────────────────────────────────────────────────────────

def main():
    global decoder

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--draft-tokens", type=int, default=1, help="Draft tokens per step (1 for now)")
    args = parser.parse_args()

    print("LAMU EAGLE Speculative Decoding Server")
    print(f"  Loading main model (4-bit)...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        ),
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
        output_hidden_states=True,
    )
    model.eval()

    config = getattr(model.config, 'text_config', model.config)
    H = config.hidden_size
    V = config.vocab_size

    print(f"  Loading EAGLE head ({EAGLE_DIR})...")
    eagle = EagleHead(H, V, INNER_DIM).cuda()
    state = torch.load(EAGLE_DIR / "eagle_head_best.pt", map_location="cuda", weights_only=True)
    eagle.load_state_dict(state)
    eagle.eval()
    eagle.half()  # FP16 for inference

    print(f"  EAGLE head: {sum(p.numel() for p in eagle.parameters())/1e6:.0f}M params")

    decoder = SpeculativeDecoder(model, tokenizer, eagle, n_draft=args.draft_tokens)

    print(f"  Ready on :{args.port}")
    print(f"  Speculative decoding: EAGLE ({args.draft_tokens} draft tokens)")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
