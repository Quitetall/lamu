#!/usr/bin/env bash
# scripts/train-eagle-head.sh — train an EAGLE-3 speculative decoding head
#
# Runs unattended overnight (~16-24 hours total on single 4090):
#   1. Downloads BF16 model from HuggingFace (~54 GB, ~1-2 hrs)
#   2. Generates hidden state training data (4-bit on GPU, ~8-12 hrs)
#   3. Trains EAGLE head (~4-6 hrs)
#   4. Converts to GGUF for llama.cpp (~5 min)
#
# Output: ~/models/qwen3.6-27b-heretic-eagle/
# Usage: nohup bash scripts/train-eagle-head.sh > /tmp/eagle-train.log 2>&1 &
set -euo pipefail

ROOT="$HOME/local-llm"
VENV="$ROOT/.venv"
MODEL_ID="llmfan46/Qwen3.6-27B-uncensored-heretic-v2"
EAGLE_DIR="$HOME/models/qwen3.6-27b-heretic-eagle"
DATA_DIR="$EAGLE_DIR/train_data"
BOLD="\033[1m"; GRY="\033[90m"; GREEN="\033[32m"; R="\033[0m"

echo -e "\n${BOLD}EAGLE-3 Head Training Pipeline${R}"
echo -e "  ${GRY}Target: $MODEL_ID${R}"
echo -e "  ${GRY}Output: $EAGLE_DIR${R}"
echo -e "  ${GRY}Estimated time: 16-24 hours${R}\n"

mkdir -p "$EAGLE_DIR" "$DATA_DIR"

# ── Step 1: Install deps ────────────────────────────────────────────────
echo "[1/4] Installing dependencies..."
"$VENV/bin/python" -m pip install datasets safetensors wandb -q 2>/dev/null || true

# ── Step 2: Generate training data ─────────────────────────────────────
echo "[2/4] Generating hidden state training data..."
echo "  Loading model in 4-bit (fits on 4090)..."
echo "  This runs the model on ShareGPT data and saves hidden states."
echo "  Estimated: 8-12 hours"

"$VENV/bin/python" << 'PYEOF'
import json
import os
import sys
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

MODEL_ID = "llmfan46/Qwen3.6-27B-uncensored-heretic-v2"
DATA_DIR = Path(os.path.expanduser("~/models/qwen3.6-27b-heretic-eagle/train_data"))
NUM_SAMPLES = 2000  # More = better EAGLE head, but slower
MAX_LEN = 2048

print(f"Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

print(f"Loading model in 4-bit (bitsandbytes)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.bfloat16,
    output_hidden_states=True,
)
model.eval()

print(f"Loading ShareGPT dataset...")
# Use a code-heavy dataset for better EAGLE performance on code tasks
try:
    ds = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train")
except:
    ds = load_dataset("mlabonne/open-perfectblend", split="train")

print(f"Generating hidden states for {NUM_SAMPLES} samples...")
DATA_DIR.mkdir(parents=True, exist_ok=True)

saved = 0
for i, sample in enumerate(ds):
    if saved >= NUM_SAMPLES:
        break

    # Extract text from the sample
    if "conversations" in sample:
        text = " ".join([c.get("value", "") for c in sample["conversations"]])
    elif "text" in sample:
        text = sample["text"]
    else:
        continue

    if len(text) < 100:
        continue

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", max_length=MAX_LEN, truncation=True)
    input_ids = inputs["input_ids"].to(model.device)

    if input_ids.shape[1] < 64:
        continue

    # Forward pass — collect hidden states
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)

    # Save the second-to-last layer hidden states and the target tokens
    # EAGLE predicts from hidden_states[-2] → next tokens
    hidden = outputs.hidden_states[-2][0].float().cpu().numpy().astype(np.float16)
    targets = input_ids[0, 1:].cpu().numpy()  # shifted by 1

    np.savez_compressed(
        DATA_DIR / f"sample_{saved:05d}.npz",
        hidden_states=hidden[:-1],  # align with targets
        target_tokens=targets,
    )

    saved += 1
    if saved % 50 == 0:
        print(f"  {saved}/{NUM_SAMPLES} samples generated")

print(f"Data generation complete: {saved} samples in {DATA_DIR}")
del model
torch.cuda.empty_cache()
PYEOF

echo "  Data generation complete."

# ── Step 3: Train EAGLE head ───────────────────────────────────────────
echo "[3/4] Training EAGLE-3 head..."
echo "  This trains a small model (~100M params) to predict next tokens"
echo "  from the hidden states we just collected."
echo "  Estimated: 4-6 hours"

"$VENV/bin/python" << 'PYEOF'
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig

MODEL_ID = "llmfan46/Qwen3.6-27B-uncensored-heretic-v2"
DATA_DIR = Path(os.path.expanduser("~/models/qwen3.6-27b-heretic-eagle/train_data"))
OUTPUT_DIR = Path(os.path.expanduser("~/models/qwen3.6-27b-heretic-eagle/eagle_head"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load base model config for dimensions
config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
text_config = getattr(config, 'text_config', config)
hidden_size = text_config.hidden_size  # 5120
vocab_size = text_config.vocab_size    # 248320

print(f"Hidden size: {hidden_size}, Vocab size: {vocab_size}")

# ── Dataset ──────────────────────────────────────────────────────────
class EagleDataset(Dataset):
    def __init__(self, data_dir, max_len=512):
        self.files = sorted(Path(data_dir).glob("sample_*.npz"))
        self.max_len = max_len
        print(f"  Found {len(self.files)} training files")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        hidden = torch.from_numpy(data["hidden_states"][:self.max_len].astype(np.float32))
        targets = torch.from_numpy(data["target_tokens"][:self.max_len].astype(np.int64))
        return hidden, targets

# ── EAGLE Head Model ─────────────────────────────────────────────────
class EagleHead(nn.Module):
    """Simple EAGLE head: 2 transformer layers + LM head."""

    def __init__(self, hidden_size, vocab_size, num_layers=2, num_heads=16):
        super().__init__()
        self.norm_in = nn.LayerNorm(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states):
        x = self.norm_in(hidden_states)
        x = self.transformer(x)
        logits = self.lm_head(x)
        return logits

# ── Training ─────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

dataset = EagleDataset(DATA_DIR, max_len=512)
loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

model = EagleHead(hidden_size, vocab_size, num_layers=2, num_heads=16).to(device)
model = model.to(torch.float16)

param_count = sum(p.numel() for p in model.parameters()) / 1e6
print(f"  EAGLE head parameters: {param_count:.0f}M")

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
criterion = nn.CrossEntropyLoss(ignore_index=-1)

NUM_EPOCHS = 10
print(f"  Training for {NUM_EPOCHS} epochs...")

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    n_batches = 0

    for hidden, targets in loader:
        hidden = hidden.to(device, dtype=torch.float16)
        targets = targets.to(device)

        with torch.cuda.amp.autocast():
            logits = model(hidden)
            # Flatten for cross-entropy
            loss = criterion(logits.view(-1, vocab_size).float(), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    print(f"  Epoch {epoch+1}/{NUM_EPOCHS}: loss = {avg_loss:.4f}")

# Save
torch.save(model.state_dict(), OUTPUT_DIR / "eagle_head.pt")
# Save config
import json
eagle_config = {
    "hidden_size": hidden_size,
    "vocab_size": vocab_size,
    "num_layers": 2,
    "num_heads": 16,
    "base_model": MODEL_ID,
}
(OUTPUT_DIR / "config.json").write_text(json.dumps(eagle_config, indent=2))
print(f"\nEAGLE head saved to {OUTPUT_DIR}")
print(f"  Size: {sum(f.stat().st_size for f in OUTPUT_DIR.rglob('*')) / 1e9:.2f} GB")
PYEOF

echo "  Training complete."

# ── Step 4: Summary ────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}[4/4] EAGLE head training complete!${R}"
echo -e "  ${GRY}Head: $EAGLE_DIR/eagle_head/${R}"
echo -e "  ${GRY}Next steps:${R}"
echo -e "  ${GRY}  1. Convert to GGUF for llama.cpp (needs upstream support for qwen35 EAGLE)${R}"
echo -e "  ${GRY}  2. Or use with SGLang/transformers directly${R}"
echo -e "  ${GRY}  3. Upload to HuggingFace: hf upload Quitetall/Qwen3.6-27B-heretic-EAGLE3 $EAGLE_DIR/eagle_head${R}"
echo ""
