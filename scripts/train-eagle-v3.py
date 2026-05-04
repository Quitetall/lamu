#!/usr/bin/env python3
"""EAGLE v3: Hidden state prediction with shared LM head.

Architecture: predict h_{t+1} from (h_t, embed(token_t))
- Input: concat(hidden_state[5120], token_embed[5120]) = 10240
- Fuse: Linear(10240 → 5120)
- 2× residual MLP blocks (5120-dim, SiLU, 2x expansion)
- Output: predicted hidden_state_{t+1} [5120]

At inference: draft_token = argmax(LM_head @ predicted_h)
Multi-token: chain h_t → h_{t+1} → h_{t+2} → h_{t+3}

Training loss: MSE(predicted_h_{t+1}, actual_h_{t+1})
"""

import os, sys, json, torch, torch.nn as nn, numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

EAGLE_DIR = Path(os.path.expanduser("~/models/qwen3.6-27b-heretic-eagle"))
DATA_DIR = EAGLE_DIR / "train_data_v3"
OUTPUT_DIR = EAGLE_DIR / "eagle_head_v3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

H = 5120  # hidden size
FFN_MULT = 2  # FFN expansion factor (smaller = less memory)
N_LAYERS = 2
EPOCHS = 20
LR = 1e-4
BATCH_SIZE = 4
MAX_LEN = 128  # positions per sample


class ResidualMLP(nn.Module):
    def __init__(self, dim, mult=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.up = nn.Linear(dim, dim * mult)
        self.down = nn.Linear(dim * mult, dim)
        self.act = nn.SiLU()

    def forward(self, x):
        r = x
        x = self.norm(x)
        x = self.down(self.act(self.up(x)))
        return x + r


class EagleV3(nn.Module):
    """Predicts next hidden state from current hidden state + token embedding."""

    def __init__(self, hidden_size=5120, n_layers=2, ffn_mult=2):
        super().__init__()
        # Fuse hidden state + token embedding
        self.fuse = nn.Linear(hidden_size * 2, hidden_size)
        self.fuse_norm = nn.LayerNorm(hidden_size)
        # Residual MLP blocks
        self.layers = nn.ModuleList([
            ResidualMLP(hidden_size, ffn_mult) for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_state, token_emb):
        """
        hidden_state: [batch, seq, H] or [batch, H]
        token_emb: [batch, seq, H] or [batch, H]
        Returns: predicted next hidden state [same shape]
        """
        x = torch.cat([hidden_state, token_emb], dim=-1)
        x = self.fuse_norm(self.fuse(x))
        for layer in self.layers:
            x = layer(x)
        return self.out_norm(x)

    @torch.no_grad()
    def draft_multi(self, h_t, token_id, embed_weight, lm_head_weight, n_draft=4):
        """Generate n_draft tokens autoregressively."""
        device = next(self.parameters()).device
        h = h_t.to(device)
        drafts = []

        for _ in range(n_draft):
            tok_emb = embed_weight[token_id].to(device)
            h_next = self.forward(h.unsqueeze(0), tok_emb.unsqueeze(0)).squeeze(0)
            # Apply LM head to get token prediction
            logits = h_next @ lm_head_weight.T
            token_id = logits.argmax().item()
            drafts.append(token_id)
            h = h_next

        return drafts


class HiddenStateDataset(Dataset):
    def __init__(self, data_dir, max_len=MAX_LEN):
        self.files = sorted(Path(data_dir).glob("sample_*.npz"))
        self.max_len = max_len
        print(f"  {len(self.files)} training files")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        ml = self.max_len
        h = torch.from_numpy(d["hidden_states"][:ml].astype(np.float32))
        t = torch.from_numpy(d["target_tokens"][:ml].astype(np.int64))
        return h, t


def collate(batch):
    hs, ts = zip(*batch)
    max_len = max(h.shape[0] for h in hs)
    # Pad
    ph = torch.zeros(len(hs), max_len, H)
    pt = torch.zeros(len(ts), max_len, dtype=torch.long)
    mask = torch.zeros(len(hs), max_len, dtype=torch.bool)
    for i, (h, t) in enumerate(zip(hs, ts)):
        sl = h.shape[0]
        ph[i, :sl] = h
        pt[i, :sl] = t
        mask[i, :sl] = True
    return ph, pt, mask


def main():
    device = torch.device("cuda")
    print("EAGLE v3 Training: Hidden State Prediction")
    print(f"  H={H}, layers={N_LAYERS}, FFN mult={FFN_MULT}")

    # Load token embeddings (frozen, for input fusion)
    emb_path = EAGLE_DIR / "token_embeddings.pt"
    if not emb_path.exists():
        print(f"  ERROR: {emb_path} not found. Run data gen first.")
        sys.exit(1)

    print("  Loading token embeddings...", flush=True)
    embed_weight = torch.load(emb_path, map_location="cpu", weights_only=True).float()
    print(f"  Embeddings: {embed_weight.shape}")

    # Optional: load LM head for validation (draft accuracy measurement)
    lm_head_path = EAGLE_DIR / "lm_head.pt"
    lm_head = None
    if lm_head_path.exists():
        lm_head = torch.load(lm_head_path, map_location="cpu", weights_only=True).float()
        print(f"  LM head: {lm_head.shape} (for validation)")

    # Dataset
    dataset = HiddenStateDataset(DATA_DIR, max_len=MAX_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=2, pin_memory=True, collate_fn=collate)

    # Model
    model = EagleV3(H, N_LAYERS, FFN_MULT).to(device)
    trainable = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Trainable params: {trainable:.1f}M")
    print(f"  Model memory: {trainable * 4 / 1024:.1f} GB (FP32)")

    # Move embeddings to GPU for fast lookup
    embed_weight = embed_weight.to(device)
    print(f"  Embed on GPU: {embed_weight.element_size() * embed_weight.numel() / 1e9:.2f} GB")

    # Optimizer
    import bitsandbytes as bnb
    opt = bnb.optim.AdamW8bit(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler('cuda')

    best_loss = float('inf')
    for ep in range(EPOCHS):
        model.train()
        total_loss = 0
        total_cos = 0
        n_batches = 0

        for h_batch, t_batch, mask_batch in loader:
            h_batch = h_batch.to(device)      # [B, seq, H]
            t_batch = t_batch.to(device)      # [B, seq]
            mask_batch = mask_batch.to(device) # [B, seq]

            # Training pairs: predict h[t+1] from (h[t], embed(token[t]))
            # h[t] = h_batch[:, :-1]
            # target h[t+1] = h_batch[:, 1:]
            # token at t = t_batch[:, :-1] (token that was produced at position t)
            h_input = h_batch[:, :-1]         # [B, seq-1, H]
            h_target = h_batch[:, 1:]         # [B, seq-1, H]
            tok_input = t_batch[:, :-1]       # [B, seq-1]
            valid = mask_batch[:, 1:]         # [B, seq-1] (target positions must be valid)

            # Look up token embeddings
            tok_emb = embed_weight[tok_input]  # [B, seq-1, H]

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                h_pred = model(h_input, tok_emb)  # [B, seq-1, H]

                # MSE loss on valid positions only
                diff = (h_pred - h_target) ** 2
                diff = diff * valid.unsqueeze(-1)  # mask invalid
                loss = diff.sum() / (valid.sum() * H + 1e-8)

                # Cosine similarity for monitoring
                with torch.no_grad():
                    cos = nn.functional.cosine_similarity(
                        h_pred[valid], h_target[valid], dim=-1
                    ).mean()

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            total_loss += loss.item()
            total_cos += cos.item()
            n_batches += 1

            if n_batches % 200 == 0:
                print(f"    batch {n_batches}: loss={loss.item():.6f} cos={cos.item():.4f}", flush=True)

        sched.step()
        avg_loss = total_loss / max(n_batches, 1)
        avg_cos = total_cos / max(n_batches, 1)
        print(f"  Epoch {ep+1}/{EPOCHS}: loss={avg_loss:.6f} cos_sim={avg_cos:.4f}", flush=True)

        # Validation: measure draft accuracy with LM head
        if lm_head is not None and (ep + 1) % 5 == 0:
            model.eval()
            correct = 0
            total = 0
            lm_w = lm_head.to(device).half()
            with torch.no_grad():
                for h_batch, t_batch, mask_batch in loader:
                    h_batch = h_batch.to(device)
                    t_batch = t_batch.to(device)
                    mask_batch = mask_batch.to(device)

                    h_input = h_batch[:, :-1]
                    h_target = h_batch[:, 1:]
                    tok_input = t_batch[:, :-1]
                    tok_target = t_batch[:, 1:]  # actual next tokens
                    valid = mask_batch[:, 1:]

                    tok_emb = embed_weight[tok_input]
                    h_pred = model(h_input, tok_emb).half()

                    # Apply LM head to predicted hidden state
                    # [B, seq-1, H] @ [H, V] → [B, seq-1, V]
                    pred_logits = torch.matmul(h_pred[valid], lm_w.T)
                    pred_tokens = pred_logits.argmax(dim=-1)
                    actual_tokens = tok_target[valid]

                    correct += (pred_tokens == actual_tokens).sum().item()
                    total += actual_tokens.numel()

                    if total > 10000:
                        break

            acc = correct / max(total, 1)
            print(f"  >>> Draft accuracy (LM head): {acc:.1%} ({correct}/{total})", flush=True)
            lm_w = lm_w.cpu()
            del lm_w

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), OUTPUT_DIR / "eagle_v3_best.pt")
            print(f"    -> saved best (cos={avg_cos:.4f})", flush=True)

    # Save final
    torch.save(model.state_dict(), OUTPUT_DIR / "eagle_v3.pt")
    config = {
        "architecture": "eagle_v3_hidden_state_prediction",
        "hidden_size": H,
        "n_layers": N_LAYERS,
        "ffn_mult": FFN_MULT,
        "n_draft": 4,
        "training": {
            "epochs": EPOCHS,
            "best_loss": best_loss,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "max_len": MAX_LEN,
            "n_samples": len(dataset),
        },
        "shared_weights": ["lm_head.pt", "token_embeddings.pt"],
        "input": "concat(hidden_state, token_embedding)",
        "output": "predicted_next_hidden_state",
        "draft_method": "argmax(lm_head @ predicted_h)",
    }
    (OUTPUT_DIR / "config.json").write_text(json.dumps(config, indent=2))
    print(f"\nDone! Best loss: {best_loss:.6f}")
    print(f"  Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
