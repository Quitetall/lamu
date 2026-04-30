# EAGLE-3 Head Training

## Status: In Progress (overnight pipeline running)

## What EAGLE Does
Trains a small neural network (~100M params) on top of the base model's hidden states to predict future tokens. At inference: EAGLE head proposes 3-4 tokens, base model verifies in one forward pass. If acceptance rate is 70%+, effective throughput doubles.

## Expected Result
Combined with ngram-mod:
- ngram-mod alone: 40-137 t/s
- EAGLE head alone: ~100-190 t/s
- Both together: potentially 150-200+ t/s for code tasks

## Training Pipeline (scripts/train-eagle-head.sh)
1. Download BF16 heretic model from HuggingFace (~54 GB)
2. Load in 4-bit (bitsandbytes) on 4090 — fits in 24GB
3. Run 2000 samples through model, save hidden states from 2nd-to-last layer
4. Train 2-layer transformer EAGLE head to predict next tokens from hidden states
5. Save to ~/models/qwen3.6-27b-heretic-eagle/

Estimated time: 16-24 hours total (unattended).

## Why This Would Be Novel
Nobody has published an EAGLE-3 head for Qwen3.6-27B Uncensored.
The closest: jiapingW/Qwen3.5-35B-A3B-Eagle3-Specforge (different model, MoE).

## Using the EAGLE Head (after training)
```bash
llama-server -m model.gguf --spec-draft-model eagle-head.gguf [other flags]
```
Note: EAGLE head needs to be converted to GGUF format first. llama.cpp EAGLE support for qwen35 may need upstream patches.
