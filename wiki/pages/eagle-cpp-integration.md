# EAGLE C++ Integration Spec for llama.cpp

## Status: Roadmap — ready to implement

## What Exists
- llama.cpp has `COMMON_SPECULATIVE_TYPE_EAGLE3` enum and stub (PR-18039 TODO)
- The `common_speculative_state_eagle3` struct exists with empty `draft()` method
- Mixing spec types (ngram-mod + EAGLE) is already supported by the dispatch system
- Our trained EAGLE head (v1 bottleneck, 285M params, 9.5% acceptance)

## Architecture

### Our EAGLE head (PyTorch)
```
Input: hidden_states[-1] from main model (5120 dim)
→ Linear(5120, 1024)       "down projection"
→ LayerNorm(1024)
→ TransformerEncoder(2 layers, 8 heads, FFN=4096)
→ Linear(1024, 248320)     "LM head" (vocab prediction)
→ argmax → draft token
```

### What needs to happen in C++
1. **Load weights** from a custom binary/GGUF file
2. **Extract hidden states** after each main model eval (llama_get_embeddings_ith)
3. **Run EAGLE forward pass** using ggml compute graph
4. **Return draft tokens** via the common_speculative_state interface

### ggml Implementation

The forward pass maps to these ggml operations:
```c
// down projection
cur = ggml_mul_mat(ctx, eagle_down_weight, hidden_state);  // [1, 5120] × [5120, 1024]
cur = ggml_add(ctx, cur, eagle_down_bias);

// layer norm
cur = ggml_norm(ctx, cur, eps);
cur = ggml_mul(ctx, cur, norm_weight);
cur = ggml_add(ctx, cur, norm_bias);

// 2× transformer encoder layers (self-attention + FFN)
for each layer:
    // pre-norm
    // self-attention: Q,K,V projections, scaled dot product, output projection
    // residual add
    // pre-norm
    // FFN: linear1 → GELU → linear2
    // residual add

// LM head
logits = ggml_mul_mat(ctx, eagle_lm_weight, cur);  // [1, 1024] × [1024, 248320]
```

### Weight Format

Option A: Custom binary file
```
Header: magic, n_layers, hidden_size, inner_dim, vocab_size
Tensors: down.weight, down.bias, norm.weight, norm.bias,
         layer0.attn.in_proj.{weight,bias}, layer0.attn.out_proj.{weight,bias},
         layer0.ffn.linear1.{weight,bias}, layer0.ffn.linear2.{weight,bias},
         layer0.norm1.{weight,bias}, layer0.norm2.{weight,bias},
         (same for layer1),
         lm.weight
```

Option B: GGUF (preferred — llama.cpp already has GGUF loading)
- Use convert_hf_to_gguf.py with a custom model class
- Register "eagle_bottleneck" as an architecture

### Integration Points in llama.cpp

1. `common/speculative.cpp` — implement `common_speculative_state_eagle3::draft()`
2. `common/speculative.h` — add EAGLE weight loading structures
3. `common/arg.cpp` — add `--spec-eagle-model` CLI flag
4. `tools/server/server-context.cpp` — enable embedding extraction when EAGLE is active

### Stacking with ngram-mod

The dispatch in `common_speculative::draft()` already iterates through `impls` vector.
If we add EAGLE3 alongside ngram-mod:
```cpp
configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, params));
configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_EAGLE3, params));
```
The system will try ngram-mod first (higher precedence for draftless), fall back to EAGLE.

### Files to Modify
- `common/speculative.cpp` — EAGLE3 implementation (~200 lines)
- `common/speculative.h` — EAGLE weight structures
- `common/CMakeLists.txt` — add eagle source files
- `common/arg.cpp` — CLI flag
- `tools/server/server-context.cpp` — enable embeddings for EAGLE

### Estimated Effort
- Weight conversion script (Python → GGUF): 2-3 hours
- ggml forward pass implementation: 4-6 hours
- Integration + testing: 2-3 hours
- Total: ~1-2 days focused work

### Community Impact
- First EAGLE-3 implementation in llama.cpp (the stub exists, nobody has filled it)
- First EAGLE head for Qwen3.6-27B Uncensored
- Stacking with ngram-mod (unique — nobody has done this)
- Potential upstream PR to ggml-org/llama.cpp
