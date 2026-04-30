# ngram-mod Speculative Decoding

## What It Does
Hash-based pattern matching from conversation history. No draft model needed. ~16 MB memory overhead. Gets faster as the conversation grows — code generation hits 136 t/s because code has repetitive patterns.

## Speed Progression
```
1st response:  ~40 t/s  (cold hash table)
2nd response:  ~41 t/s  (20% acceptance)
3rd+ response: 60-137 t/s (patterns accumulate)
```

## Flags (latest llama.cpp)
```
--spec-type ngram-mod
--spec-ngram-mod-n-match 24    # 24-token lookup window
--spec-ngram-mod-n-min 12      # minimum draft length
--spec-ngram-mod-n-max 48      # maximum draft length
```

Important: `--draft-min` must be > 0 (default 0 causes "low acceptance streak" resets that clear the hash pool).

## Best Use Cases
- Code refactoring (rewriting existing code — highly repetitive)
- Reasoning models (think blocks repeat patterns in the answer)
- Iterative development (similar prompts build up patterns)
- Summarization

## Limitations
- First response is normal speed (no patterns yet)
- Unrelated conversations don't benefit from each other
- Hash pool resets on server restart (no persistence yet)

## Memory Bandwidth Ceiling
RTX 4090: 1 TB/s. 27B Q4_K_M = 16 GB per token read.
Theoretical max: 16 GB / 1008 GB/s = 15.9ms/token = 63 t/s.
ngram-mod bypasses this by verifying multiple tokens per forward pass.

## Reference
- ggml-org/llama.cpp#19164
- r/LocalLLaMA appreciation post (136 t/s on Qwen3.6-27B)
