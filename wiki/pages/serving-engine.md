# Why llama-cpp-python Wins on Single 4090

## Tested Engines

| Engine | Result | Why |
|--------|--------|-----|
| **llama-cpp-python** | Works, 262K ctx | Quantizes everything including embeddings |
| **vLLM** | OOM | 248K vocab embedding unquantized (2.4 GiB × 2) = 21 GiB model alone |
| **SGLang (GGUF)** | OOM | Dequantizes GGUF to FP16 before compute (~54 GiB) |
| **SGLang (HF)** | OOM | Same as vLLM — can't quantize embeddings |

## Why llama-cpp Wins

1. **GGUF quantizes everything** — embeddings, LM head, norms — all compressed. vLLM/SGLang keep embeddings in FP16.
2. **Dynamic KV allocation** — allocates as needed. vLLM pre-allocates at startup and refuses to start if budget doesn't meet utilization threshold.
3. **Quantized KV cache** — Q4_0/Q8_0 KV with flash attention. vLLM supports FP8 KV but the model weights already use all the VRAM.
4. **No Python overhead for inference** — C++ backend with Python wrapper. The native llama-server binary is even faster (no GIL).

## The 248K Vocabulary Problem

Qwen3.6-27B has 248,320 tokens in its vocabulary. The embedding table:
- `248320 × 5120 × 2 bytes (FP16) = 2.42 GiB`
- vLLM/SGLang keep this in FP16 (AWQ/GPTQ don't quantize embeddings)
- GGUF quantizes it along with everything else

This is the fundamental reason vLLM can't fit this model on 24GB with any useful KV cache.

## Native llama-server (even faster)

Build from source for ~2x over the Python wrapper:
```bash
cd ~/llama.cpp && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="89"
cmake --build build --config Release -j4 --target llama-server
```
With ngram-mod speculation: 40-137 t/s (vs 15-30 t/s Python wrapper).
