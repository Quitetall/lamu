# RTX 4090 VRAM Budget

## Total: 24,564 MiB (23.51 GiB usable)

## Baseline Display Overhead
KDE + Firefox + Ghostty: ~1.2-1.8 GiB
Vesktop (Discord): +436 MiB (kill when not needed)
Sunshine (remote desktop): +394 MiB (kill when not needed)
Available after display: ~21.5-22.3 GiB

## Model Configurations

### Best Quality (default)
```
Q5_K_S model:         ~18 GiB
Q8_0 KV (108K ctx):    ~3.5 GiB
Compute buffers:        ~1 GiB
Total:                 ~22.5 GiB — tight fit, kill Discord/Sunshine
```

### Max Context
```
Q4_K_M model:          ~16 GiB
Q4_0 KV (262K ctx):     ~4 GiB
Compute buffers:        ~1.4 GiB
Total:                 ~21.4 GiB — fits with display server
```

### ComfyUI (image/video)
```
FLUX.1 Dev FP8:        ~17 GiB
SDXL:                   ~8 GiB
Can't run alongside Qwen — must swap (just swap-comfyui)
```

## Cannot Fit on 24GB
- vLLM + any 27B model (embeddings unquantized = 21+ GiB model alone)
- MoE 35B-A3B GGUF (20 GiB model weights)
- Two models simultaneously
