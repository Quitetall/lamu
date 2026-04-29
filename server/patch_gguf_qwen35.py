"""
Monkey-patch transformers to support qwen35 GGUF architecture.

Qwen3.5/3.6 uses the 'qwen35' arch label in GGUF files, but transformers
only maps 'qwen3'. This patch bridges the gap so SGLang can load qwen35
GGUFs directly without downloading HF safetensors.

Import this before starting SGLang:
    import server.patch_gguf_qwen35  # noqa: F401
"""

# ── Step 1: Register qwen35 model_type → Qwen3_5Config ─────────────────

from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

# Directly inject into the live CONFIG_MAPPING so AutoConfig resolves "qwen35"
# to Qwen3_5Config without trying to import a nonexistent transformers.models.qwen35 module.
CONFIG_MAPPING._extra_content["qwen35"] = Qwen3_5Config

# ── Step 2: Register qwen35 in GGUF mappings ───────────────────────────

import transformers.modeling_gguf_pytorch_utils as gguf_utils

_QWEN35_CONFIG_MAPPING = {
    "context_length": "max_position_embeddings",
    "block_count": "num_hidden_layers",
    "feed_forward_length": "intermediate_size",
    "embedding_length": "hidden_size",
    "rope.dimension_count": None,
    "rope.freq_base": "rope_theta",
    "attention.head_count": "num_attention_heads",
    "attention.head_count_kv": "num_key_value_heads",
    "attention.layer_norm_rms_epsilon": "rms_norm_eps",
    "vocab_size": "vocab_size",
}

if "qwen35" not in gguf_utils.GGUF_CONFIG_MAPPING:
    gguf_utils.GGUF_CONFIG_MAPPING["qwen35"] = _QWEN35_CONFIG_MAPPING

if "qwen35" not in gguf_utils.GGUF_SUPPORTED_ARCHITECTURES:
    gguf_utils.GGUF_SUPPORTED_ARCHITECTURES.append("qwen35")

if hasattr(gguf_utils, "GGUF_TOKENIZER_MAPPING") and "qwen35" not in gguf_utils.GGUF_TOKENIZER_MAPPING:
    ref = gguf_utils.GGUF_TOKENIZER_MAPPING.get("qwen3", gguf_utils.GGUF_TOKENIZER_MAPPING.get("qwen2", {}))
    gguf_utils.GGUF_TOKENIZER_MAPPING["qwen35"] = ref

if hasattr(gguf_utils, "GGUF_TO_TRANSFORMERS_ARCHITECTURE_MAPPING"):
    if "qwen35" not in gguf_utils.GGUF_TO_TRANSFORMERS_ARCHITECTURE_MAPPING:
        gguf_utils.GGUF_TO_TRANSFORMERS_ARCHITECTURE_MAPPING["qwen35"] = "qwen3_5"

if hasattr(gguf_utils, "GGUF_TENSOR_NAME_MAPPING"):
    if "qwen35" not in gguf_utils.GGUF_TENSOR_NAME_MAPPING:
        ref = gguf_utils.GGUF_TENSOR_NAME_MAPPING.get("qwen3", gguf_utils.GGUF_TENSOR_NAME_MAPPING.get("llama", {}))
        gguf_utils.GGUF_TENSOR_NAME_MAPPING["qwen35"] = ref

# ── Step 3: Register qwen35 tokenizer converter ────────────────────────

try:
    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS
    if "qwen35" not in GGUF_TO_FAST_CONVERTERS and "qwen3" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["qwen35"] = GGUF_TO_FAST_CONVERTERS["qwen3"]
except Exception:
    pass

# ── Step 4: Register qwen35 in SGLang's config + model registries ──────

try:
    from sglang.srt.utils.hf_transformers.config import _CONFIG_REGISTRY
    if "qwen35" not in _CONFIG_REGISTRY and "qwen3_5" in _CONFIG_REGISTRY:
        _CONFIG_REGISTRY["qwen35"] = _CONFIG_REGISTRY["qwen3_5"]
except Exception:
    pass

try:
    from sglang.srt.models.registry import ModelRegistry
    if hasattr(ModelRegistry, "_models") and "qwen3_5" in ModelRegistry._models:
        if "qwen35" not in ModelRegistry._models:
            ModelRegistry._models["qwen35"] = ModelRegistry._models["qwen3_5"]
except Exception:
    pass
