"""
SGLang launcher with qwen35 GGUF patch.
Applies the monkey-patch then starts SGLang server.
"""
import multiprocessing
import sys
import os

# Required for SGLang's multiprocessing
multiprocessing.set_start_method("spawn", force=True)

# Add repo root to path
sys.path.insert(0, os.path.expanduser("~/local-llm"))

# Apply the patch BEFORE sglang imports transformers GGUF utils
import server.patch_gguf_qwen35  # noqa: F401

if __name__ == "__main__":
    from sglang.launch_server import prepare_server_args, run_server
    args = prepare_server_args(sys.argv[1:])
    run_server(args)
