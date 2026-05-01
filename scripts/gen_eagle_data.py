"""Generate EAGLE training data from the model's own code completions.

No external datasets needed. Feeds code prompts to the model, captures
hidden states during generation. Training data matches inference exactly.
"""
import os, sys, torch, numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "llmfan46/Qwen3.6-27B-uncensored-heretic-v2"
DATA_DIR = Path(os.path.expanduser("~/models/qwen3.6-27b-heretic-eagle/train_data_v3"))
NUM_SAMPLES = 10000
MAX_LEN = 512

# Code prompts — the model generates completions, we capture hidden states
PROMPTS = [
    "def is_prime(n):", "class LinkedList:", "def binary_search(arr, target):",
    "def quicksort(arr):", "class Stack:", "def fibonacci(n):",
    "def merge_sort(arr):", "class LRUCache:", "def bfs(graph, start):",
    "def dfs(graph, start):", "class TreeNode:", "def parse_json(s):",
    "async def fetch_data(url):", "class HTTPServer:", "def tokenize(text):",
    "def matrix_multiply(a, b):", "class Database:", "def compress(data):",
    "def decrypt(ciphertext, key):", "class NeuralNetwork:", "def gradient_descent(f, x0):",
    "class WebSocket:", "def render_template(template, context):", "def validate_email(s):",
    "import torch\ndef train(model, data):", "from fastapi import FastAPI\napp = FastAPI()\n@app.get('/')\ndef ",
    "import numpy as np\ndef fft(signal):", "class Tokenizer:", "def attention(q, k, v):",
    "def convolution(image, kernel):", "class Optimizer:", "def backpropagate(loss):",
    "def read_csv(path):", "class DataFrame:", "def plot(data):",
    "def authenticate(token):", "class Router:", "def handle_request(req):",
    "def serialize(obj):", "class Cache:", "def memoize(func):",
    "# Implement a red-black tree\nclass RBTree:", "# HTTP client with retries\nasync def fetch(",
    "# Parse command line args\ndef parse_args():", "# Database connection pool\nclass Pool:",
    "# Websocket chat server\nclass ChatServer:", "# File watcher\ndef watch(path):",
    "# Rate limiter\nclass RateLimiter:", "# JSON schema validator\ndef validate(schema, data):",
    "# Git diff parser\ndef parse_diff(text):", "# Markdown to HTML\ndef md_to_html(text):",
]

DATA_DIR.mkdir(parents=True, exist_ok=True)
existing = len(list(DATA_DIR.glob("sample_*.npz")))
if existing >= NUM_SAMPLES:
    print(f"Already have {existing} samples"); sys.exit(0)

print(f"Generating {NUM_SAMPLES} samples from model completions", flush=True)
print(f"Resuming from {existing}", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"),
    device_map="auto", trust_remote_code=True, dtype=torch.bfloat16, output_hidden_states=True,
)
model.eval()

saved = existing
prompt_idx = 0

while saved < NUM_SAMPLES:
    prompt = PROMPTS[prompt_idx % len(PROMPTS)]
    prompt_idx += 1

    # Vary the prompt to get diverse completions
    if saved % 3 == 0:
        text = f"<|im_start|>user\nComplete this Python code:\n```python\n{prompt}\n```<|im_end|>\n<|im_start|>assistant\n"
    elif saved % 3 == 1:
        text = f"<|im_start|>user\nWrite production-quality Python:\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    else:
        text = f"<|im_start|>system\nYou are an expert Python programmer.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(text, return_tensors="pt", max_length=MAX_LEN, truncation=True)
    input_ids = inputs["input_ids"].to(model.device)

    if input_ids.shape[1] < 32:
        continue

    try:
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
        hidden = outputs.hidden_states[-1][0].float().cpu().numpy().astype(np.float16)
        targets = input_ids[0, 1:].cpu().numpy()

        np.savez_compressed(
            DATA_DIR / f"sample_{saved:05d}.npz",
            hidden_states=hidden[:-1],
            target_tokens=targets,
        )
        saved += 1
        if saved % 500 == 0:
            print(f"  {saved}/{NUM_SAMPLES}", flush=True)
        del outputs, hidden
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        continue

print(f"Done: {saved} samples in {DATA_DIR}", flush=True)
del model; torch.cuda.empty_cache()
