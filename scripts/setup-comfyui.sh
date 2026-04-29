#!/usr/bin/env bash
# scripts/setup-comfyui.sh — install ComfyUI + video nodes + FLUX model
set -euo pipefail

COMFY_DIR="$HOME/ComfyUI"
BOLD="\033[1m"; GRY="\033[90m"; GREEN="\033[32m"; R="\033[0m"

echo -e "\n${BOLD}Setting up ComfyUI${R}\n"

# Clone if needed
if [[ ! -d "$COMFY_DIR" ]]; then
  echo -e "  Cloning ComfyUI..."
  git clone https://github.com/comfyanonymous/ComfyUI "$COMFY_DIR"
fi

# Venv (separate from LLM stack to avoid dep conflicts)
cd "$COMFY_DIR"
if [[ ! -f ".venv/bin/activate" ]]; then
  echo -e "  Creating venv..."
  python3.12 -m venv .venv
fi

echo -e "  Installing requirements..."
.venv/bin/pip install -r requirements.txt -q

# Video nodes
echo -e "  Installing video custom nodes..."
mkdir -p custom_nodes
cd custom_nodes

if [[ ! -d "ComfyUI-WanVideoWrapper" ]]; then
  git clone https://github.com/kijai/ComfyUI-WanVideoWrapper
  cd ComfyUI-WanVideoWrapper && "$COMFY_DIR/.venv/bin/pip" install -r requirements.txt -q && cd ..
fi

if [[ ! -d "ComfyUI-LTXVideo" ]]; then
  git clone https://github.com/Lightricks/ComfyUI-LTXVideo
  cd ComfyUI-LTXVideo && "$COMFY_DIR/.venv/bin/pip" install -r requirements.txt -q 2>/dev/null && cd ..
fi

echo -e "\n${GREEN}ComfyUI ready.${R}"
echo -e "  ${GRY}Start:  just serve-comfyui${R}"
echo -e "  ${GRY}URL:    http://localhost:8188${R}"
echo -e "  ${GRY}Models: download FLUX/SDXL via the ComfyUI model manager${R}"
