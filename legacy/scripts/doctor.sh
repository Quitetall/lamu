#!/usr/bin/env bash
# scripts/doctor.sh — diagnose all problems with the LAMU stack
set -euo pipefail

GRY="\033[90m"; GREEN="\033[32m"; YEL="\033[33m"; RED="\033[31m"; R="\033[0m"
PASS="${GREEN}✓${R}"
FAIL="${RED}✗${R}"
WARN="${YEL}!${R}"

echo -e "\n${GRY}LAMU Doctor — checking everything...${R}\n"

errors=0

# ── 1. GPU ──
echo -e "  ${GRY}[GPU]${R}"
if nvidia-smi &>/dev/null; then
  VRAM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  VRAM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')
  VRAM_FREE=$((VRAM_TOTAL - VRAM_USED))
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
  echo -e "    $PASS GPU: $GPU_NAME ($VRAM_USED/$VRAM_TOTAL MiB used, $VRAM_FREE MiB free)"
  if [ "$VRAM_FREE" -lt 2000 ]; then
    echo -e "    $WARN Low VRAM ($VRAM_FREE MiB free). Kill unused GPU processes."
    errors=$((errors+1))
  fi
else
  echo -e "    $FAIL nvidia-smi not found — no GPU detected"
  errors=$((errors+1))
fi

# ── 2. Models on disk ──
echo -e "  ${GRY}[Models]${R}"
HERETIC=$(ls ~/models/qwen3.6-27b-heretic/*Q4_K_M*.gguf 2>/dev/null | head -1)
if [ -n "$HERETIC" ]; then
  echo -e "    $PASS Qwen3.6-27B heretic: $(basename $HERETIC)"
else
  echo -e "    $FAIL Qwen3.6-27B not found. Run: just setup-qwen36"
  errors=$((errors+1))
fi

if [ -f ~/models/gpt2-xl-gguf/gpt2-xl.Q4_K_M.gguf ]; then
  echo -e "    $PASS GPT-2 XL: gpt2-xl.Q4_K_M.gguf"
else
  echo -e "    $WARN GPT-2 XL not found. Run: hf download RichardErkhov/openai-community_-_gpt2-xl-gguf gpt2-xl.Q4_K_M.gguf --local-dir ~/models/gpt2-xl-gguf"
fi

if [ -f ~/models/qwen3.6-dflash-gguf/dflash-3.6-q4km.gguf ]; then
  echo -e "    $PASS DFlash draft: dflash-3.6-q4km.gguf"
else
  echo -e "    $WARN DFlash draft GGUF not found (needed for 82 t/s mode)"
fi

if [ -d ~/models/qwen3.6-27b-dflash-draft ]; then
  echo -e "    $PASS DFlash draft (safetensors): z-lab/Qwen3.6-27B-DFlash"
else
  echo -e "    $WARN DFlash safetensors draft not found (needed for 106 t/s lucebox mode)"
fi

# ── 3. Binaries ──
echo -e "  ${GRY}[Binaries]${R}"
LLAMA=~/llama.cpp/build/bin/llama-server
if [ -f "$LLAMA" ]; then
  echo -e "    $PASS llama-server: $(stat -c '%y' $LLAMA | cut -d. -f1)"
else
  echo -e "    $FAIL llama-server not built. cd ~/llama.cpp/build && cmake --build . --target llama-server -j\$(nproc)"
  errors=$((errors+1))
fi

MEGA_SO=$(ls ~/local-llm/lucebox-hub/megakernel/*megakernel*.so 2>/dev/null | head -1)
if [ -n "$MEGA_SO" ]; then
  echo -e "    $PASS megakernel .so: $(basename $MEGA_SO)"
else
  echo -e "    $FAIL megakernel not compiled. cd lucebox-hub/megakernel && CXX=g++-14 python setup.py build_ext --inplace"
  errors=$((errors+1))
fi

if command -v g++-14 &>/dev/null; then
  echo -e "    $PASS gcc-14: $(g++-14 --version | head -1)"
else
  echo -e "    $FAIL gcc-14 not installed (required for CUDA builds). sudo pacman -S gcc14"
  errors=$((errors+1))
fi

# ── 4. Services ──
echo -e "  ${GRY}[Services]${R}"
if curl -sf http://localhost:8020/health &>/dev/null; then
  MODEL=$(curl -s http://localhost:8020/v1/models 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "unknown")
  echo -e "    $PASS :8020 — $MODEL"
else
  echo -e "    $WARN :8020 — not running (start: just start)"
fi

if curl -sf http://localhost:8001/health &>/dev/null; then
  echo -e "    $PASS :8001 — megakernel 0.8B"
else
  echo -e "    $WARN :8001 — megakernel not running"
fi

if curl -sf http://localhost:8000/v1/models &>/dev/null; then
  echo -e "    $PASS :8000 — DFlash"
else
  echo -e "    ${GRY}    :8000 — DFlash not running (optional)${R}"
fi

# ── 5. Python env ──
echo -e "  ${GRY}[Python]${R}"
VENV=~/local-llm/.venv/bin/python
if [ -f "$VENV" ]; then
  PY_VER=$($VENV --version 2>&1)
  echo -e "    $PASS venv: $PY_VER"
else
  echo -e "    $FAIL .venv not found. python3.12 -m venv .venv"
  errors=$((errors+1))
fi

# ── 6. MCP ──
echo -e "  ${GRY}[MCP]${R}"
if [ -f ~/local-llm/server/mcp_qwen.py ]; then
  echo -e "    $PASS mcp_qwen.py exists"
else
  echo -e "    $FAIL MCP server missing"
  errors=$((errors+1))
fi

if grep -q "local-llm" ~/.claude.json 2>/dev/null || grep -q "local-llm" ~/.claude/settings.json 2>/dev/null; then
  echo -e "    $PASS MCP configured in Claude settings"
else
  echo -e "    $WARN MCP not found in ~/.claude.json (Claude Code won't see local models)"
fi

# ── 7. Disk ──
echo -e "  ${GRY}[Disk]${R}"
MODELS_SIZE=$(du -sh ~/models/ 2>/dev/null | cut -f1)
echo -e "    ${GRY}~/models/: $MODELS_SIZE${R}"
FREE_DISK=$(df -h ~ | awk 'NR==2{print $4}')
echo -e "    ${GRY}Free disk: $FREE_DISK${R}"

# ── Summary ──
echo ""
if [ "$errors" -eq 0 ]; then
  echo -e "  ${GREEN}All checks passed.${R} Run ${GRY}just start${R} to boot the stack."
else
  echo -e "  ${RED}$errors issue(s) found.${R} Fix the ✗ items above."
fi
echo ""
