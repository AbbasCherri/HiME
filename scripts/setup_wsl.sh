#!/usr/bin/env bash
# One-time setup on the WSL (Ubuntu) execution machine. Safe to re-run.
# Usage: bash setup_wsl.sh [install_dir]     (default: ~/HiME)
set -euo pipefail

REPO_URL="${HIME_REPO_URL:-https://github.com/AbbasCherri/HiME.git}"
DIR="${1:-$HOME/HiME}"

echo "==> System packages"
sudo apt-get update -y
sudo apt-get install -y git curl ca-certificates build-essential

echo "==> uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

echo "==> Repository at $DIR"
if [ -d "$DIR/.git" ]; then
  git -C "$DIR" pull --ff-only
else
  git clone "$REPO_URL" "$DIR"
fi
cd "$DIR"

echo "==> Python 3.12 and locked dependencies"
uv python install 3.12
uv sync --python 3.12 --extra embed
uv run python -c "import numpy, scipy, hypothesis, hime; print('numpy', numpy.__version__, 'scipy', scipy.__version__)"
uv run python -c "import torch; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

echo "==> Dataset (read-only; never modified)"
DATA="${HIME_DATA:-$HOME/HiME-dataset}"
if [ -f "$DATA/derived/buildings.parquet" ]; then
  echo "found $DATA"; du -sh "$DATA" 2>/dev/null || true
else
  echo "WARNING: $DATA/derived/buildings.parquet missing (upload still running?)"
fi

echo "==> Hugging Face access (DINOv3 weights are gated)"
if [ -z "${HF_TOKEN:-}" ] && ! uv run hf auth whoami >/dev/null 2>&1; then
  echo "Not logged in. Accept the DINOv3 licence on huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m"
  echo "and facebook/dinov3-vitb16-pretrain-lvd1689m, then run:  uv run hf auth login"
fi

echo "==> Output directory on disk (not /tmp)"
mkdir -p runs

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
else
  echo "No GPU visible; only Phase 9 embeddings need it."
fi

echo "Done. Next: bash scripts/run_tests.sh   (set HIME_DATA=/path/to/release for real-data tests)"
