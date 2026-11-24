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
uv sync --python 3.12
uv run python -c "import numpy, scipy, hypothesis, hime; print('numpy', numpy.__version__, 'scipy', scipy.__version__)"

echo "==> Output directory on disk (not /tmp)"
mkdir -p runs

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
else
  echo "No GPU visible; only Phase 9 embeddings need it."
fi

echo "Done. Next: bash scripts/run_tests.sh   (set HIME_DATA=/path/to/release for real-data tests)"
