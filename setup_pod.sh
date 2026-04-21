#!/usr/bin/env bash
# Run once on a fresh RunPod container to install dependencies and verify GPU.
set -e

# Install uv if missing
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
fi

# Remove any stale torch builds to avoid filling the disk
uv pip uninstall torch torchvision torchaudio 2>/dev/null || true
uv cache clean 2>/dev/null || true

# Install all dependencies (picks up cu128 torch from pyproject.toml)
uv sync

# Verify
uv run python -c "
import torch
print(f'torch:  {torch.__version__}')
print(f'cuda:   {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'gpu:    {torch.cuda.get_device_name(0)}')
"
