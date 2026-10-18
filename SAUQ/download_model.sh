#!/usr/bin/env bash
# Download a HuggingFace model into /Data/guillaume.henrotte/SADT/models/
#
# Usage:
#   bash SAUQ/download_model.sh <hf_token> [model_id]
#
# Default model_id: meta-llama/Llama-3.1-8B-Instruct
# Example:
#   bash SAUQ/download_model.sh hf_xxxx
#   bash SAUQ/download_model.sh hf_xxxx mistralai/Mistral-7B-Instruct-v0.2

set -euo pipefail

HF_TOKEN="${1:?Usage: $0 <hf_token> [model_id]}"
MODEL_ID="${2:-meta-llama/Llama-3.1-8B-Instruct}"
MODEL_NAME=$(basename "$MODEL_ID")
DEST="/Data/guillaume.henrotte/SADT/models/$MODEL_NAME"

echo "=== Downloading $MODEL_ID → $DEST ==="
mkdir -p "$DEST"

export HF_HOME=/Data/guillaume.henrotte/.cache/huggingface
export PYTHONPATH=/Data/guillaume.henrotte/.local/lib/python3.9/site-packages

python3 - <<PYEOF
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id="$MODEL_ID",
    local_dir="$DEST",
    token="$HF_TOKEN",
    ignore_patterns=["*.bin"],   # prefer safetensors
)
print("Download complete: $DEST")
PYEOF
