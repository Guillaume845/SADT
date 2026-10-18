#!/usr/bin/env bash
#SBATCH --job-name=exp5b_samsum
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1:55:00
#SBATCH --output=/users/eleves-b/2024/guillaume.henrotte/slurm_%j.out
#SBATCH --error=/users/eleves-b/2024/guillaume.henrotte/slurm_%j.err

set -euo pipefail

SRC_HOST=rouloul.polytechnique.fr
WORKDIR=/Data/guillaume.henrotte/SADT/SADT
MODEL_PATH=/Data/guillaume.henrotte/SADT/models/llava-v1.5-7b
RSYNC_SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no"

mkdir -p "$WORKDIR" \
         /Data/guillaume.henrotte/SADT/models \
         /Data/guillaume.henrotte/.local \
         /Data/guillaume.henrotte/.nltk_data \
         /Data/guillaume.henrotte/.cache/huggingface

echo "=== [exp5b_samsum] Syncing code ==="
rsync -a --info=progress2 -e "$RSYNC_SSH" \
    "$SRC_HOST:/Data/guillaume.henrotte/SADT/SADT/" "$WORKDIR/"

echo "=== [exp5b_samsum] Syncing Python packages ==="
rsync -a --info=progress2 -e "$RSYNC_SSH" \
    "$SRC_HOST:/Data/guillaume.henrotte/.local/" /Data/guillaume.henrotte/.local/

echo "=== [exp5b_samsum] Syncing model (skip existing) ==="
rsync -a --ignore-existing --info=progress2 -e "$RSYNC_SSH" \
    "$SRC_HOST:/Data/guillaume.henrotte/SADT/models/" /Data/guillaume.henrotte/SADT/models/

echo "=== [exp5b_samsum] Syncing NLTK ==="
rsync -a --ignore-existing -e "$RSYNC_SSH" \
    "$SRC_HOST:/Data/guillaume.henrotte/.nltk_data/" /Data/guillaume.henrotte/.nltk_data/

echo "=== [exp5b_samsum] Syncing HF cache (SamSum dataset) ==="
rsync -a --ignore-existing -e "$RSYNC_SSH" \
    "$SRC_HOST:/Data/guillaume.henrotte/.cache/huggingface/" \
    /Data/guillaume.henrotte/.cache/huggingface/

export PYTHONUSERBASE=/Data/guillaume.henrotte/.local
export PYTHONPATH="$WORKDIR:/Data/guillaume.henrotte/.local/lib/python3.9/site-packages"
export NLTK_DATA=/Data/guillaume.henrotte/.nltk_data
export HF_HOME=/Data/guillaume.henrotte/.cache/huggingface
export TRANSFORMERS_CACHE=/Data/guillaume.henrotte/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0

cd "$WORKDIR"

sync_back() {
    echo "=== [exp5b_samsum] Syncing outputs back ==="
    rsync -a -e "$RSYNC_SSH" \
        "$WORKDIR/outputs/" "$SRC_HOST:/Data/guillaume.henrotte/SADT/SADT/outputs/"
}
trap sync_back EXIT

echo "=== [exp5b_samsum] Starting experiment ==="
python3 -m SAUQ.exp5b_samsum \
    --model-path "$MODEL_PATH" \
    --max-new-tokens 100 \
    --limit 400 \
    --hf-cache /Data/guillaume.henrotte/.cache/huggingface \
    --output-dir outputs/exp5b_samsum

echo "=== [exp5b_samsum] Done ==="
