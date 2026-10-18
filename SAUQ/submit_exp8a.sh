#!/usr/bin/env bash
# Submit exp8a (PR vs output length) to SLURM.
# Usage: bash SAUQ/submit_exp8a.sh <node> [model_name]

set -euo pipefail

NODE="${1:?Usage: $0 <node> [model_name]}"
MODEL_NAME="${2:-Llama-3.1-8B-Instruct}"

SRC_HOST=rouloul.polytechnique.fr
SRC_WORKDIR=/Data/guillaume.henrotte/SADT/SADT
SRC_MODELDIR=/Data/guillaume.henrotte/SADT/models
SRC_LOCALDIR=/Data/guillaume.henrotte/.local
SRC_NLTK=/Data/guillaume.henrotte/.nltk_data
SRC_HFCACHE=/Data/guillaume.henrotte/.cache/huggingface

WORKDIR=/Data/guillaume.henrotte/SADT/SADT
MODEL_PATH=/Data/guillaume.henrotte/SADT/models/$MODEL_NAME

echo "Submitting exp8a_pr_length on $NODE with model $MODEL_NAME …"

sbatch \
  --job-name="exp8a_pr_length" \
  --nodelist="$NODE" \
  --cpus-per-task=4 \
  --mem=48G \
  --time="1:55:00" \
  --output="/users/eleves-b/2024/guillaume.henrotte/slurm_%j.out" \
  --error="/users/eleves-b/2024/guillaume.henrotte/slurm_%j.err" \
  << SBATCH
#!/usr/bin/env bash
set -euo pipefail

RSYNC_SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no"

mkdir -p $WORKDIR $SRC_MODELDIR /Data/guillaume.henrotte/.local \
         /Data/guillaume.henrotte/.nltk_data /Data/guillaume.henrotte/.cache/huggingface

echo "=== [exp8a] Syncing code ==="
rsync -a --info=progress2 -e "\$RSYNC_SSH" $SRC_HOST:$SRC_WORKDIR/ $WORKDIR/

echo "=== [exp8a] Syncing Python packages ==="
rsync -a --info=progress2 -e "\$RSYNC_SSH" $SRC_HOST:$SRC_LOCALDIR/ /Data/guillaume.henrotte/.local/

echo "=== [exp8a] Syncing model ==="
rsync -a --ignore-existing --info=progress2 -e "\$RSYNC_SSH" \
    $SRC_HOST:$SRC_MODELDIR/ /Data/guillaume.henrotte/SADT/models/

echo "=== [exp8a] Syncing NLTK ==="
rsync -a --ignore-existing -e "\$RSYNC_SSH" \
    $SRC_HOST:$SRC_NLTK/ /Data/guillaume.henrotte/.nltk_data/

echo "=== [exp8a] Syncing HF cache ==="
rsync -a --ignore-existing -e "\$RSYNC_SSH" \
    $SRC_HOST:$SRC_HFCACHE/ /Data/guillaume.henrotte/.cache/huggingface/

export PYTHONUSERBASE=/Data/guillaume.henrotte/.local
export PYTHONPATH=$WORKDIR:/Data/guillaume.henrotte/.local/lib/python3.9/site-packages\${PYTHONPATH:+:\$PYTHONPATH}
export NLTK_DATA=/Data/guillaume.henrotte/.nltk_data
export HF_HOME=/Data/guillaume.henrotte/.cache/huggingface
export TRANSFORMERS_CACHE=/Data/guillaume.henrotte/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd $WORKDIR

sync_back() {
    echo "=== [exp8a] Syncing outputs back ==="
    rsync -a -e "\$RSYNC_SSH" $WORKDIR/outputs/ $SRC_HOST:$SRC_WORKDIR/outputs/ || true
}
trap sync_back EXIT

echo "=== [exp8a] Starting experiment ==="
python -m SAUQ.exp8a_pr_length \
    --model-path $MODEL_PATH \
    --max-new-tokens 100 \
    --limit 400 \
    --hf-cache /Data/guillaume.henrotte/.cache/huggingface \
    --output-dir outputs/exp8a_pr_length

echo "=== [exp8a] Done ==="
SBATCH

echo "Job submitted. Monitor with: squeue -u guillaume.henrotte"
