#!/usr/bin/env bash
# Submit SADT experiments to SLURM — one job per experiment, each on a different GPU node.
# Usage:
#   bash experiments/submit_slurm.sh          # smoke test (LIMIT=1)
#   bash experiments/submit_slurm.sh full     # full run (all images)
#   bash experiments/submit_slurm.sh full 1   # Exp1 only
#   bash experiments/submit_slurm.sh full 2   # Exp2 only
#   bash experiments/submit_slurm.sh full 4   # Exp4 only

set -euo pipefail

LIMIT="${1:-1}"
EXP_FILTER="${2:-all}"   # "all", "1", "2", or "4"

if [ "$LIMIT" = "full" ]; then
    LIMIT_ARG=""
    JOB_SUFFIX="full"
    TIME="2:00:00"
else
    LIMIT_ARG="--limit $LIMIT"
    JOB_SUFFIX="smoke"
    TIME="0:20:00"
fi

# GPU nodes
GPU_NODES=(allemagne belgique autriche hollande hongrie groenland)
NODE_EXP1="ablette"           # RTX 3090, 24 GB — full precision
NODE_EXP2="anchois"           # RTX 3090, 24 GB — full precision
NODE_EXP4="anguille"          # RTX 3090, 24 GB — full precision

# Source node where code/model/data live (login node)
SRC_HOST=rouloul.polytechnique.fr
SRC_WORKDIR=/Data/guillaume.henrotte/SADT/SADT
SRC_MODELDIR=/Data/guillaume.henrotte/SADT/models
SRC_LOCALDIR=/Data/guillaume.henrotte/.local
SRC_NLTK=/Data/guillaume.henrotte/.nltk_data

WORKDIR=/Data/guillaume.henrotte/SADT/SADT

# ---------------------------------------------------------------------------
# Helper: emit the common job header (sync + env setup) as a heredoc fragment
# ---------------------------------------------------------------------------
submit_job() {
    local node="$1"
    local job_name="$2"
    local exp_cmd="$3"      # the python -m ... line(s) to run

    sbatch \
      --job-name="${job_name}_${JOB_SUFFIX}" \
      --nodelist="$node" \
      --cpus-per-task=4 \
      --mem=32G \
      --time="$TIME" \
      --output="/users/eleves-b/2024/guillaume.henrotte/slurm_%j.out" \
      --error="/users/eleves-b/2024/guillaume.henrotte/slurm_%j.err" \
      << SBATCH
#!/usr/bin/env bash
set -euo pipefail

RSYNC_SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no"

mkdir -p $WORKDIR /Data/guillaume.henrotte/SADT/models /Data/guillaume.henrotte/.local /Data/guillaume.henrotte/.nltk_data

echo "=== [$job_name] Syncing code from $SRC_HOST ==="
rsync -a --info=progress2 -e "\$RSYNC_SSH" $SRC_HOST:$SRC_WORKDIR/ $WORKDIR/

echo "=== [$job_name] Syncing Python packages ==="
rsync -a --info=progress2 -e "\$RSYNC_SSH" $SRC_HOST:$SRC_LOCALDIR/ /Data/guillaume.henrotte/.local/

echo "=== [$job_name] Syncing model ==="
rsync -a --info=progress2 --ignore-existing -e "\$RSYNC_SSH" $SRC_HOST:$SRC_MODELDIR/ /Data/guillaume.henrotte/SADT/models/

echo "=== [$job_name] Syncing NLTK data ==="
rsync -a --ignore-existing -e "\$RSYNC_SSH" $SRC_HOST:$SRC_NLTK/ /Data/guillaume.henrotte/.nltk_data/

export PYTHONUSERBASE=/Data/guillaume.henrotte/.local
export PYTHONPATH=$WORKDIR:/Data/guillaume.henrotte/.local/lib/python3.9/site-packages\${PYTHONPATH:+:\$PYTHONPATH}
export NLTK_DATA=/Data/guillaume.henrotte/.nltk_data
export CUDA_VISIBLE_DEVICES=0

cd $WORKDIR

MODEL_PATH=/Data/guillaume.henrotte/SADT/models/llava-v1.5-7b
IMAGE_DIR=data/test_imgs
LABELS_JSON=data/metadata/hallucination_results_val500.json
CHAIR_CACHE=data/chair.pkl

sync_back() {
  echo "=== [$job_name] Syncing outputs back to $SRC_HOST ==="
  rsync -a -e "\$RSYNC_SSH" $WORKDIR/outputs/ $SRC_HOST:$SRC_WORKDIR/outputs/
  echo "=== Sync done ==="
}

$exp_cmd

sync_back
echo "=== [$job_name] Done ==="
SBATCH
}

# ---------------------------------------------------------------------------
# Submit each experiment on its own node
# ---------------------------------------------------------------------------
if [ "$EXP_FILTER" = "all" ] || [ "$EXP_FILTER" = "1" ]; then
    echo "Submitting Exp1 on $NODE_EXP1 ..."
    submit_job "$NODE_EXP1" "sadt_exp1" \
        "python -m experiments.exp1_sink_corrected_topk --model-path \$MODEL_PATH --image-dir \$IMAGE_DIR --labels-json \$LABELS_JSON --chair-cache \$CHAIR_CACHE $LIMIT_ARG --output-dir outputs/exp1_fp16"
fi

if [ "$EXP_FILTER" = "all" ] || [ "$EXP_FILTER" = "2" ]; then
    echo "Submitting Exp2 on $NODE_EXP2 ..."
    submit_job "$NODE_EXP2" "sadt_exp2" \
        "python -m experiments.exp2_spectral_tdo --model-path \$MODEL_PATH --image-dir \$IMAGE_DIR --labels-json \$LABELS_JSON --chair-cache \$CHAIR_CACHE $LIMIT_ARG --output-dir outputs/exp2_fp16"
fi

if [ "$EXP_FILTER" = "all" ] || [ "$EXP_FILTER" = "4" ]; then
    echo "Submitting Exp4 on $NODE_EXP4 ..."
    submit_job "$NODE_EXP4" "sadt_exp4" \
        "python -m experiments.exp4_type_classification --model-path \$MODEL_PATH --image-dir \$IMAGE_DIR --labels-json \$LABELS_JSON --chair-cache \$CHAIR_CACHE $LIMIT_ARG --output-dir outputs/exp4_fp16"
fi

echo "All jobs submitted. Check status with: squeue -u guillaume.henrotte"
