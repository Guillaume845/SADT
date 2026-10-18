#!/usr/bin/env bash
# Submit SAUQ experiments to SLURM.
#
# Usage:
#   bash SAUQ/submit_sauq.sh exp5         # Exp 5 — locality vs information selection
#   bash SAUQ/submit_sauq.sh rebuild1     # Rebuild 1 — Figure 4 replication
#   bash SAUQ/submit_sauq.sh all          # Both experiments (on separate nodes)
#
# Outputs land in outputs/exp5_locality/ and outputs/rebuild_1_copy_task/
# on the login node after the job finishes.

set -euo pipefail

TARGET="${1:-all}"   # "exp5" | "rebuild1" | "all"

TIME="1:45:00"

SRC_HOST=rouloul.polytechnique.fr
SRC_WORKDIR=/Data/guillaume.henrotte/SADT/SADT
SRC_MODELDIR=/Data/guillaume.henrotte/SADT/models
SRC_LOCALDIR=/Data/guillaume.henrotte/.local
SRC_NLTK=/Data/guillaume.henrotte/.nltk_data

WORKDIR=/Data/guillaume.henrotte/SADT/SADT
MODEL_PATH=/Data/guillaume.henrotte/SADT/models/llava-v1.5-7b

# Nodes with RTX A4000 (16 GB) on the SallesInfo partition
NODE_EXP5="ablette"
NODE_RB1="anchois"

submit_job() {
    local node="$1"
    local job_name="$2"
    local exp_cmd="$3"

    sbatch \
      --job-name="$job_name" \
      --nodelist="$node" \
      --cpus-per-task=4 \
      --mem=48G \
      --time="$TIME" \
      --output="/users/eleves-b/2024/guillaume.henrotte/slurm_%j.out" \
      --error="/users/eleves-b/2024/guillaume.henrotte/slurm_%j.err" \
      << SBATCH
#!/usr/bin/env bash
set -euo pipefail

RSYNC_SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no"

mkdir -p $WORKDIR /Data/guillaume.henrotte/SADT/models \
         /Data/guillaume.henrotte/.local /Data/guillaume.henrotte/.nltk_data

echo "=== [$job_name] Syncing code ==="
rsync -a --info=progress2 -e "\$RSYNC_SSH" $SRC_HOST:$SRC_WORKDIR/ $WORKDIR/

echo "=== [$job_name] Syncing Python packages ==="
rsync -a --info=progress2 -e "\$RSYNC_SSH" $SRC_HOST:$SRC_LOCALDIR/ /Data/guillaume.henrotte/.local/

echo "=== [$job_name] Syncing model (skip existing) ==="
rsync -a --ignore-existing --info=progress2 -e "\$RSYNC_SSH" \
    $SRC_HOST:$SRC_MODELDIR/ /Data/guillaume.henrotte/SADT/models/

echo "=== [$job_name] Syncing NLTK ==="
rsync -a --ignore-existing -e "\$RSYNC_SSH" \
    $SRC_HOST:$SRC_NLTK/ /Data/guillaume.henrotte/.nltk_data/

export PYTHONUSERBASE=/Data/guillaume.henrotte/.local
export PYTHONPATH=$WORKDIR:/Data/guillaume.henrotte/.local/lib/python3.9/site-packages\${PYTHONPATH:+:\$PYTHONPATH}
export NLTK_DATA=/Data/guillaume.henrotte/.nltk_data
export CUDA_VISIBLE_DEVICES=0

cd $WORKDIR

sync_back() {
  echo "=== [$job_name] Syncing outputs back ==="
  rsync -a -e "\$RSYNC_SSH" $WORKDIR/outputs/ $SRC_HOST:$SRC_WORKDIR/outputs/
}

$exp_cmd

sync_back
echo "=== [$job_name] Done ==="
SBATCH
}

EXP5_CMD="python -m SAUQ.exp5_locality_uncertainty \
    --model-path $MODEL_PATH \
    --max-new-tokens 120 \
    --output-dir outputs/exp5_locality"

RB1_CMD="python -m SAUQ.rebuild_1_copy_task \
    --model-path $MODEL_PATH \
    --ns 10 20 30 50 75 100 \
    --trials 3 \
    --seed 42 \
    --output-dir outputs/rebuild_1_copy_task"

case "$TARGET" in
  exp5)
    echo "Submitting Exp 5 on $NODE_EXP5 …"
    submit_job "$NODE_EXP5" "sauq_exp5" "$EXP5_CMD"
    ;;
  rebuild1)
    echo "Submitting Rebuild 1 on $NODE_RB1 …"
    submit_job "$NODE_RB1" "sauq_rb1" "$RB1_CMD"
    ;;
  all)
    echo "Submitting Exp 5 on $NODE_EXP5 …"
    submit_job "$NODE_EXP5" "sauq_exp5" "$EXP5_CMD"
    echo "Submitting Rebuild 1 on $NODE_RB1 …"
    submit_job "$NODE_RB1" "sauq_rb1" "$RB1_CMD"
    ;;
  *)
    echo "Unknown target: $TARGET. Use: exp5 | rebuild1 | all"
    exit 1
    ;;
esac

echo "Jobs submitted. Monitor with: squeue -u guillaume.henrotte"
