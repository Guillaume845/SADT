#!/usr/bin/env bash
# Run all three SADT × SAUQ experiments sequentially.
# Edit MODEL_PATH before running.

set -euo pipefail

# Packages and data are on /Data because home-dir quota is full
export PYTHONUSERBASE=/Data/guillaume.henrotte/.local
export PYTHONPATH=/Data/guillaume.henrotte/.local/lib/python3.9/site-packages${PYTHONPATH:+:$PYTHONPATH}
export NLTK_DATA=/Data/guillaume.henrotte/.nltk_data

MODEL_PATH="${MODEL_PATH:-/Data/guillaume.henrotte/SADT/models/llava-v1.5-7b}"
IMAGE_DIR="${IMAGE_DIR:-data/test_imgs}"
LABELS_JSON="${LABELS_JSON:-data/metadata/hallucination_results_val500.json}"
CHAIR_CACHE="${CHAIR_CACHE:-data/chair.pkl}"
LIMIT="${LIMIT:-}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
    LIMIT_ARG="--limit $LIMIT"
fi

COMMON_ARGS=(
    --model-path "$MODEL_PATH"
    --image-dir  "$IMAGE_DIR"
    --labels-json "$LABELS_JSON"
    --chair-cache "$CHAIR_CACHE"
    $LIMIT_ARG
)

echo "================================================================"
echo "Exp 1 — Sink-corrected LLCC"
echo "================================================================"
CUDA_VISIBLE_DEVICES=$GPU python -m experiments.exp1_sink_corrected_topk \
    "${COMMON_ARGS[@]}" \
    --output-dir outputs/exp1_sink_corrected

echo ""
echo "================================================================"
echo "Exp 2 — Spectral TDO (PR separation)"
echo "================================================================"
CUDA_VISIBLE_DEVICES=$GPU python -m experiments.exp2_spectral_tdo \
    "${COMMON_ARGS[@]}" \
    --output-dir outputs/exp2_spectral_tdo

echo ""
echo "================================================================"
echo "Exp 4 — Type classification without mask-and-regenerate"
echo "================================================================"
CUDA_VISIBLE_DEVICES=$GPU python -m experiments.exp4_type_classification \
    "${COMMON_ARGS[@]}" \
    --output-dir outputs/exp4_type_classification

echo ""
echo "All experiments done.  Results in outputs/exp{1,2,4}_*/"
