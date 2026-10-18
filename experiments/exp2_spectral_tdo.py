"""Exp 2 — Spectral TDO: Participation Ratio separates hallucinated descriptions.

Hypothesis: descriptions with object hallucinations have a different Participation
Ratio (PR) of the Token Difference Operator D_{μ,π} = Ω_μ⁻¹ A^T Ω_π A than
hallucination-free descriptions.

A ∈ R^{T×P} is the mean attention matrix (generated tokens × image patches)
averaged over layers 20-27.  PR = (Σλ)² / (n·Σλ²) ∈ (0,1].

High PR = many independent routing directions = richer visual coverage.
Low PR  = routing concentrated in few directions = less grounded generation.

Outputs
-------
outputs/exp2_spectral_tdo/per_image.json  — PR value + hallucination label per image
outputs/exp2_spectral_tdo/summary.json    — AUC, Mann-Whitney U, median PR by class
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.sadt.core import generate_with_trace
from tools.sadt.pipeline_utils import (
    chair_words,
    collect_images,
    load_chair_evaluator,
    load_labels,
    parse_layers,
    save_json_output,
)
from experiments.utils import (
    compute_full_attention_matrix,
    compute_tdo_participation_ratio,
    compute_spectral_leading_ratio,
    default_config,
    get_img_idx,
    load_model,
)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def auc_from_scores(scores: np.ndarray, labels: np.ndarray) -> float:
    """Simple AUC via Mann-Whitney U (no sklearn needed)."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    u = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return float(u / (len(pos) * len(neg)))


def mann_whitney_u_p(scores: np.ndarray, labels: np.ndarray) -> float:
    """Two-sided p-value via scipy if available, else returns nan."""
    try:
        from scipy.stats import mannwhitneyu
        pos = scores[labels == 1]
        neg = scores[labels == 0]
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        _, p = mannwhitneyu(pos, neg, alternative="two-sided")
        return float(p)
    except ImportError:
        return float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--model-base", default=None)
    p.add_argument("--image-file", default=None)
    p.add_argument("--image-dir", default="data/test_imgs")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--prompt", default="Describe this image.")
    p.add_argument("--labels-json", default="data/metadata/hallucination_results_val500.json")
    p.add_argument("--chair-cache", default="data/chair.pkl")
    p.add_argument("--coco-annotations", default=None)
    p.add_argument("--image-attention-layers", default="19:25")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--output-dir", default="outputs/exp2_spectral_tdo")
    return p


def main():
    args = build_parser().parse_args()
    tokenizer, model, image_processor, model_name = load_model(args.model_path, args.model_base)

    config = default_config(
        image_attention_layers=parse_layers(args.image_attention_layers),
    )

    labels = load_labels(args.labels_json)
    chair_evaluator = load_chair_evaluator(args.chair_cache, args.coco_annotations)
    if chair_evaluator is None:
        raise ValueError("CHAIR evaluator required. Pass --chair-cache or --coco-annotations.")

    images = collect_images(args.image_file, args.image_dir, args.limit)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_image: List[Dict] = []

    for image_file in tqdm(images):
        image_name = Path(image_file).name
        label_item = labels.get(image_name)

        output, input_ids, response = generate_with_trace(
            model, tokenizer, image_processor, model_name, image_file, args.prompt,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        )

        img_idx = get_img_idx(input_ids)

        # Full attention matrix A [T, P] summed over target layers
        A = compute_full_attention_matrix(
            output, img_idx, config.image_token_count, config.image_attention_layers
        )

        pr = compute_tdo_participation_ratio(A)
        leading = compute_spectral_leading_ratio(A)

        # Hallucination label from labels_json (1 = has any hallucination)
        gt_hallu_words = []
        if label_item:
            gt_hallu_words = [
                w for w in (label_item.get("hallucination_words") or [])
                if w not in config.excluded_words
            ]

        # Also get CHAIR-based hallucination words from the generated response
        chair_item = chair_words(chair_evaluator, image_name, response)
        chair_hallu = []
        if chair_item:
            chair_hallu = [w for w in (chair_item.get("hallucination_words") or [])
                           if w not in config.excluded_words]

        # Use labels_json hallucination_words as the primary label
        # Fall back to CHAIR if labels_json doesn't have hallucination_words
        hallu_words = gt_hallu_words if gt_hallu_words is not None else chair_hallu
        is_hallucinated = int(len(hallu_words) > 0)

        record = {
            "img_id": image_name,
            "is_hallucinated": is_hallucinated,
            "hallucination_words": hallu_words,
            "n_generated_tokens": len(output["attentions"]),
            "pr": pr,
            "leading_ratio": leading,
        }
        per_image.append(record)
        print(f"{image_name}: hallu={is_hallucinated} PR={pr:.4f} λ₁/Σ={leading:.4f}")

    # --- Analysis ---
    pr_arr = np.array([r["pr"] for r in per_image])
    lead_arr = np.array([r["leading_ratio"] for r in per_image])
    label_arr = np.array([r["is_hallucinated"] for r in per_image])

    n_hallu = int(label_arr.sum())
    n_clean = int((1 - label_arr).sum())

    pr_hallu = pr_arr[label_arr == 1]
    pr_clean = pr_arr[label_arr == 0]
    lead_hallu = lead_arr[label_arr == 1]
    lead_clean = lead_arr[label_arr == 0]

    # AUC: test if LOW PR predicts hallucination (scores inverted)
    auc_pr = auc_from_scores(-pr_arr, label_arr)
    auc_lead = auc_from_scores(lead_arr, label_arr)
    p_pr = mann_whitney_u_p(pr_arr, label_arr)
    p_lead = mann_whitney_u_p(lead_arr, label_arr)

    summary = {
        "n_images": len(per_image),
        "n_hallucinated": n_hallu,
        "n_clean": n_clean,
        "pr": {
            "hallu_median": float(np.median(pr_hallu)) if len(pr_hallu) else float("nan"),
            "hallu_mean": float(np.mean(pr_hallu)) if len(pr_hallu) else float("nan"),
            "clean_median": float(np.median(pr_clean)) if len(pr_clean) else float("nan"),
            "clean_mean": float(np.mean(pr_clean)) if len(pr_clean) else float("nan"),
            "auc_low_pr_predicts_hallu": auc_pr,
            "mann_whitney_p": p_pr,
        },
        "leading_ratio": {
            "hallu_median": float(np.median(lead_hallu)) if len(lead_hallu) else float("nan"),
            "clean_median": float(np.median(lead_clean)) if len(lead_clean) else float("nan"),
            "auc_high_lead_predicts_hallu": auc_lead,
            "mann_whitney_p": p_lead,
        },
    }

    save_json_output(str(out_dir / "per_image.json"), {"records": per_image})
    save_json_output(str(out_dir / "summary.json"), summary)

    print("\n=== Exp 2 — Spectral TDO ===")
    print(f"N images: {len(per_image)}  ({n_hallu} hallucinated, {n_clean} clean)")
    print(f"PR  — hallucinated median: {summary['pr']['hallu_median']:.4f}  "
          f"clean median: {summary['pr']['clean_median']:.4f}  "
          f"AUC: {auc_pr:.4f}  p={p_pr:.4f}")
    print(f"λ₁/Σ — hallucinated median: {summary['leading_ratio']['hallu_median']:.4f}  "
          f"clean median: {summary['leading_ratio']['clean_median']:.4f}  "
          f"AUC: {auc_lead:.4f}  p={p_lead:.4f}")
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
