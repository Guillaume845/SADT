"""Exp 2b — Per-word TDO: concentrated visual routing separates grounded from hallucinated words.

Hypothesis: for a well-grounded object word, attention concentrates on a few specific
patches consistently across layers → A_word ∈ R^{L×P} is approximately rank-1 →
few active TDO directions → LOW Participation Ratio.

For a hallucinated word the PR depends on the hallucination type:
  Type 1 (visual_uncertainty): model looks for visual evidence but can't find it →
    attention still concentrates, possibly on sink patches → PR does NOT rise much.
  Type 2 (contextual_prior): object is generated from language priors, not vision →
    attention has no specific anchor → diffuse across patches → HIGHER PR.

Core prediction: HIGH PR correlates with Type 2 hallucinations, not Type 1.

Ground truth types are obtained via HARM (mask top patches + regenerate + CHAIR):
  word disappears after masking → Type 1 (visual evidence was load-bearing)
  word persists  after masking → Type 2 (language prior, vision irrelevant)

A_word[l, j] = attention from the word's token to patch j at layer l.
Shape: [L × P],  L = len(image_attention_layers),  P = n_image_patches.

Outputs
-------
outputs/exp2_per_word_tdo/per_word.json   — PR + type label per (image, word)
outputs/exp2_per_word_tdo/summary.json    — AUC/Mann-Whitney split by label type
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.sadt.core import (
    SADTConfig,
    SADTDetection,
    _attention_for_token,
    _plural,
    _response_ids,
    detect_hallucinations,
    find_token_spans,
    generate_with_trace,
)
from tools.sadt.pipeline_utils import (
    chair_words,
    collect_images,
    load_chair_evaluator,
    load_labels,
    mask_only_words,
    parse_layers,
    prepare_vcd_image_tensor,
    save_json_output,
)
from experiments.utils import (
    compute_tdo_participation_ratio,
    compute_spectral_leading_ratio,
    default_config,
    get_img_idx,
    load_model,
)

# label constants
GROUNDED = 0
TYPE1    = 1   # visual_uncertainty  — disappears after masking
TYPE2    = 2   # contextual_prior    — persists  after masking


# ---------------------------------------------------------------------------
# HARM: classify each detected hallucination as Type 1 or Type 2
# ---------------------------------------------------------------------------

def harm_classify(
    *,
    image_file: str,
    image_name: str,
    detected_words: List[str],
    detections: List[SADTDetection],
    image_processor: Any,
    model: Any,
    tokenizer: Any,
    model_name: str,
    config: SADTConfig,
    chair_evaluator: Any,
    max_new_tokens: int,
    temperature: float,
) -> Dict[str, int]:
    """Return {word: TYPE1 | TYPE2} for each detected hallucination word."""
    if not detected_words:
        return {}

    original_image_sizes = [Image.open(image_file).size]

    masked_image = mask_only_words(
        image_file=image_file,
        detections=detections,
        words=detected_words,
        config=config,
        image_mean=getattr(image_processor, "image_mean", None),
    )
    if masked_image is None:
        return {w: TYPE1 for w in detected_words}

    masked_tensor, _ = prepare_vcd_image_tensor(
        masked_image, image_processor, model, image_sizes=original_image_sizes,
    )
    _, _, harm_response = generate_with_trace(
        model, tokenizer, image_processor, model_name, image_file, "",
        image_tensor=masked_tensor,
        image_sizes=original_image_sizes,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    harm_chair = chair_words(chair_evaluator, image_name, harm_response)
    harm_hallu = set(harm_chair.get("hallucination_words") or []) if harm_chair else set()

    return {w: (TYPE2 if w in harm_hallu else TYPE1) for w in detected_words}


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def auc_from_scores(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    u = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return float(u / (len(pos) * len(neg)))


def mann_whitney_u_p(scores: np.ndarray, labels: np.ndarray) -> float:
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


def pr_stats(pr: np.ndarray, labels: np.ndarray, target: int) -> Dict:
    mask = labels == target
    vals = pr[mask]
    ref  = pr[labels == GROUNDED]
    bin_labels = np.concatenate([np.ones(len(vals)), np.zeros(len(ref))])
    bin_scores = np.concatenate([vals, ref])
    return {
        "n": int(mask.sum()),
        "median": float(np.median(vals)) if len(vals) else float("nan"),
        "mean":   float(np.mean(vals))   if len(vals) else float("nan"),
        "auc_vs_grounded": auc_from_scores(bin_scores, bin_labels),
        "mann_whitney_p":  mann_whitney_u_p(bin_scores, bin_labels),
    }


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
    p.add_argument("--image-attention-layers", default="19:25")
    p.add_argument("--topk-patches", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--output-dir", default="outputs/exp2_per_word_tdo")
    return p


def main():
    args = build_parser().parse_args()
    tokenizer, model, image_processor, model_name = load_model(args.model_path, args.model_base)

    config = default_config(
        image_attention_layers=parse_layers(args.image_attention_layers),
        topk_patches=args.topk_patches,
    )

    labels = load_labels(args.labels_json)
    chair_evaluator = load_chair_evaluator(args.chair_cache, None)
    if chair_evaluator is None:
        raise ValueError("CHAIR evaluator required. Pass --chair-cache.")

    images = collect_images(args.image_file, args.image_dir, args.limit)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_word: List[Dict] = []

    for image_file in tqdm(images):
        image_name = Path(image_file).name

        output, input_ids, response = generate_with_trace(
            model, tokenizer, image_processor, model_name, image_file, args.prompt,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        )

        img_idx = get_img_idx(input_ids)
        generated_ids = _response_ids(output, input_ids)

        chair_item = chair_words(chair_evaluator, image_name, response)
        if not chair_item:
            continue

        generated_words = [
            w for w in (chair_item.get("generated_words") or [])
            if w not in config.excluded_words
        ]
        hallu_set = set(
            w for w in (chair_item.get("hallucination_words") or [])
            if w not in config.excluded_words
        )

        # LLCC detections needed for HARM masking (only if there are hallucinations)
        hallu_words_list = [w for w in generated_words if w in hallu_set]
        type_labels: Dict[str, int] = {}
        if hallu_words_list:
            detections = detect_hallucinations(
                words=generated_words, output=output, input_ids=input_ids,
                model=model, tokenizer=tokenizer, config=config,
            )
            detected = [d.word for d in detections if d.is_hallucination]
            # Only run HARM for words that LLCC also detects as hallucinations
            harm_words = [w for w in hallu_words_list if w in detected]
            if harm_words:
                type_labels = harm_classify(
                    image_file=image_file,
                    image_name=image_name,
                    detected_words=harm_words,
                    detections=detections,
                    image_processor=image_processor,
                    model=model,
                    tokenizer=tokenizer,
                    model_name=model_name,
                    config=config,
                    chair_evaluator=chair_evaluator,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )

        # Per-word TDO
        for word in generated_words:
            token_ids = tokenizer(word, add_special_tokens=False)["input_ids"]
            token_indices = find_token_spans(token_ids, generated_ids)
            if not token_indices:
                plural = _plural(word)
                if plural:
                    token_ids = tokenizer(plural, add_special_tokens=False)["input_ids"]
                    token_indices = find_token_spans(token_ids, generated_ids)
            if not token_indices:
                continue

            token_idx = token_indices[0]

            # A_word[l, j] = attention from this token to patch j at layer l  [L × P]
            attention_img = _attention_for_token(
                output["attentions"], token_idx, img_idx, config.image_token_count
            )
            A_word = attention_img[list(config.image_attention_layers)]

            pr      = compute_tdo_participation_ratio(A_word)
            leading = compute_spectral_leading_ratio(A_word)

            if word in hallu_set:
                label = type_labels.get(word, TYPE1)  # default TYPE1 if HARM not run
            else:
                label = GROUNDED

            per_word.append({
                "img_id":    image_name,
                "word":      word,
                "label":     label,   # 0=grounded, 1=type1, 2=type2
                "pr":        pr,
                "leading_ratio": leading,
            })
            label_str = {GROUNDED: "grounded", TYPE1: "type1", TYPE2: "type2"}[label]
            print(f"  {image_name} [{word}] {label_str}  PR={pr:.4f}  λ₁/Σ={leading:.4f}")

    if not per_word:
        print("No words found. Check --limit or data paths.")
        return

    save_json_output(str(out_dir / "per_word.json"), {"records": per_word})

    pr_arr    = np.array([r["pr"]            for r in per_word])
    lead_arr  = np.array([r["leading_ratio"] for r in per_word])
    label_arr = np.array([r["label"]         for r in per_word])

    n_grounded = int((label_arr == GROUNDED).sum())
    n_type1    = int((label_arr == TYPE1).sum())
    n_type2    = int((label_arr == TYPE2).sum())

    summary = {
        "n_words":    len(per_word),
        "n_grounded": n_grounded,
        "n_type1":    n_type1,
        "n_type2":    n_type2,
        "grounded_pr_median": float(np.median(pr_arr[label_arr == GROUNDED])) if n_grounded else float("nan"),
        # Type 1 vs grounded — hypothesis: NO significant difference in PR
        "type1_vs_grounded": pr_stats(pr_arr, label_arr, TYPE1),
        # Type 2 vs grounded — hypothesis: HIGH PR for type2 → AUC > 0.5
        "type2_vs_grounded": pr_stats(pr_arr, label_arr, TYPE2),
        # Leading ratio same split
        "type1_leading_vs_grounded": pr_stats(lead_arr, label_arr, TYPE1),
        "type2_leading_vs_grounded": pr_stats(lead_arr, label_arr, TYPE2),
    }

    save_json_output(str(out_dir / "summary.json"), summary)

    print("\n=== Exp 2b — Per-word TDO ===")
    print(f"N words: {len(per_word)}  "
          f"(grounded={n_grounded}, type1={n_type1}, type2={n_type2})")
    print(f"Grounded PR median:  {summary['grounded_pr_median']:.4f}")
    t1 = summary["type1_vs_grounded"]
    t2 = summary["type2_vs_grounded"]
    print(f"Type 1 PR median:    {t1['median']:.4f}  "
          f"AUC vs grounded: {t1['auc_vs_grounded']:.4f}  p={t1['mann_whitney_p']:.4f}")
    print(f"Type 2 PR median:    {t2['median']:.4f}  "
          f"AUC vs grounded: {t2['auc_vs_grounded']:.4f}  p={t2['mann_whitney_p']:.4f}")
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
