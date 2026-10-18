"""Exp 2c — Causal-prefix TDO: visual routing up to each object token.

For each object word at generation position t*, compute the TDO participation
ratio of the CAUSAL prefix matrix A[:t*+1, P]:

    A[t, j] = sum_{l in layers} attention(token t → patch j at layer l)

This is the same [T × P] structure as SAUQ, but evaluated at each object token's
timestep rather than globally over the full caption.

Hypothesis: at the moment a Type 2 (contextual prior) hallucination is generated,
the preceding visual routing is diffuse — the model has not been grounding tokens
on specific patches — so PR(A[:t*+1]) is HIGH.

For Type 1 (visual uncertainty) the routing may still be concentrated (just on the
wrong patches), so PR should not differ significantly from grounded words.

HARM (mask + regen + CHAIR) provides the Type 1 / Type 2 ground truth.

Outputs
-------
outputs/exp2_causal_tdo/per_word.json   — PR per (image, word) with type label
outputs/exp2_causal_tdo/summary.json    — AUC / Mann-Whitney by label type
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.sadt.core import (
    SADTConfig, SADTDetection,
    _attention_for_token, _plural, _response_ids,
    detect_hallucinations, find_token_spans, generate_with_trace,
)
from tools.sadt.pipeline_utils import (
    chair_words, collect_images, load_chair_evaluator, load_labels,
    mask_only_words, parse_layers, prepare_vcd_image_tensor, save_json_output,
)
from experiments.utils import (
    compute_tdo_participation_ratio, compute_spectral_leading_ratio,
    default_config, get_img_idx, load_model,
)

GROUNDED, TYPE1, TYPE2 = 0, 1, 2


def compute_prefix_attention_matrix(output, img_idx, image_token_count, layers, max_t):
    """A[:max_t+1, P] summed over target layers."""
    attentions = output["attentions"]
    rows = []
    for t in range(min(max_t + 1, len(attentions))):
        att = _attention_for_token(attentions, t, img_idx, image_token_count)
        rows.append(att[list(layers)].sum(axis=0))
    return np.stack(rows)  # [t*+1, P]


def harm_classify(*, image_file, image_name, detected_words, detections,
                  image_processor, model, tokenizer, model_name, config,
                  chair_evaluator, max_new_tokens, temperature):
    if not detected_words:
        return {}
    original_image_sizes = [Image.open(image_file).size]
    masked_image = mask_only_words(
        image_file=image_file, detections=detections, words=detected_words,
        config=config, image_mean=getattr(image_processor, "image_mean", None),
    )
    if masked_image is None:
        return {w: TYPE1 for w in detected_words}
    masked_tensor, _ = prepare_vcd_image_tensor(
        masked_image, image_processor, model, image_sizes=original_image_sizes,
    )
    _, _, harm_response = generate_with_trace(
        model, tokenizer, image_processor, model_name, image_file, "",
        image_tensor=masked_tensor, image_sizes=original_image_sizes,
        max_new_tokens=max_new_tokens, temperature=temperature,
    )
    harm_chair = chair_words(chair_evaluator, image_name, harm_response)
    harm_hallu = set(harm_chair.get("hallucination_words") or []) if harm_chair else set()
    return {w: (TYPE2 if w in harm_hallu else TYPE1) for w in detected_words}


def auc_from_scores(scores, labels):
    pos, neg = scores[labels == 1], scores[labels == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    u = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return float(u / (len(pos) * len(neg)))


def mann_whitney_p(scores, labels):
    try:
        from scipy.stats import mannwhitneyu
        pos, neg = scores[labels == 1], scores[labels == 0]
        if not len(pos) or not len(neg):
            return float("nan")
        _, p = mannwhitneyu(pos, neg, alternative="two-sided")
        return float(p)
    except ImportError:
        return float("nan")


def stats_vs_grounded(pr_arr, label_arr, target):
    mask = label_arr == target
    vals = pr_arr[mask]
    ref  = pr_arr[label_arr == GROUNDED]
    if not len(vals) or not len(ref):
        return {"n": int(mask.sum()), "median": float("nan"), "auc": float("nan"), "p": float("nan")}
    bl = np.concatenate([np.ones(len(vals)), np.zeros(len(ref))])
    bs = np.concatenate([vals, ref])
    return {
        "n":      int(mask.sum()),
        "median": float(np.median(vals)),
        "auc":    auc_from_scores(bs, bl),
        "p":      mann_whitney_p(bs, bl),
    }


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
    p.add_argument("--output-dir", default="outputs/exp2_causal_tdo")
    return p


def main():
    args = build_parser().parse_args()
    tokenizer, model, image_processor, model_name = load_model(args.model_path, args.model_base)
    config = default_config(
        image_attention_layers=parse_layers(args.image_attention_layers),
        topk_patches=args.topk_patches,
    )
    labels     = load_labels(args.labels_json)
    chair_eval = load_chair_evaluator(args.chair_cache, None)
    if chair_eval is None:
        raise ValueError("CHAIR evaluator required.")

    images  = collect_images(args.image_file, args.image_dir, args.limit)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_word: List[Dict] = []

    for image_file in tqdm(images):
        image_name = Path(image_file).name
        output, input_ids, response = generate_with_trace(
            model, tokenizer, image_processor, model_name, image_file, args.prompt,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        )
        img_idx      = get_img_idx(input_ids)
        generated_ids = _response_ids(output, input_ids)

        chair_item = chair_words(chair_eval, image_name, response)
        if not chair_item:
            continue

        generated_words = [w for w in (chair_item.get("generated_words") or [])
                           if w not in config.excluded_words]
        hallu_set = set(w for w in (chair_item.get("hallucination_words") or [])
                        if w not in config.excluded_words)

        # HARM to get type labels for hallucinated words
        hallu_list = [w for w in generated_words if w in hallu_set]
        type_labels: Dict[str, int] = {}
        if hallu_list:
            detections = detect_hallucinations(
                words=generated_words, output=output, input_ids=input_ids,
                model=model, tokenizer=tokenizer, config=config,
            )
            detected = [d.word for d in detections if d.is_hallucination]
            harm_words = [w for w in hallu_list if w in detected]
            if harm_words:
                type_labels = harm_classify(
                    image_file=image_file, image_name=image_name,
                    detected_words=harm_words, detections=detections,
                    image_processor=image_processor, model=model,
                    tokenizer=tokenizer, model_name=model_name,
                    config=config, chair_evaluator=chair_eval,
                    max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                )

        for word in generated_words:
            token_ids = tokenizer(word, add_special_tokens=False)["input_ids"]
            spans = find_token_spans(token_ids, generated_ids)
            if not spans:
                plural = _plural(word)
                if plural:
                    token_ids = tokenizer(plural, add_special_tokens=False)["input_ids"]
                    spans = find_token_spans(token_ids, generated_ids)
            if not spans:
                continue

            t_star = spans[0]  # position of the object token in the generated sequence

            # Causal prefix: A[:t*+1, P]
            A_causal = compute_prefix_attention_matrix(
                output, img_idx, config.image_token_count,
                config.image_attention_layers, max_t=t_star,
            )

            pr      = compute_tdo_participation_ratio(A_causal)
            leading = compute_spectral_leading_ratio(A_causal)
            label   = type_labels.get(word, TYPE1) if word in hallu_set else GROUNDED

            per_word.append({
                "img_id": image_name, "word": word,
                "label": label, "t_star": t_star,
                "pr": pr, "leading_ratio": leading,
            })
            lstr = {GROUNDED: "grounded", TYPE1: "type1", TYPE2: "type2"}[label]
            print(f"  {image_name} [{word}@t={t_star}] {lstr}  PR={pr:.4f}")

    if not per_word:
        print("No words found.")
        return

    save_json_output(str(out_dir / "per_word.json"), {"records": per_word})

    pr_arr    = np.array([r["pr"]    for r in per_word])
    label_arr = np.array([r["label"] for r in per_word])

    summary = {
        "n_words":    len(per_word),
        "n_grounded": int((label_arr == GROUNDED).sum()),
        "n_type1":    int((label_arr == TYPE1).sum()),
        "n_type2":    int((label_arr == TYPE2).sum()),
        "grounded_pr_median": float(np.median(pr_arr[label_arr == GROUNDED])),
        "type1_vs_grounded":  stats_vs_grounded(pr_arr, label_arr, TYPE1),
        "type2_vs_grounded":  stats_vs_grounded(pr_arr, label_arr, TYPE2),
    }
    save_json_output(str(out_dir / "summary.json"), summary)

    print("\n=== Exp 2c — Causal-prefix TDO ===")
    print(f"N={summary['n_words']}  grounded={summary['n_grounded']}  "
          f"type1={summary['n_type1']}  type2={summary['n_type2']}")
    print(f"Grounded median PR: {summary['grounded_pr_median']:.4f}")
    t1, t2 = summary["type1_vs_grounded"], summary["type2_vs_grounded"]
    print(f"Type1  median={t1['median']:.4f}  AUC={t1['auc']:.4f}  p={t1['p']:.4f}")
    print(f"Type2  median={t2['median']:.4f}  AUC={t2['auc']:.4f}  p={t2['p']:.4f}")
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
