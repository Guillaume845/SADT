"""Exp 2d — Object-tokens TDO: one matrix per image, rows = object token positions.

For each image, build A_objects by selecting only the rows of the full [T × P]
attention matrix that correspond to object word tokens:

    A_objects[[t_obj1, t_obj2, ..., t_objN], :]   shape [N_objects × P]

This tests whether the JOINT visual routing of all mentioned objects is more
concentrated (low PR) for clean images than for images with hallucinations.

Hypothesis:
  - Clean image: each object attends to distinct, specific patches → A_objects has
    N independent concentrated directions → PR reflects compact, specific coverage.
  - Image with Type 2 hallucination: the hallucinated object's row is diffuse
    (no visual anchor) → adds a "noisy" direction → raises PR.

HARM provides per-word Type 1 / Type 2 labels. At the image level we distinguish:
  - all_grounded:  no hallucinations
  - has_type1:     at least one Type 1 hallucination, no Type 2
  - has_type2:     at least one Type 2 hallucination

Outputs
-------
outputs/exp2_objects_tdo/per_image.json  — PR per image with type breakdown
outputs/exp2_objects_tdo/summary.json    — AUC / Mann-Whitney by image class
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
    compute_full_attention_matrix, compute_tdo_participation_ratio,
    compute_spectral_leading_ratio, default_config, get_img_idx, load_model,
)

GROUNDED, TYPE1, TYPE2 = 0, 1, 2
IMG_ALL_GROUNDED, IMG_HAS_TYPE1, IMG_HAS_TYPE2 = 0, 1, 2


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


def stats_vs_ref(pr_arr, label_arr, target, ref=IMG_ALL_GROUNDED):
    mask = label_arr == target
    vals = pr_arr[mask]
    ref_vals = pr_arr[label_arr == ref]
    if not len(vals) or not len(ref_vals):
        return {"n": int(mask.sum()), "median": float("nan"), "auc": float("nan"), "p": float("nan")}
    bl = np.concatenate([np.ones(len(vals)), np.zeros(len(ref_vals))])
    bs = np.concatenate([vals, ref_vals])
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
    p.add_argument("--output-dir", default="outputs/exp2_objects_tdo")
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

    per_image: List[Dict] = []

    for image_file in tqdm(images):
        image_name = Path(image_file).name
        output, input_ids, response = generate_with_trace(
            model, tokenizer, image_processor, model_name, image_file, args.prompt,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        )
        img_idx       = get_img_idx(input_ids)
        generated_ids = _response_ids(output, input_ids)

        chair_item = chair_words(chair_eval, image_name, response)
        if not chair_item:
            continue

        generated_words = [w for w in (chair_item.get("generated_words") or [])
                           if w not in config.excluded_words]
        hallu_set = set(w for w in (chair_item.get("hallucination_words") or [])
                        if w not in config.excluded_words)

        # Full [T × P] attention matrix (all generated tokens)
        A_full = compute_full_attention_matrix(
            output, img_idx, config.image_token_count, config.image_attention_layers
        )

        # HARM type labels
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

        # Collect object token positions
        object_rows = []
        word_labels: Dict[str, int] = {}
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
            t_star = spans[0]
            if t_star < len(A_full):
                object_rows.append(t_star)
            word_labels[word] = type_labels.get(word, TYPE1) if word in hallu_set else GROUNDED

        if not object_rows:
            continue

        # A_objects: only the rows corresponding to object tokens
        A_objects = A_full[object_rows, :]  # [N_objects × P]
        pr      = compute_tdo_participation_ratio(A_objects)
        leading = compute_spectral_leading_ratio(A_objects)

        # Image-level class
        word_type_set = set(word_labels.values())
        if TYPE2 in word_type_set:
            img_class = IMG_HAS_TYPE2
        elif TYPE1 in word_type_set:
            img_class = IMG_HAS_TYPE1
        else:
            img_class = IMG_ALL_GROUNDED

        per_image.append({
            "img_id":    image_name,
            "img_class": img_class,
            "n_objects": len(object_rows),
            "word_types": {w: int(v) for w, v in word_labels.items()},
            "pr":         pr,
            "leading_ratio": leading,
        })
        cls_str = {IMG_ALL_GROUNDED: "all_grounded", IMG_HAS_TYPE1: "has_type1",
                   IMG_HAS_TYPE2: "has_type2"}[img_class]
        print(f"{image_name} [{cls_str}, N_obj={len(object_rows)}]  PR={pr:.4f}")

    if not per_image:
        print("No images processed.")
        return

    save_json_output(str(out_dir / "per_image.json"), {"records": per_image})

    pr_arr    = np.array([r["pr"]        for r in per_image])
    label_arr = np.array([r["img_class"] for r in per_image])

    summary = {
        "n_images":        len(per_image),
        "n_all_grounded":  int((label_arr == IMG_ALL_GROUNDED).sum()),
        "n_has_type1":     int((label_arr == IMG_HAS_TYPE1).sum()),
        "n_has_type2":     int((label_arr == IMG_HAS_TYPE2).sum()),
        "grounded_pr_median": float(np.median(pr_arr[label_arr == IMG_ALL_GROUNDED])),
        "type1_vs_grounded":  stats_vs_ref(pr_arr, label_arr, IMG_HAS_TYPE1),
        "type2_vs_grounded":  stats_vs_ref(pr_arr, label_arr, IMG_HAS_TYPE2),
    }
    save_json_output(str(out_dir / "summary.json"), summary)

    print("\n=== Exp 2d — Object-tokens TDO ===")
    print(f"N={summary['n_images']}  grounded={summary['n_all_grounded']}  "
          f"type1={summary['n_has_type1']}  type2={summary['n_has_type2']}")
    print(f"Grounded median PR: {summary['grounded_pr_median']:.4f}")
    t1, t2 = summary["type1_vs_grounded"], summary["type2_vs_grounded"]
    print(f"Type1  median={t1['median']:.4f}  AUC={t1['auc']:.4f}  p={t1['p']:.4f}")
    print(f"Type2  median={t2['median']:.4f}  AUC={t2['auc']:.4f}  p={t2['p']:.4f}")
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
