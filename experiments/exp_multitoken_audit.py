"""Empirical audit: LLCC behaviour on mono-token vs multi-token COCO objects.

For each COCO object word generated in the caption we record:
  - n_subtokens          : number of LLaVA sub-tokens the word maps to
  - subtoken_ids         : the raw token IDs
  - subtoken_strings     : decoded sub-token strings
  - attention_score      : avg image-attention at the FIRST sub-token position
  - passes_threshold     : attention_score > tau_attn
  - top3_patches         : top-3 patch indices
  - logitlens_tokens_per_layer : per-layer argmax visual tokens (list of lists)
  - right_num            : number of (layer, patch) pairs where Sim=True
  - is_hallucination_pred: LLCC decision (object and right_num==0)
  - is_hallucination_gt  : ground-truth from CHAIR labels

Output
------
outputs/multitoken_audit/
  records.json     — per-(image, word) rows
  summary.json     — precision/recall split by mono/multi-token
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.sadt.core import (
    SADTConfig,
    _attention_for_token,
    _response_ids,
    find_token_spans,
    generate_with_trace,
    is_semantically_consistent,
    _plural,
    _norm_token,
)
from tools.sadt.pipeline_utils import (
    chair_words,
    collect_images,
    load_chair_evaluator,
    load_labels,
    parse_layers,
)
from experiments.utils import get_img_idx, load_model


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


def tokenize_word(word: str, tokenizer) -> List[int]:
    return tokenizer(word, add_special_tokens=False)["input_ids"]


def run_llcc_detailed(
    word: str,
    output: Any,
    input_ids,
    model: Any,
    tokenizer: Any,
    config: SADTConfig,
) -> Optional[Dict]:
    """Run LLCC exactly as detect_hallucinations does, but record more detail."""
    import torch
    attentions = output["attentions"]
    hidden_states = output["hidden_states"][0]
    generated_ids = _response_ids(output, input_ids)
    img_idx = get_img_idx(input_ids)

    word_norm = word.lower().strip()
    if word_norm in config.excluded_words:
        return None

    token_ids = tokenize_word(word_norm, tokenizer)
    token_indices = find_token_spans(token_ids, generated_ids)
    if not token_indices:
        plural = _plural(word_norm)
        if plural:
            token_ids = tokenize_word(plural, tokenizer)
            token_indices = find_token_spans(token_ids, generated_ids)
    if not token_indices:
        return None

    n_subtokens = len(token_ids)
    subtoken_strings = [tokenizer.decode([tid], skip_special_tokens=False) for tid in token_ids]

    # FIRST sub-token position — exactly what the paper uses
    token_idx = token_indices[0]
    attention_img = _attention_for_token(attentions, token_idx, img_idx, config.image_token_count)
    avg_attention = float(attention_img[list(config.image_attention_layers)].sum(axis=1).mean())
    is_object = avg_attention > config.image_attention_threshold

    agg = attention_img[list(config.image_attention_layers)].sum(axis=0)
    k = min(config.topk_patches, agg.size)
    top_patches = [int(x) for x in np.argsort(agg)[::-1][:k]]

    logitlens_per_layer: Dict[int, List[str]] = {}
    right_num = 0

    with torch.no_grad():
        for layer in config.image_attention_layers:
            layer_att = attention_img[layer]
            lk = min(config.topk_patches, layer_att.size)
            layer_patches = np.argsort(layer_att)[::-1][:lk]
            abs_idx = torch.as_tensor(layer_patches + img_idx, device=model.device)
            li = max(0, min(layer, len(hidden_states) - 1))
            vis_h = hidden_states[li][:, abs_idx, :].to(model.device)
            vis_logits = model.lm_head(vis_h)
            tids = torch.argmax(vis_logits, dim=-1).squeeze(0)
            vis_toks = [tokenizer.decode([int(t)], skip_special_tokens=False) for t in tids]
            logitlens_per_layer[int(layer)] = vis_toks
            right_num += sum(is_semantically_consistent(word_norm, t, config) for t in vis_toks)

    # Also record LAST sub-token attention (for comparison)
    last_token_idx_in_gen = token_indices[0] + n_subtokens - 1
    if last_token_idx_in_gen < len(attentions):
        att_last = _attention_for_token(attentions, last_token_idx_in_gen, img_idx, config.image_token_count)
        avg_att_last = float(att_last[list(config.image_attention_layers)].sum(axis=1).mean())
    else:
        avg_att_last = None

    return {
        "word": word_norm,
        "n_subtokens": n_subtokens,
        "subtoken_ids": token_ids,
        "subtoken_strings": subtoken_strings,
        "attention_first_subtoken": avg_attention,
        "attention_last_subtoken": avg_att_last,
        "passes_threshold": is_object,
        "top3_patches": top_patches,
        "logitlens_per_layer": {str(k): v for k, v in logitlens_per_layer.items()},
        "right_num": right_num,
        "is_hallucination_pred": bool(is_object and right_num == 0),
    }


def detection_metrics(records, key="is_hallucination_pred"):
    tp = fp = fn = 0
    for r in records:
        pred = r[key]
        gt = r["is_hallucination_gt"]
        if gt and pred:
            tp += 1
        elif pred and not gt:
            fp += 1
        elif gt and not pred:
            fn += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "n": tp + fp + fn}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--image-dir", default="data/test_imgs")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--labels-json", default="data/metadata/hallucination_results_val500.json")
    p.add_argument("--chair-cache", default=None)
    p.add_argument("--image-attention-layers", default="20:27")
    p.add_argument("--image-attention-threshold", type=float, default=0.0)
    p.add_argument("--semantic-matcher", default="coco")
    p.add_argument("--output-dir", default="outputs/multitoken_audit")
    return p


def main():
    args = build_parser().parse_args()
    tokenizer, model, image_processor, model_name = load_model(args.model_path)

    config = SADTConfig(
        image_attention_layers=tuple(parse_layers(args.image_attention_layers)),
        image_attention_threshold=args.image_attention_threshold,
        topk_patches=3,
        semantic_matcher=args.semantic_matcher,
    )

    labels = load_labels(args.labels_json)
    chair_evaluator = load_chair_evaluator(args.chair_cache, None)
    images = collect_images(None, args.image_dir, args.limit)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tokenize all COCO classes once
    coco_token_counts = {cls: len(tokenize_word(cls, tokenizer)) for cls in COCO_CLASSES}

    all_rows: List[Dict] = []

    for image_file in tqdm(images):
        image_name = Path(image_file).name
        label_item = labels.get(image_name)
        if not label_item:
            continue

        output, input_ids, response = generate_with_trace(
            model, tokenizer, image_processor, model_name, image_file,
            "Describe this image.", max_new_tokens=256, temperature=0.0,
        )

        our_chair = chair_words(chair_evaluator, image_name, response)
        if our_chair:
            object_words = [w for w in our_chair["generated_words"] if w not in config.excluded_words]
            gt_hallu = set(our_chair["hallucination_words"]) - set(config.excluded_words)
        else:
            object_words = [w for w in (label_item.get("generated_words") or []) if w not in config.excluded_words]
            gt_hallu = set(label_item.get("hallucination_words") or []) - set(config.excluded_words)

        for word in object_words:
            row = run_llcc_detailed(word, output, input_ids, model, tokenizer, config)
            if row is None:
                continue
            row["img_id"] = image_name
            row["is_hallucination_gt"] = bool(word in gt_hallu)
            row["is_multi_token"] = row["n_subtokens"] > 1
            # COCO canonical token count (word may be a synonym, use as-is)
            row["coco_n_subtokens"] = coco_token_counts.get(word, row["n_subtokens"])
            all_rows.append(row)

        print(f"{image_name}: {len(object_words)} words, "
              f"{sum(1 for r in all_rows if r['img_id']==image_name and r['is_multi_token'])} multi-token")

    if not all_rows:
        print("No rows collected.")
        return

    # Save all rows
    rows_serializable = []
    for r in all_rows:
        r2 = dict(r)
        r2["logitlens_per_layer"] = {k: v for k, v in r["logitlens_per_layer"].items()}
        rows_serializable.append(r2)

    with open(out_dir / "records.json", "w") as f:
        json.dump(rows_serializable, f, indent=2)

    # --- Summary ---
    mono = [r for r in all_rows if not r["is_multi_token"]]
    multi = [r for r in all_rows if r["is_multi_token"]]

    # Words that pass the attention threshold
    mono_pass = [r for r in mono if r["passes_threshold"]]
    multi_pass = [r for r in multi if r["passes_threshold"]]

    metrics_mono = detection_metrics(mono)
    metrics_multi = detection_metrics(multi)
    metrics_mono_pass = detection_metrics(mono_pass)
    metrics_multi_pass = detection_metrics(multi_pass)

    # Break down multi-token by number of sub-tokens
    by_ntok = {}
    for r in multi:
        n = r["n_subtokens"]
        by_ntok.setdefault(n, []).append(r)
    ntok_metrics = {n: detection_metrics(rows) for n, rows in by_ntok.items()}

    # Examples of multi-token FP/FN for report
    multi_fp = [r for r in multi if r["is_hallucination_pred"] and not r["is_hallucination_gt"]]
    multi_fn = [r for r in multi if not r["is_hallucination_pred"] and r["is_hallucination_gt"]]

    summary = {
        "n_images": len(set(r["img_id"] for r in all_rows)),
        "n_words_total": len(all_rows),
        "n_mono_token": len(mono),
        "n_multi_token": len(multi),
        "metrics_all": detection_metrics(all_rows),
        "metrics_mono_token": metrics_mono,
        "metrics_multi_token": metrics_multi,
        "metrics_mono_pass_threshold": metrics_mono_pass,
        "metrics_multi_pass_threshold": metrics_multi_pass,
        "metrics_by_n_subtokens": {str(k): v for k, v in ntok_metrics.items()},
        "config": {
            "image_attention_layers": list(config.image_attention_layers),
            "image_attention_threshold": config.image_attention_threshold,
            "semantic_matcher": config.semantic_matcher,
            "topk_patches": config.topk_patches,
        },
        "multi_fp_examples": [
            {"word": r["word"], "subtokens": r["subtoken_strings"],
             "att": round(r["attention_first_subtoken"], 4),
             "right_num": r["right_num"],
             "ll_sample": list(r["logitlens_per_layer"].values())[:2]}
            for r in multi_fp[:10]
        ],
        "multi_fn_examples": [
            {"word": r["word"], "subtokens": r["subtoken_strings"],
             "att": round(r["attention_first_subtoken"], 4),
             "right_num": r["right_num"]}
            for r in multi_fn[:10]
        ],
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Multi-token audit ===")
    print(f"Total words: {len(all_rows)}  mono: {len(mono)}  multi: {len(multi)}")
    print(f"{'':30s}  P       R       F1      n")
    print(f"{'All':30s}  {detection_metrics(all_rows)['precision']:.3f}   {detection_metrics(all_rows)['recall']:.3f}   {detection_metrics(all_rows)['f1']:.3f}   {len(all_rows)}")
    print(f"{'Mono-token':30s}  {metrics_mono['precision']:.3f}   {metrics_mono['recall']:.3f}   {metrics_mono['f1']:.3f}   {len(mono)}")
    print(f"{'Multi-token':30s}  {metrics_multi['precision']:.3f}   {metrics_multi['recall']:.3f}   {metrics_multi['f1']:.3f}   {len(multi)}")
    for n, m in sorted(ntok_metrics.items()):
        print(f"  {n}-subtoken:                    {m['precision']:.3f}   {m['recall']:.3f}   {m['f1']:.3f}   {m['n']}")
    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
