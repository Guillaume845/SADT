"""Exp 1 — Sink-corrected LLCC detection.

Hypothesis: replacing raw attention patch-selection with sink-corrected scores
  α_{tj} / μ_j  (where μ_j = mean attention received by patch j)
improves LLCC detection F1 by reducing the influence of attention-sink patches
that dominate raw attention but carry no object-specific visual information.

Outputs
-------
outputs/exp1_sink_corrected/results.json   — per-image detection records
outputs/exp1_sink_corrected/summary.json   — baseline vs corrected metrics table
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
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
    find_token_spans,
    generate_with_trace,
    is_semantically_consistent,
)
from tools.sadt.pipeline_utils import (
    chair_words,
    collect_images,
    load_chair_evaluator,
    load_labels,
    parse_layers,
    save_json_output,
)
from experiments.utils import (
    compute_sink_weights,
    default_config,
    get_img_idx,
    get_image_token_count,
    load_model,
)


# ---------------------------------------------------------------------------
# Sink-corrected variant of detect_hallucinations
# ---------------------------------------------------------------------------

def _sink_corrected_top_patches(
    layer_att: np.ndarray,  # [n_patches] for a single layer
    mu: np.ndarray,          # [n_patches] sink weights
    topk: int,
) -> np.ndarray:
    corrected = layer_att / (mu + 1e-8)
    k = min(topk, corrected.size)
    return np.argsort(corrected)[::-1][:k]


def detect_hallucinations_sink_corrected(
    *,
    words: Sequence[str],
    output: Any,
    input_ids: Any,
    model: Any,
    tokenizer: Any,
    config: SADTConfig,
    mu: np.ndarray,
) -> List[SADTDetection]:
    """detect_hallucinations with sink-corrected patch selection per layer."""
    attentions = output["attentions"]
    hidden_states = output["hidden_states"][0]
    generated_ids = _response_ids(output, input_ids)
    img_idx = get_img_idx(input_ids)
    detections: List[SADTDetection] = []

    for raw_word in words:
        word = raw_word.lower().strip()
        if not word or word in config.excluded_words:
            continue
        token_ids = tokenizer(word, add_special_tokens=False)["input_ids"]
        token_indices = find_token_spans(token_ids, generated_ids)
        if not token_indices:
            plural = _plural(word)
            token_ids = tokenizer(plural, add_special_tokens=False)["input_ids"] if plural else token_ids
            token_indices = find_token_spans(token_ids, generated_ids)
        if not token_indices:
            continue

        token_idx = token_indices[0]
        attention_img = _attention_for_token(attentions, token_idx, img_idx, config.image_token_count)
        avg_attention = float(attention_img[list(config.image_attention_layers)].sum(axis=1).mean())
        is_object = avg_attention > config.image_attention_threshold

        # Top patches using sink-corrected scores
        agg = attention_img[list(config.image_attention_layers)].sum(axis=0)
        corrected_agg = agg / (mu + 1e-8)
        k = min(config.topk_patches, corrected_agg.size)
        top_patch_indices = [int(x) for x in np.argsort(corrected_agg)[::-1][:k]]

        visual_tokens_by_layer: Dict[int, List[str]] = {}
        visual_token_ids_by_layer: Dict[int, List[int]] = {}
        right_num = 0
        veed_token = None
        veed_token_id = None
        best_patch_idx = top_patch_indices[0] if top_patch_indices else None

        import torch
        with torch.no_grad():
            for layer in config.image_attention_layers:
                layer_att = attention_img[layer]
                layer_patch_indices = _sink_corrected_top_patches(layer_att, mu, config.topk_patches)
                absolute_indices = torch.as_tensor(layer_patch_indices + img_idx, device=model.device)
                layer_index = max(0, min(layer, len(hidden_states) - 1))
                visual_hidden = hidden_states[layer_index][:, absolute_indices, :].to(model.device)
                visual_logits = model.lm_head(visual_hidden)
                token_ids_tensor = torch.argmax(visual_logits, dim=-1).squeeze(0)
                token_ids_list = [int(x) for x in token_ids_tensor.detach().cpu().tolist()]
                visual_tokens = [tokenizer.decode([tid], skip_special_tokens=False) for tid in token_ids_list]
                visual_tokens_by_layer[int(layer)] = visual_tokens
                visual_token_ids_by_layer[int(layer)] = token_ids_list
                right_num += sum(is_semantically_consistent(word, t, config) for t in visual_tokens)

                if best_patch_idx is not None and int(best_patch_idx) in [int(x) for x in layer_patch_indices]:
                    local = [int(x) for x in layer_patch_indices].index(int(best_patch_idx))
                    veed_token = visual_tokens[local]
                    veed_token_id = token_ids_list[local]

        detections.append(SADTDetection(
            word=word,
            token_indices=token_indices,
            avg_image_attention=avg_attention,
            is_object_token=is_object,
            is_hallucination=is_object and right_num == 0,
            right_num=right_num,
            top_patch_indices=top_patch_indices,
            visual_tokens_by_layer=visual_tokens_by_layer,
            visual_token_ids_by_layer=visual_token_ids_by_layer,
            veed_token=veed_token,
            veed_token_id=veed_token_id,
        ))
    return detections


# ---------------------------------------------------------------------------
# Detection metrics
# ---------------------------------------------------------------------------

def detection_metrics(records: List[Dict]) -> Dict[str, float]:
    tp = fp = fn = 0
    for r in records:
        true_hallu = set(r["hallucination_words"])
        pred_hallu = set(r["detected_hallucinations"])
        tp += len(true_hallu & pred_hallu)
        fp += len(pred_hallu - true_hallu)
        fn += len(true_hallu - pred_hallu)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


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
    p.add_argument("--chair-cache", default=None)
    p.add_argument("--image-attention-layers", default="19:25")
    p.add_argument("--image-attention-threshold", type=float, default=0.0)
    p.add_argument("--topk-patches", type=int, default=3)
    p.add_argument("--semantic-matcher", default="predefined")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--output-dir", default="outputs/exp1_sink_corrected")
    p.add_argument("--use-paper-words", action="store_true", default=False,
                   help="Use pre-stored generated_words/hallucination_words from labels JSON "
                        "instead of re-running CHAIR on our caption.")
    p.add_argument("--offset", type=int, default=0,
                   help="Skip the first N images (for splitting a run into multiple jobs).")
    return p


def main():
    args = build_parser().parse_args()
    tokenizer, model, image_processor, model_name = load_model(args.model_path, args.model_base)

    config = default_config(
        image_attention_layers=parse_layers(args.image_attention_layers),
        topk_patches=args.topk_patches,
        image_attention_threshold=args.image_attention_threshold,
        semantic_matcher=args.semantic_matcher,
    )

    labels = load_labels(args.labels_json)
    chair_evaluator = load_chair_evaluator(args.chair_cache, None)

    images = collect_images(args.image_file, args.image_dir, args.limit, offset=args.offset)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paper_records: List[Dict] = []
    baseline_records: List[Dict] = []
    corrected_records: List[Dict] = []

    from tools.sadt.core import detect_hallucinations as _baseline_detect

    for image_file in tqdm(images):
        image_name = Path(image_file).name
        label_item = labels.get(image_name)

        # --- Paper baseline: detections stored directly in labels ---
        gt_hallu = []
        paper_det = []
        if label_item:
            gt_hallu = [w for w in (label_item.get("hallucination_words") or [])
                        if w not in config.excluded_words]
            paper_det = [w for w in (label_item.get("detected_hallucinations") or [])
                         if w not in config.excluded_words]
        paper_records.append({
            "img_id": image_name,
            "hallucination_words": sorted(set(gt_hallu)),
            "detected_hallucinations": paper_det,
        })

        output, input_ids, response = generate_with_trace(
            model, tokenizer, image_processor, model_name, image_file, args.prompt,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        )

        img_idx = get_img_idx(input_ids)

        if args.use_paper_words:
            # Use the paper's pre-stored word lists so we evaluate on the same input.
            our_gt_hallu = gt_hallu
            object_words = [w for w in (label_item.get("generated_words") or [])
                            if w not in config.excluded_words] if label_item else []
        else:
            # Run CHAIR on our own caption to get our own generated_words and gt.
            our_chair = chair_words(chair_evaluator, image_name, response)
            if our_chair:
                our_gt_hallu = [w for w in our_chair["hallucination_words"]
                                if w not in config.excluded_words]
                object_words = [w for w in our_chair["generated_words"]
                                if w not in config.excluded_words]
            else:
                our_gt_hallu = gt_hallu
                object_words = [w for w in (label_item.get("generated_words") or [])
                                if w not in config.excluded_words] if label_item else []

        # --- Baseline LLCC (our re-run, our own word list + our own GT) ---
        baseline_dets = _baseline_detect(
            words=object_words, output=output, input_ids=input_ids,
            model=model, tokenizer=tokenizer, config=config,
        )
        baseline_records.append({
            "img_id": image_name,
            "hallucination_words": sorted(set(our_gt_hallu)),
            "detected_hallucinations": [d.word for d in baseline_dets if d.is_hallucination],
        })

        # --- Sink-corrected LLCC ---
        mu = compute_sink_weights(output, img_idx, config.image_token_count, config.image_attention_layers)
        corrected_dets = detect_hallucinations_sink_corrected(
            words=object_words, output=output, input_ids=input_ids,
            model=model, tokenizer=tokenizer, config=config, mu=mu,
        )
        corrected_records.append({
            "img_id": image_name,
            "hallucination_words": sorted(set(our_gt_hallu)),
            "detected_hallucinations": [d.word for d in corrected_dets if d.is_hallucination],
        })

        print(f"{image_name}: gt={sorted(set(gt_hallu))} "
              f"paper={paper_det} "
              f"baseline={baseline_records[-1]['detected_hallucinations']} "
              f"corrected={corrected_records[-1]['detected_hallucinations']}")

    paper_metrics = detection_metrics(paper_records)
    baseline_metrics = detection_metrics(baseline_records)
    corrected_metrics = detection_metrics(corrected_records)

    summary = {
        "paper_llcc": paper_metrics,
        "baseline_llcc": baseline_metrics,
        "sink_corrected_llcc": corrected_metrics,
        "delta_f1_vs_paper": baseline_metrics["f1"] - paper_metrics["f1"],
        "delta_f1_sink_vs_baseline": corrected_metrics["f1"] - baseline_metrics["f1"],
        "n_images": len(images),
    }

    save_json_output(str(out_dir / "results_paper.json"), {"records": paper_records})
    save_json_output(str(out_dir / "results_baseline.json"), {"records": baseline_records})
    save_json_output(str(out_dir / "results_corrected.json"), {"records": corrected_records})
    save_json_output(str(out_dir / "summary.json"), summary)

    print("\n=== Exp 1 — Sink-corrected LLCC ===")
    print(f"Paper:     P={paper_metrics['precision']:.4f}  R={paper_metrics['recall']:.4f}  F1={paper_metrics['f1']:.4f}")
    print(f"Baseline:  P={baseline_metrics['precision']:.4f}  R={baseline_metrics['recall']:.4f}  F1={baseline_metrics['f1']:.4f}  (ΔF1 vs paper: {summary['delta_f1_vs_paper']:+.4f})")
    print(f"Corrected: P={corrected_metrics['precision']:.4f}  R={corrected_metrics['recall']:.4f}  F1={corrected_metrics['f1']:.4f}  (ΔF1 vs baseline: {summary['delta_f1_sink_vs_baseline']:+.4f})")
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
