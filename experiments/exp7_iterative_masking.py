"""Exp 7 — Iterative LLCC: re-verify after each masked regeneration (max 5 rounds).

After detecting hallucinations with LLCC, mask their patches and regenerate.  Then
re-apply LLCC on the new caption.  If new hallucinations appear, mask and regenerate
again.  Repeat up to 5 times per image.

The cumulative detected set (union across all iterations) is compared against GT to
measure detection F1.  The final regenerated caption's CHAIR score measures mitigation.

Outputs
-------
outputs/exp7_iterative_masking/results_paper.json
outputs/exp7_iterative_masking/results_baseline.json
outputs/exp7_iterative_masking/results_exp7.json
outputs/exp7_iterative_masking/summary.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.sadt.core import detect_hallucinations, generate_with_trace
from tools.sadt.pipeline_utils import (
    chair_words, collect_images, load_chair_evaluator, load_labels,
    mask_only_words, parse_layers, prepare_vcd_image_tensor, save_json_output,
)
from experiments.utils import default_config, load_model

MAX_REGEN = 5


def detection_metrics(records: List[Dict]) -> Dict[str, Any]:
    tp = fp = fn = 0
    for r in records:
        gt  = set(r["hallucination_words"])
        det = set(r["detected_hallucinations"])
        tp += len(gt & det)
        fp += len(det - gt)
        fn += len(gt - det)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def aggregate_chair(records: List[Dict], field: str) -> Dict[str, float]:
    n = len(records)
    if not n:
        return {"CHAIRs": float("nan"), "CHAIRi": float("nan")}
    total_gen = total_hallu = hallu_sents = 0
    for r in records:
        c = r.get(field) or {}
        gen   = len(set(c.get("generated_words") or []))
        hallu = len(set(c.get("hallucination_words") or []))
        total_gen   += gen
        total_hallu += hallu
        hallu_sents += int(hallu > 0)
    return {
        "CHAIRs": hallu_sents / n,
        "CHAIRi": total_hallu / total_gen if total_gen else 0.0,
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
    p.add_argument("--semantic-matcher", default="coco")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--output-dir", default="outputs/exp7_iterative_masking")
    p.add_argument("--offset", type=int, default=0)
    return p


def main():
    args = build_parser().parse_args()
    tokenizer, model, image_processor, model_name = load_model(args.model_path, args.model_base)
    config = default_config(
        image_attention_layers=parse_layers(args.image_attention_layers),
        topk_patches=args.topk_patches,
        semantic_matcher=args.semantic_matcher,
    )

    labels     = load_labels(args.labels_json)
    chair_eval = load_chair_evaluator(args.chair_cache, None)
    if chair_eval is None:
        raise ValueError("CHAIR evaluator required.")

    images  = collect_images(args.image_file, args.image_dir, args.limit, offset=args.offset)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paper_records:    List[Dict] = []
    baseline_records: List[Dict] = []
    exp7_records:     List[Dict] = []

    for image_file in tqdm(images):
        image_name = Path(image_file).name
        label_item = labels.get(image_name)

        # --- Paper baseline (pre-stored detections) ---
        gt_hallu  = []
        paper_det = []
        if label_item:
            gt_hallu  = [w for w in (label_item.get("hallucination_words") or [])
                         if w not in config.excluded_words]
            paper_det = [w for w in (label_item.get("detected_hallucinations") or [])
                         if w not in config.excluded_words]
        paper_records.append({
            "img_id": image_name,
            "hallucination_words": sorted(set(gt_hallu)),
            "detected_hallucinations": paper_det,
        })

        # --- Initial generation ---
        output, input_ids, response = generate_with_trace(
            model, tokenizer, image_processor, model_name, image_file, args.prompt,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        )

        # --- CHAIR on original caption (GT anchor stays fixed throughout) ---
        orig_chair = chair_words(chair_eval, image_name, response)
        if orig_chair:
            our_gt_hallu = [w for w in orig_chair["hallucination_words"] if w not in config.excluded_words]
            object_words = [w for w in orig_chair["generated_words"]     if w not in config.excluded_words]
        else:
            our_gt_hallu = gt_hallu
            object_words = ([w for w in (label_item.get("generated_words") or [])
                             if w not in config.excluded_words] if label_item else [])

        # --- Baseline LLCC (iteration 0 only) ---
        baseline_dets = detect_hallucinations(
            words=object_words, output=output, input_ids=input_ids,
            model=model, tokenizer=tokenizer, config=config,
        )
        baseline_detected = [d.word for d in baseline_dets if d.is_hallucination]
        baseline_records.append({
            "img_id": image_name,
            "hallucination_words": sorted(set(our_gt_hallu)),
            "detected_hallucinations": baseline_detected,
        })

        # --- Exp 7: iterative LLCC + mask + regen (up to MAX_REGEN times) ---
        all_detected: set = set(baseline_detected)
        current_output    = output
        current_input_ids = input_ids
        current_response  = response
        current_dets      = baseline_dets
        current_words     = object_words
        n_regen           = 0
        per_iter          = []

        for iteration in range(MAX_REGEN):
            detected_this_iter = [d.word for d in current_dets if d.is_hallucination]
            per_iter.append({
                "iteration": iteration,
                "detected": detected_this_iter,
                "caption": current_response,
            })

            if not detected_this_iter:
                break

            # Mask detected words' patches and regenerate
            original_image_sizes = [Image.open(image_file).size]
            masked_image = mask_only_words(
                image_file=image_file,
                detections=current_dets,
                words=detected_this_iter,
                config=config,
                image_mean=getattr(image_processor, "image_mean", None),
            )

            if masked_image is None:
                break

            masked_tensor, _ = prepare_vcd_image_tensor(
                masked_image, image_processor, model, image_sizes=original_image_sizes,
            )
            regen_output, regen_input_ids, regen_response = generate_with_trace(
                model, tokenizer, image_processor, model_name, image_file, args.prompt,
                image_tensor=masked_tensor, image_sizes=original_image_sizes,
                max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            )
            n_regen += 1

            # Re-run LLCC on new caption
            regen_chair = chair_words(chair_eval, image_name, regen_response)
            regen_words = ([w for w in regen_chair["generated_words"] if w not in config.excluded_words]
                           if regen_chair else [])
            regen_dets = detect_hallucinations(
                words=regen_words, output=regen_output, input_ids=regen_input_ids,
                model=model, tokenizer=tokenizer, config=config,
            )

            new_detected = [d.word for d in regen_dets if d.is_hallucination]
            all_detected |= set(new_detected)

            current_output    = regen_output
            current_input_ids = regen_input_ids
            current_response  = regen_response
            current_dets      = regen_dets
            current_words     = regen_words

        # Final caption chair (after iterative regen)
        final_chair = chair_words(chair_eval, image_name, current_response) or {}

        exp7_records.append({
            "img_id": image_name,
            "hallucination_words": sorted(set(our_gt_hallu)),
            "detected_hallucinations": sorted(all_detected),
            "n_regen": n_regen,
            "per_iter": per_iter,
            "final_caption": current_response,
            "final_chair": {
                "generated_words":    final_chair.get("generated_words", []),
                "hallucination_words": final_chair.get("hallucination_words", []),
            },
        })

        print(f"{image_name}: gt={sorted(set(our_gt_hallu))} "
              f"baseline={baseline_detected} exp7={sorted(all_detected)} n_regen={n_regen}")

    paper_metrics    = detection_metrics(paper_records)
    baseline_metrics = detection_metrics(baseline_records)
    exp7_metrics     = detection_metrics(exp7_records)

    summary = {
        "paper_llcc":    paper_metrics,
        "baseline_llcc": baseline_metrics,
        "exp7_combined": exp7_metrics,
        "delta_f1_baseline_vs_paper": baseline_metrics["f1"] - paper_metrics["f1"],
        "delta_f1_exp7_vs_baseline":  exp7_metrics["f1"] - baseline_metrics["f1"],
        "delta_f1_exp7_vs_paper":     exp7_metrics["f1"] - paper_metrics["f1"],
        "avg_n_regen": sum(r["n_regen"] for r in exp7_records) / len(exp7_records) if exp7_records else 0.0,
        "final_caption_chair": aggregate_chair(exp7_records, "final_chair"),
        "n_images": len(images),
    }

    save_json_output(str(out_dir / "results_paper.json"),    {"records": paper_records})
    save_json_output(str(out_dir / "results_baseline.json"), {"records": baseline_records})
    save_json_output(str(out_dir / "results_exp7.json"),     {"records": exp7_records})
    save_json_output(str(out_dir / "summary.json"), summary)

    print("\n=== Exp 7 — Iterative LLCC (max 5 regen) ===")
    print(f"Paper:    P={paper_metrics['precision']:.4f}  R={paper_metrics['recall']:.4f}  F1={paper_metrics['f1']:.4f}")
    print(f"Baseline: P={baseline_metrics['precision']:.4f}  R={baseline_metrics['recall']:.4f}  F1={baseline_metrics['f1']:.4f}  "
          f"(ΔF1 vs paper: {summary['delta_f1_baseline_vs_paper']:+.4f})")
    print(f"Exp7:     P={exp7_metrics['precision']:.4f}  R={exp7_metrics['recall']:.4f}  F1={exp7_metrics['f1']:.4f}  "
          f"(ΔF1 vs baseline: {summary['delta_f1_exp7_vs_baseline']:+.4f}  "
          f"ΔF1 vs paper: {summary['delta_f1_exp7_vs_paper']:+.4f})")
    print(f"Avg regen per image: {summary['avg_n_regen']:.2f}")
    fc = summary["final_caption_chair"]
    print(f"Final caption CHAIR: CHAIRs={fc['CHAIRs']:.4f}  CHAIRi={fc['CHAIRi']:.4f}")
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
