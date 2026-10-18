"""Exp 6 — Extended LLCC: mask consistent words and flag persisters as Type 2.

Standard LLCC flags a word as hallucinated when the logit-lens is inconsistent
(right_num == 0 → Type 1).  This experiment extends the detector: for words where
the logit-lens IS consistent (right_num > 0, currently not flagged), we mask their
high-attention patches and regenerate.  If the word persists in the new caption it
has no visual anchor → declare it as a Type 2 (contextual-prior) hallucination.

Combined detector = Type 1 (LLCC) ∪ Type 2 (mask-persists).

Three detectors are compared:
  paper_llcc     — pre-stored paper detections (upper bound reference)
  baseline_llcc  — standard LLCC (right_num == 0 only)
  exp6_combined  — LLCC + Type 2 masking extension

One extra forward pass is made per image (mask all consistent words together).

Outputs
-------
outputs/exp6_type2_masking/results_paper.json
outputs/exp6_type2_masking/results_baseline.json
outputs/exp6_type2_masking/results_exp6.json
outputs/exp6_type2_masking/summary.json
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
    p.add_argument("--output-dir", default="outputs/exp6_type2_masking")
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
    exp6_records:     List[Dict] = []

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

        # --- Generate caption ---
        output, input_ids, response = generate_with_trace(
            model, tokenizer, image_processor, model_name, image_file, args.prompt,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        )

        # --- CHAIR on our caption ---
        our_chair = chair_words(chair_eval, image_name, response)
        if our_chair:
            our_gt_hallu = [w for w in our_chair["hallucination_words"] if w not in config.excluded_words]
            object_words = [w for w in our_chair["generated_words"]     if w not in config.excluded_words]
        else:
            our_gt_hallu = gt_hallu
            object_words = ([w for w in (label_item.get("generated_words") or [])
                             if w not in config.excluded_words] if label_item else [])

        # --- LLCC detection ---
        detections = detect_hallucinations(
            words=object_words, output=output, input_ids=input_ids,
            model=model, tokenizer=tokenizer, config=config,
        )
        type1_detected = [d.word for d in detections if d.is_hallucination]

        baseline_records.append({
            "img_id": image_name,
            "hallucination_words": sorted(set(our_gt_hallu)),
            "detected_hallucinations": type1_detected,
        })

        # --- Exp 6: for logit-lens-consistent words, mask patches and check persistence ---
        # consistent_dets: object tokens where logit-lens agreed (right_num > 0), not flagged by LLCC
        consistent_dets = [d for d in detections if d.is_object_token and not d.is_hallucination]
        type2_detected: List[str] = []

        if consistent_dets:
            consistent_words = [d.word for d in consistent_dets]
            original_image_sizes = [Image.open(image_file).size]

            masked_image = mask_only_words(
                image_file=image_file,
                detections=detections,
                words=consistent_words,
                config=config,
                image_mean=getattr(image_processor, "image_mean", None),
            )

            if masked_image is not None:
                masked_tensor, _ = prepare_vcd_image_tensor(
                    masked_image, image_processor, model, image_sizes=original_image_sizes,
                )
                _, _, regen_response = generate_with_trace(
                    model, tokenizer, image_processor, model_name, image_file, args.prompt,
                    image_tensor=masked_tensor, image_sizes=original_image_sizes,
                    max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                )
                regen_chair = chair_words(chair_eval, image_name, regen_response)
                regen_words = set(regen_chair["generated_words"]) if regen_chair else set()
                # Word persists after its patches are masked → no visual anchor → Type 2
                type2_detected = [w for w in consistent_words if w in regen_words]

        combined_detected = sorted(set(type1_detected) | set(type2_detected))
        exp6_records.append({
            "img_id": image_name,
            "hallucination_words": sorted(set(our_gt_hallu)),
            "detected_hallucinations": combined_detected,
            "type1_detected": type1_detected,
            "type2_detected": type2_detected,
        })

        print(f"{image_name}: gt={sorted(set(our_gt_hallu))} "
              f"type1={type1_detected} type2={type2_detected} combined={combined_detected}")

    paper_metrics    = detection_metrics(paper_records)
    baseline_metrics = detection_metrics(baseline_records)
    exp6_metrics     = detection_metrics(exp6_records)

    summary = {
        "paper_llcc":    paper_metrics,
        "baseline_llcc": baseline_metrics,
        "exp6_combined": exp6_metrics,
        "delta_f1_baseline_vs_paper": baseline_metrics["f1"] - paper_metrics["f1"],
        "delta_f1_exp6_vs_baseline":  exp6_metrics["f1"] - baseline_metrics["f1"],
        "delta_f1_exp6_vs_paper":     exp6_metrics["f1"] - paper_metrics["f1"],
        "n_images": len(images),
    }

    save_json_output(str(out_dir / "results_paper.json"),    {"records": paper_records})
    save_json_output(str(out_dir / "results_baseline.json"), {"records": baseline_records})
    save_json_output(str(out_dir / "results_exp6.json"),     {"records": exp6_records})
    save_json_output(str(out_dir / "summary.json"), summary)

    print("\n=== Exp 6 — Extended LLCC (Type 2 masking) ===")
    print(f"Paper:    P={paper_metrics['precision']:.4f}  R={paper_metrics['recall']:.4f}  F1={paper_metrics['f1']:.4f}")
    print(f"Baseline: P={baseline_metrics['precision']:.4f}  R={baseline_metrics['recall']:.4f}  F1={baseline_metrics['f1']:.4f}  "
          f"(ΔF1 vs paper: {summary['delta_f1_baseline_vs_paper']:+.4f})")
    print(f"Exp6:     P={exp6_metrics['precision']:.4f}  R={exp6_metrics['recall']:.4f}  F1={exp6_metrics['f1']:.4f}  "
          f"(ΔF1 vs baseline: {summary['delta_f1_exp6_vs_baseline']:+.4f}  "
          f"ΔF1 vs paper: {summary['delta_f1_exp6_vs_paper']:+.4f})")
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
