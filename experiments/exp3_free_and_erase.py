"""Exp 3 — Free-and-Erase: let the model write Type 2 hallucinations, then delete them.

For Type 2 (contextual prior) hallucinations, SADT mask-and-regenerate fails by
construction: the hallucinated word persists after masking because it is driven by
language priors, not by visual evidence.

This experiment tests a complementary strategy:
  1. Generate the full caption without any intervention — let the model write
     whatever it wants, including the Type 2 hallucinated word.
  2. Run HARM (mask + regen + CHAIR) to classify each hallucination as Type 1 or 2.
  3. Post-hoc erase all Type 2 words from the original caption.

Hypothesis: for Type 2 hallucinations, the "free-and-erase" corrected caption has a
lower CHAIR score than the SADT mask-and-regenerate caption, because the latter still
contains the hallucinated word while the former removes it directly.

Three captions are compared per image:
  - original       : raw model output, no correction
  - sadt_regen     : HARM mask-and-regenerate (standard SADT intervention)
  - exp3_erase     : original caption with Type 2 words post-hoc erased

Outputs
-------
outputs/exp3_free_and_erase/per_image.json   — per-image records, all three captions
outputs/exp3_free_and_erase/summary.json     — CHAIRs / CHAIRi for each method
"""

from __future__ import annotations

import argparse
import re
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
    SADTConfig, SADTDetection,
    detect_hallucinations, generate_with_trace,
)
from tools.sadt.pipeline_utils import (
    chair_words, collect_images, load_chair_evaluator, load_labels,
    mask_only_words, parse_layers, prepare_vcd_image_tensor, save_json_output,
)
from experiments.utils import default_config, get_img_idx, load_model

TYPE1, TYPE2 = 1, 2


# ---------------------------------------------------------------------------
# Text erasure
# ---------------------------------------------------------------------------

def erase_words(text: str, words: List[str]) -> str:
    """Remove each word from text (case-insensitive, whole-word match)."""
    for word in words:
        text = re.sub(r'\b' + re.escape(word) + r's?\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\s([,\.;:!?])', r'\1', text)
    return text


# ---------------------------------------------------------------------------
# HARM: mask + regen + classify
# ---------------------------------------------------------------------------

def harm_run(
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
) -> Dict[str, Any]:
    """Run HARM and return {response, hallucination_words, type_labels}."""
    original_image_sizes = [Image.open(image_file).size]

    masked_image = mask_only_words(
        image_file=image_file, detections=detections, words=detected_words,
        config=config, image_mean=getattr(image_processor, "image_mean", None),
    )
    if masked_image is None:
        return {
            "response": "",
            "hallucination_words": list(detected_words),
            "type_labels": {w: TYPE2 for w in detected_words},
        }

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

    type_labels = {w: (TYPE2 if w in harm_hallu else TYPE1) for w in detected_words}
    return {
        "response":          harm_response,
        "hallucination_words": list(harm_hallu),
        "type_labels":       type_labels,
    }


# ---------------------------------------------------------------------------
# CHAIR helpers
# ---------------------------------------------------------------------------

def chair_record(evaluator, image_name, response, excluded_words):
    item = chair_words(evaluator, image_name, response)
    if not item:
        return {"generated_words": [], "hallucination_words": []}
    excl = set(excluded_words)
    gen   = [w for w in (item.get("generated_words") or []) if w not in excl]
    hallu = [w for w in (item.get("hallucination_words") or []) if w not in excl]
    return {"generated_words": gen, "hallucination_words": hallu}


def aggregate_chair(records: List[Dict], field: str) -> Dict[str, float]:
    n = len(records)
    if not n:
        return {"CHAIRs": float("nan"), "CHAIRi": float("nan")}
    total_gen = total_hallu = hallucinated_sentences = 0
    for r in records:
        c = r.get(field, {})
        gen   = len(set(c.get("generated_words") or []))
        hallu = len(set(c.get("hallucination_words") or []))
        total_gen   += gen
        total_hallu += hallu
        hallucinated_sentences += int(hallu > 0)
    return {
        "CHAIRs": hallucinated_sentences / n,
        "CHAIRi": total_hallu / total_gen if total_gen else 0.0,
        "n_hallucinated_images": hallucinated_sentences,
        "n_images": n,
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
    p.add_argument("--output-dir", default="outputs/exp3_free_and_erase")
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

        # --- Step 1: free generation (no intervention) ---
        output, input_ids, original_response = generate_with_trace(
            model, tokenizer, image_processor, model_name, image_file, args.prompt,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        )

        orig_chair = chair_record(chair_eval, image_name, original_response, config.excluded_words)
        generated_words = orig_chair["generated_words"]
        hallu_words     = orig_chair["hallucination_words"]

        # Skip images with no detected hallucinations — nothing to correct
        if not hallu_words:
            per_image.append({
                "img_id":   image_name,
                "original": {**orig_chair, "text": original_response},
                "sadt_regen": {**orig_chair, "text": original_response},
                "exp3_erase": {**orig_chair, "text": original_response},
                "type1_words": [], "type2_words": [],
                "skipped": True,
            })
            continue

        # --- Step 2: LLCC detection + HARM classification ---
        detections = detect_hallucinations(
            words=generated_words, output=output, input_ids=input_ids,
            model=model, tokenizer=tokenizer, config=config,
        )
        detected_hallu = [d.word for d in detections if d.is_hallucination
                          and d.word in set(hallu_words)]

        harm_result: Dict[str, Any] = {"response": original_response,
                                        "hallucination_words": hallu_words,
                                        "type_labels": {w: TYPE2 for w in detected_hallu}}
        if detected_hallu:
            harm_result = harm_run(
                image_file=image_file, image_name=image_name,
                detected_words=detected_hallu, detections=detections,
                image_processor=image_processor, model=model,
                tokenizer=tokenizer, model_name=model_name,
                config=config, chair_evaluator=chair_eval,
                max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            )

        type_labels = harm_result["type_labels"]
        type1_words = [w for w, t in type_labels.items() if t == TYPE1]
        type2_words = [w for w, t in type_labels.items() if t == TYPE2]

        # --- Step 3: SADT regen result (HARM response) ---
        sadt_response = harm_result["response"]
        sadt_chair    = chair_record(chair_eval, image_name, sadt_response, config.excluded_words)

        # --- Step 4: Exp 3 — erase Type 2 words from original caption ---
        erased_response = erase_words(original_response, type2_words)
        erased_chair    = chair_record(chair_eval, image_name, erased_response, config.excluded_words)

        record = {
            "img_id":     image_name,
            "type1_words": type1_words,
            "type2_words": type2_words,
            "original":   {"text": original_response,  **orig_chair},
            "sadt_regen": {"text": sadt_response,       **sadt_chair},
            "exp3_erase": {"text": erased_response,     **erased_chair},
            "skipped":    False,
        }
        per_image.append(record)

        print(f"{image_name}: type1={type1_words} type2={type2_words}")
        print(f"  CHAIR orig={len(orig_chair['hallucination_words'])}  "
              f"sadt={len(sadt_chair['hallucination_words'])}  "
              f"exp3={len(erased_chair['hallucination_words'])}")

    save_json_output(str(out_dir / "per_image.json"), {"records": per_image})

    # --- Summary ---
    all_records = [r for r in per_image if not r.get("skipped")]

    summary = {
        "n_images_total":      len(per_image),
        "n_images_with_hallu": len(all_records),
        "n_images_with_type2": sum(1 for r in all_records if r["type2_words"]),
        "original":   aggregate_chair(all_records, "original"),
        "sadt_regen": aggregate_chair(all_records, "sadt_regen"),
        "exp3_erase": aggregate_chair(all_records, "exp3_erase"),
    }

    # Type 2 images only — the key comparison
    type2_records = [r for r in all_records if r["type2_words"]]
    summary["type2_images_only"] = {
        "n": len(type2_records),
        "original":   aggregate_chair(type2_records, "original"),
        "sadt_regen": aggregate_chair(type2_records, "sadt_regen"),
        "exp3_erase": aggregate_chair(type2_records, "exp3_erase"),
    }

    save_json_output(str(out_dir / "summary.json"), summary)

    print("\n=== Exp 3 — Free-and-Erase ===")
    print(f"N={summary['n_images_total']}  with_hallu={summary['n_images_with_hallu']}  "
          f"with_type2={summary['n_images_with_type2']}")
    for method in ("original", "sadt_regen", "exp3_erase"):
        m = summary[method]
        print(f"  {method:12s}  CHAIRs={m['CHAIRs']:.4f}  CHAIRi={m['CHAIRi']:.4f}")
    print(f"\nType 2 images only (n={summary['type2_images_only']['n']}):")
    for method in ("original", "sadt_regen", "exp3_erase"):
        m = summary["type2_images_only"][method]
        print(f"  {method:12s}  CHAIRs={m['CHAIRs']:.4f}  CHAIRi={m['CHAIRi']:.4f}")
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
