"""Exp 4 — Predict hallucination type without mask-and-regenerate.

The expensive SADT classification step (HARM: mask top patches, regenerate,
check if hallucination persists) is used as ground truth.  We then test whether
three cheap spectral features extracted from the *original* generation trace can
predict Type 1 (visual_uncertainty) vs Type 2 (contextual_prior) without any
additional inference.

Features per detected hallucination
-------------------------------------
1. entropy     — mean entropy of Logit Lens softmax at top patch over layers
                 High → ambiguous visual region (Type 1 candidate)
2. conc        — max(α) / mean(α) over patches (attention concentration)
                 High → one patch dominates (Type 1 candidate)
3. sink_ratio  — corrected score of raw-top patch / max corrected score
                 < 1 → top patch is a sink (Type 2 candidate: attention spread)

Ground truth
------------
Runs HARM inline: mask detected hallucinations → regenerate → CHAIR check.
  Type 1: hallucination disappears after masking  (visual evidence removed)
  Type 2: hallucination persists after masking    (contextual prior drives it)

Classifier
----------
Logistic regression with leave-one-out cross-validation (or train/test split).

Outputs
-------
outputs/exp4_type_classification/samples.json   — per-hallucination features + labels
outputs/exp4_type_classification/summary.json   — classification accuracy, F1, AUC
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from tools.sadt.core import (
    SADTConfig,
    SADTDetection,
    generate_with_trace,
    detect_hallucinations,
    filter_words_by_image_attention,
    mask_high_attention_regions_vcd,
)
from tools.sadt.pipeline_utils import (
    chair_words,
    choose_words,
    collect_images,
    load_chair_evaluator,
    load_labels,
    mask_only_words,
    parse_layers,
    prepare_vcd_image_tensor,
    save_json_output,
)
from experiments.utils import (
    compute_attention_concentration,
    compute_full_attention_matrix,
    compute_logit_lens_entropy,
    compute_sink_weights,
    compute_sink_ratio,
    default_config,
    get_img_idx,
    load_model,
)


# ---------------------------------------------------------------------------
# Ground truth: classify via HARM (mask + regen + CHAIR check)
# ---------------------------------------------------------------------------

def classify_via_harm(
    *,
    image_file: str,
    detected_hallucinations: List[str],
    detections: List[SADTDetection],
    original_response: str,
    image_processor: Any,
    model: Any,
    tokenizer: Any,
    model_name: str,
    config: SADTConfig,
    chair_evaluator: Any,
    image_name: str,
    max_new_tokens: int,
    temperature: float,
) -> Dict[str, str]:
    """Return {word: 'type1'|'type2'} for each detected hallucination."""
    if not detected_hallucinations:
        return {}

    original_image_sizes = [Image.open(image_file).size]

    # Create HARM masked image for all detected hallucinations
    masked_image = mask_only_words(
        image_file=image_file,
        detections=detections,
        words=detected_hallucinations,
        config=config,
        image_mean=getattr(image_processor, "image_mean", None),
    )
    if masked_image is None:
        return {w: "type1" for w in detected_hallucinations}

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

    types: Dict[str, str] = {}
    for word in detected_hallucinations:
        types[word] = "type2" if word in harm_hallu else "type1"
    return types


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(
    *,
    detection: SADTDetection,
    output: Any,
    model: Any,
    img_idx: int,
    image_token_count: int,
    layers: Sequence[int],
    mu: np.ndarray,
) -> Dict[str, float]:
    token_idx = detection.token_indices[0]
    top_patch = detection.top_patch_indices[0] if detection.top_patch_indices else 0

    entropy = compute_logit_lens_entropy(
        output, model, img_idx, token_idx, top_patch, layers,
    )
    conc = compute_attention_concentration(
        output, img_idx, image_token_count, token_idx, layers,
    )
    sink = compute_sink_ratio(
        output, img_idx, image_token_count, token_idx, layers, mu,
    )
    return {"entropy": entropy, "concentration": conc, "sink_ratio": sink}


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def train_and_evaluate(
    samples: List[Dict],
    test_ratio: float = 0.3,
) -> Dict[str, Any]:
    """Logistic regression with a random 70/30 split and a leave-one-out pass."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    feat_keys = ["entropy", "concentration", "sink_ratio"]
    X = np.array([[s["features"][k] for k in feat_keys] for s in samples])
    y = np.array([1 if s["type"] == "type2" else 0 for s in samples])

    if len(X) < 4:
        return {"error": "Too few samples for classification."}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000)

    # Leave-one-out CV
    loo_preds = cross_val_predict(clf, X_scaled, y, cv=LeaveOneOut(), method="predict_proba")[:, 1]
    loo_labels = (loo_preds >= 0.5).astype(int)

    acc = float(accuracy_score(y, loo_labels))
    f1 = float(f1_score(y, loo_labels, zero_division=0))
    try:
        auc = float(roc_auc_score(y, loo_preds))
    except Exception:
        auc = float("nan")

    # Fit on all data for feature importances
    clf.fit(X_scaled, y)
    coefs = {k: float(c) for k, c in zip(feat_keys, clf.coef_[0])}

    return {
        "n_samples": len(samples),
        "n_type1": int((y == 0).sum()),
        "n_type2": int((y == 1).sum()),
        "loo_accuracy": acc,
        "loo_f1": f1,
        "loo_auc": auc,
        "feature_coefs": coefs,
        "baseline_accuracy": float(max(y.mean(), 1 - y.mean())),
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
    p.add_argument("--coco-annotations", default=None)
    p.add_argument("--image-attention-layers", default="19:25")
    p.add_argument("--image-attention-threshold", type=float, default=0.0)
    p.add_argument("--topk-patches", type=int, default=3)
    p.add_argument("--semantic-matcher", default="predefined")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--output-dir", default="outputs/exp4_type_classification")
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
    chair_evaluator = load_chair_evaluator(args.chair_cache, args.coco_annotations)
    if chair_evaluator is None:
        raise ValueError("CHAIR evaluator required. Pass --chair-cache or --coco-annotations.")

    images = collect_images(args.image_file, args.image_dir, args.limit)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_samples: List[Dict] = []

    for image_file in tqdm(images):
        image_name = Path(image_file).name
        label_item = labels.get(image_name)

        # --- Original generation ---
        output, input_ids, response = generate_with_trace(
            model, tokenizer, image_processor, model_name, image_file, args.prompt,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        )

        img_idx = get_img_idx(input_ids)
        mu = compute_sink_weights(output, img_idx, config.image_token_count, config.image_attention_layers)

        # --- LLCC detection ---
        chair_item = chair_words(chair_evaluator, image_name, response)
        word_info = choose_words(explicit_words=None, label_item=label_item, chair_item=chair_item)

        attention_filter = filter_words_by_image_attention(
            words=word_info["generated_words"], output=output,
            input_ids=input_ids, tokenizer=tokenizer, config=config,
        )
        object_words = [w for w in attention_filter["object_words"] if w not in config.excluded_words]

        detections = detect_hallucinations(
            words=object_words, output=output, input_ids=input_ids,
            model=model, tokenizer=tokenizer, config=config,
        )
        detected_hallucinations = [
            d.word for d in detections if d.is_hallucination and d.word not in config.excluded_words
        ]

        if not detected_hallucinations:
            continue

        # --- Ground truth type labels via HARM ---
        type_labels = classify_via_harm(
            image_file=image_file,
            detected_hallucinations=detected_hallucinations,
            detections=detections,
            original_response=response,
            image_processor=image_processor,
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            config=config,
            chair_evaluator=chair_evaluator,
            image_name=image_name,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        # --- Feature extraction (no extra inference) ---
        det_by_word = {d.word: d for d in detections}
        for word, htype in type_labels.items():
            det = det_by_word.get(word)
            if det is None:
                continue
            feats = extract_features(
                detection=det,
                output=output,
                model=model,
                img_idx=img_idx,
                image_token_count=config.image_token_count,
                layers=config.image_attention_layers,
                mu=mu,
            )
            all_samples.append({
                "img_id": image_name,
                "word": word,
                "type": htype,
                "features": feats,
            })
            print(f"  {image_name} [{word}] → {htype}  features={feats}")

    if not all_samples:
        print("No typed hallucinations found. Check --limit or data paths.")
        return

    save_json_output(str(out_dir / "samples.json"), {"samples": all_samples})

    # --- Classification ---
    try:
        metrics = train_and_evaluate(all_samples)
    except ImportError:
        metrics = {"error": "scikit-learn required: pip install scikit-learn"}

    save_json_output(str(out_dir / "summary.json"), metrics)

    print("\n=== Exp 4 — Type Classification ===")
    if "error" in metrics:
        print(f"Error: {metrics['error']}")
    else:
        print(f"Samples: {metrics['n_samples']}  (type1={metrics['n_type1']}, type2={metrics['n_type2']})")
        print(f"LOO accuracy: {metrics['loo_accuracy']:.4f}  (baseline={metrics['baseline_accuracy']:.4f})")
        print(f"LOO F1:       {metrics['loo_f1']:.4f}")
        print(f"LOO AUC:      {metrics['loo_auc']:.4f}")
        print(f"Feature coefs: {metrics['feature_coefs']}")
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
