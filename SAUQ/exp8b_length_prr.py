"""
Exp 8b — Output length as hallucination detection criterion (PRR).

Same generation pipeline as rebuild_samsum, but the PRR is computed using
n_tokens (output length) as the sole uncertainty estimator. Both directions
are tested: longer = more uncertain ("long") and shorter = more uncertain
("short", i.e. neg_n_tokens).

Question: does output length alone predict summary quality better than chance?

Output
------
outputs/exp8b_length_prr/
  results.json   — per-sample metrics
  summary.json   — PRR for all methods + length variants
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SAUQ.rauq_spectral import SpectralRAUQ
from SAUQ.model_utils import nlp, mean_token_entropy, compute_rauq
from SAUQ.rebuild_samsum import (
    load_llm, collect_stats, format_prompt, rouge_l, compute_prr, make_sauq,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--load-4bit", action="store_true", default=False)
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--limit", type=int, default=400)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--output-dir", default="outputs/exp8b_length_prr")
    p.add_argument("--hf-cache", default=None)
    return p


def main():
    args = build_parser().parse_args()

    if args.hf_cache:
        os.environ["HF_HOME"] = args.hf_cache
        os.environ["TRANSFORMERS_CACHE"] = args.hf_cache

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SAMSum …")
    from datasets import load_dataset
    dataset = load_dataset("knkarthick/samsum", split="test")
    samples = list(dataset)[args.offset: args.offset + args.limit]
    print(f"  {len(samples)} samples")

    print("Loading model …")
    tokenizer, model = load_llm(args.model_path, load_4bit=args.load_4bit)
    n_layers = model.config.num_hidden_layers
    n_heads  = model.config.num_attention_heads
    print(f"  {n_layers} layers × {n_heads} heads")

    sauq_estimator = make_sauq(n_layers, n_heads)
    results: List[Dict] = []

    for i, sample in enumerate(samples):
        sid = sample.get("id", str(i))
        dialogue, reference = sample["dialogue"], sample["summary"]
        print(f"\n[{i+1}/{len(samples)}] id={sid}")

        prompt = format_prompt(tokenizer, dialogue)
        stats = collect_stats(model, tokenizer, prompt,
                              max_new_tokens=args.max_new_tokens)
        if stats is None:
            print("  → too few tokens, skipping")
            continue

        ll  = stats["greedy_log_likelihoods"][0]
        sq  = stats["attention_all"][0]

        nlp_val  = float(nlp(ll))
        ent_val  = float(mean_token_entropy(stats["_token_probs"]))
        rauq_val = float(compute_rauq(sq, ll, n_layers, n_heads))
        sauq_val = float(sauq_estimator(stats)[0])
        n_tok    = int(len(stats["greedy_tokens"][0]))

        hypothesis = tokenizer.decode(stats["greedy_tokens"][0],
                                      skip_special_tokens=True).strip()
        rl = rouge_l(hypothesis, reference)

        print(f"  SAUQ={sauq_val:.4f}  RAUQ={rauq_val:.4f}  "
              f"NLP={nlp_val:.4f}  n_tokens={n_tok}  ROUGE-L={rl:.4f}")

        results.append({
            "id": sid, "rouge_l": rl,
            "sauq": sauq_val, "rauq": rauq_val,
            "nlp": nlp_val, "mean_token_entropy": ent_val,
            "n_tokens": n_tok,
            "neg_n_tokens": -n_tok,  # shorter = more uncertain direction
        })

        with open(out_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    if not results:
        print("No results.")
        return

    rouge_vals = np.array([r["rouge_l"] for r in results])

    # All methods including both length directions
    methods = ["sauq", "rauq", "nlp", "mean_token_entropy",
               "n_tokens", "neg_n_tokens"]

    from scipy.stats import spearmanr
    summary: Dict = {
        "n_samples": len(results),
        "model": args.model_path,
        "rouge_l_mean": float(rouge_vals.mean()),
        "rouge_l_std":  float(rouge_vals.std()),
    }

    print(f"\n=== Exp 8b — Length as hallucination criterion (n={len(results)}) ===")
    print(f"  ROUGE-L mean={rouge_vals.mean():.4f}  std={rouge_vals.std():.4f}")
    print(f"\n{'Method':<22} {'PRR':>8} {'Spearman ρ':>12} {'p':>8}")

    for method in methods:
        vals = np.array([r[method] for r in results])
        prr  = compute_prr(vals, rouge_vals)
        rho, pval = spearmanr(-vals, rouge_vals)
        summary[method] = {"prr": prr, "spearman_rho": float(rho),
                           "spearman_p": float(pval)}
        print(f"  {method:<20} {prr:>8.4f} {rho:>12.4f} {pval:>8.4f}")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
