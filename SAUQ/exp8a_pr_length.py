"""
Exp 8a — Correlation between Participation Ratio (PR / SAUQ) and output length.

Same generation pipeline as rebuild_samsum, but the analysis measures:
  Spearman ρ(uncertainty_metric, n_tokens)  for SAUQ, RAUQ, NLP, entropy.

Question: do more uncertain summaries tend to be longer or shorter?

Output
------
outputs/exp8a_pr_length/
  results.json   — per-sample metrics (same fields as rebuild_samsum + n_tokens)
  summary.json   — PRR + Spearman ρ vs ROUGE-L + Spearman ρ vs n_tokens
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

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
    p.add_argument("--output-dir", default="outputs/exp8a_pr_length")
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
    n_heads = model.config.num_attention_heads
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
              f"NLP={nlp_val:.4f}  Ent={ent_val:.4f}  "
              f"n_tokens={n_tok}  ROUGE-L={rl:.4f}")

        results.append({
            "id": sid, "rouge_l": rl,
            "sauq": sauq_val, "rauq": rauq_val,
            "nlp": nlp_val, "mean_token_entropy": ent_val,
            "n_tokens": n_tok,
        })

        with open(out_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    if not results:
        print("No results.")
        return

    rouge_vals  = np.array([r["rouge_l"]   for r in results])
    length_vals = np.array([r["n_tokens"]  for r in results])
    methods = ["sauq", "rauq", "nlp", "mean_token_entropy"]

    from scipy.stats import spearmanr
    summary: Dict = {
        "n_samples": len(results),
        "model": args.model_path,
        "rouge_l_mean": float(rouge_vals.mean()),
        "rouge_l_std":  float(rouge_vals.std()),
        "length_mean": float(length_vals.mean()),
        "length_std":  float(length_vals.std()),
    }

    print(f"\n=== Exp 8a — PR vs length (n={len(results)}) ===")
    print(f"  ROUGE-L mean={rouge_vals.mean():.4f}  "
          f"length mean={length_vals.mean():.1f}")
    print(f"\n{'Method':<22} {'PRR':>8} {'ρ vs ROUGE-L':>14} {'ρ vs length':>12} {'p(length)':>10}")

    for method in methods:
        vals = np.array([r[method] for r in results])
        prr  = compute_prr(vals, rouge_vals)
        rho_q,  p_q  = spearmanr(-vals, rouge_vals)
        rho_len, p_len = spearmanr(vals, length_vals)
        summary[method] = {
            "prr": prr,
            "spearman_rho_vs_rouge": float(rho_q),
            "spearman_p_vs_rouge":   float(p_q),
            "spearman_rho_vs_length": float(rho_len),
            "spearman_p_vs_length":   float(p_len),
        }
        print(f"  {method:<20} {prr:>8.4f} {rho_q:>14.4f} {rho_len:>12.4f} {p_len:>10.4f}")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
