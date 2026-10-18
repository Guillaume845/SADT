"""
Rebuild 1 — Perturbation sensitivity analysis (replication of Figure 4).

Replicates the controlled copy-task experiment from:
  "Spectral Geometry of Attention: From Information Routing to Uncertainty"
  Viganò, Melzi, Ovsjanikov — preprint 2025.

Task: the model receives a prompt asking it to copy a binary sequence of
length n.  We compare two input families:
  α (easy): all-zeros sequence  0 0 0 ... 0
  β (hard): all-zeros with a single 1 at a random position r

Prediction: as n grows, probability-based metrics (NLP, mean token entropy)
become indistinguishable across α/β, while attention-routing metrics
(RAUQ, SAUQ) remain sensitive to the perturbation.

Output
------
outputs/rebuild_1_copy_task/
  results.json          — per-(n, family, trial) raw metrics
  summary.json          — mean metrics per (n, family)
  figure4_replica.png   — 4-panel plot mirroring Figure 4
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SAUQ.model_utils import (
    load_text_model,
    format_copy_prompt,
    collect_stats,
    nlp,
    mean_token_entropy,
    compute_rauq,
)
from SAUQ.rauq_spectral import SpectralRAUQ


# ---------------------------------------------------------------------------
# Sequence generation
# ---------------------------------------------------------------------------

def make_easy(n: int) -> str:
    """All-zeros sequence."""
    return " ".join(["0"] * n)


def make_hard(n: int, rng: random.Random) -> str:
    """All-zeros with one 1 at a random position."""
    bits = ["0"] * n
    bits[rng.randint(0, n - 1)] = "1"
    return " ".join(bits)


# ---------------------------------------------------------------------------
# SAUQ estimator
# ---------------------------------------------------------------------------

def make_estimator(n_layers: int, n_heads: int) -> SpectralRAUQ:
    return SpectralRAUQ(
        alpha=0.2,
        n_layers=n_layers,
        n_heads=n_heads,
        use_entropy=False,
        mass="attn_mass",
        attn_mode="square",
        rank_estimator="pr",
        selection_mode="square",
        head_selection_weighting="euclidean",
    )


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_one(
    model, tokenizer, estimator, sequence: str, max_new_tokens: int, n_layers: int, n_heads: int
) -> Dict[str, float] | None:
    prompt = format_copy_prompt(sequence)
    stats = collect_stats(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
    if stats is None:
        return None

    sauq_val = float(estimator(stats)[0])
    rauq_val = compute_rauq(
        stats["attention_all"][0],
        stats["greedy_log_likelihoods"][0],
        n_layers, n_heads,
    )
    nlp_val = nlp(stats["greedy_log_likelihoods"][0])
    ent_val = mean_token_entropy(stats["_token_probs"])

    return {
        "sauq": sauq_val,
        "rauq": rauq_val,
        "nlp": nlp_val,
        "mean_entropy": ent_val,
        "n_tokens_gen": int(len(stats["greedy_tokens"][0])),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--load-4bit", action="store_true", default=False)
    p.add_argument(
        "--ns",
        nargs="+",
        type=int,
        default=[10, 20, 30, 50, 75, 100],
        help="Sequence lengths n to evaluate",
    )
    p.add_argument("--trials", type=int, default=3, help="Trials per (n, family)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="outputs/rebuild_1_copy_task")
    return p


def main():
    args = build_parser().parse_args()
    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model …")
    tokenizer, model = load_text_model(args.model_path, load_4bit=args.load_4bit)
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    print(f"  {n_layers} layers × {n_heads} heads")

    estimator = make_estimator(n_layers, n_heads)

    all_rows: List[dict] = []
    metrics = ["nlp", "mean_entropy", "rauq", "sauq"]

    for n in args.ns:
        # Allow ~2n+10 tokens to copy, plus buffer for partial/eos
        max_new = 2 * n + 20

        for family, label in [("easy", "α"), ("hard", "β")]:
            for trial in range(args.trials):
                seq = make_easy(n) if family == "easy" else make_hard(n, rng)
                print(f"  n={n:4d}  {label}={family}  trial={trial}  seq[:30]={seq[:30]} …", end=" ", flush=True)
                row = run_one(model, tokenizer, estimator, seq, max_new, n_layers, n_heads)
                if row is None:
                    print("SKIP")
                    continue
                row.update({"n": n, "family": family, "trial": trial, "sequence": seq})
                all_rows.append(row)
                print(f"SAUQ={row['sauq']:.4f}  NLP={row['nlp']:.4f}  T={row['n_tokens_gen']}")

    if not all_rows:
        print("No results.")
        return

    with open(out_dir / "results.json", "w") as f:
        json.dump(all_rows, f, indent=2)

    # --- Summary: mean per (n, family) ---
    summary: dict = {}
    for n in args.ns:
        summary[str(n)] = {}
        for family in ("easy", "hard"):
            rows = [r for r in all_rows if r["n"] == n and r["family"] == family]
            if not rows:
                continue
            summary[str(n)][family] = {
                m: float(np.mean([r[m] for r in rows])) for m in metrics
            }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # --- Figure 4 replica ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        panel_metrics = ["nlp", "mean_entropy", "rauq", "sauq"]
        panel_labels = ["Negative Log Probability", "Mean Token Entropy", "RAUQ", "Ours (SAUQ)"]
        ns_sorted = sorted(args.ns)

        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        colors = {"easy": "#1f77b4", "hard": "#d62728"}
        family_label = {"easy": "α (easy: 0ⁿ)", "hard": "β (hard: 0…1…0)"}

        for ax, metric, panel_label in zip(axes, panel_metrics, panel_labels):
            for family in ("easy", "hard"):
                ys = []
                for n in ns_sorted:
                    nd = summary.get(str(n), {}).get(family)
                    ys.append(nd[metric] if nd else float("nan"))
                ax.plot(ns_sorted, ys, marker="o", label=family_label[family],
                        color=colors[family], linewidth=2)
            ax.set_xlabel("n (sequence length)")
            ax.set_title(panel_label, fontsize=11)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel("Score")
        plt.suptitle(
            "Perturbation Sensitivity Across Metrics\n"
            "(replication of Figure 4 — Viganò et al. 2025)",
            fontsize=12,
        )
        plt.tight_layout()
        plt.savefig(out_dir / "figure4_replica.png", dpi=150)
        print(f"\nFigure saved to {out_dir}/figure4_replica.png")
    except Exception as e:
        print(f"Plot failed: {e}")

    print(f"\n=== Rebuild 1 — Final summary ===")
    print(f"{'n':>5}  {'family':>6}  {'NLP':>8}  {'Entropy':>8}  {'RAUQ':>8}  {'SAUQ':>8}")
    for n in ns_sorted:
        for family in ("easy", "hard"):
            nd = summary.get(str(n), {}).get(family)
            if nd:
                print(f"{n:>5}  {family:>6}  {nd['nlp']:>8.4f}  {nd['mean_entropy']:>8.4f}  "
                      f"{nd['rauq']:>8.4f}  {nd['sauq']:>8.4f}")
    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
