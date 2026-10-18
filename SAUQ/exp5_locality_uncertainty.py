"""
Exp 5 — Does question locality correlate with information selection?

Given a fixed passage, we ask local (fact-lookup) and global (synthesis)
questions and measure row-level attention concentration for each answer:

  1. row_sq  = mean Σ_j A_{t,j}²  over all rows t and heads.
               High → concentrated (model selects specific tokens).
               Low  → scattered (non-selection of information).

  2. row_entropy = mean Shannon entropy -Σ_j A_{t,j} log A_{t,j}.
               High → scattered (non-selection).
               Low  → concentrated.

Hypothesis: local questions (one relevant span) → more concentrated rows;
global questions (synthesising the whole passage) → scattered rows.

Output
------
outputs/exp5_locality/
  results.json   — per-question metrics
  summary.json   — group means, Mann-Whitney p-value, Spearman ρ
  figure_exp5.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SAUQ.model_utils import (
    load_text_model,
    format_qa_prompt,
    collect_stats,
    nlp,
    mean_token_entropy,
)


# ---------------------------------------------------------------------------
# Fixed corpus text
# ---------------------------------------------------------------------------

PASSAGE = """
The French Revolution (1789–1799) was a period of radical political and
societal transformation in France that had profound consequences for the
rest of Europe and the world. The crisis began in 1788 when France faced
severe financial difficulties and poor harvests leading to food shortages.
In May 1789, King Louis XVI convened the Estates-General for the first time
since 1614, and the Third Estate declared itself the National Assembly on
17 June 1789. The storming of the Bastille on 14 July 1789 became the
symbolic beginning of the Revolution. In August 1789 the National Assembly
adopted the Declaration of the Rights of Man and of the Citizen, a foundational
document articulating principles of liberty, equality, and popular sovereignty.
The monarchy was abolished in September 1792 and the First French Republic was
proclaimed. Louis XVI was tried for treason and executed by guillotine on
21 January 1793. The Reign of Terror (September 1793 – July 1794), orchestrated
by Maximilien Robespierre and the Committee of Public Safety, saw approximately
17,000 people officially executed. Robespierre himself was arrested and
guillotined on 28 July 1794, ending the Terror. The Revolution concluded with
Napoleon Bonaparte's coup of 18 Brumaire (9 November 1799), which dissolved
the Directory and established the Consulate. The Revolution's long-term legacy
included the spread of nationalism and democratic ideals across Europe, the
permanent weakening of aristocratic privilege, and the transformation of France
from a feudal monarchy into a modern nation-state.
""".strip()

# ---------------------------------------------------------------------------
# Questions — locality label
#   0 = local  (answer lies in one specific text span)
#   1 = global (requires synthesising multiple parts of the text)
# ---------------------------------------------------------------------------

QUESTIONS = [
    # ---- LOCAL (locality=0) ----
    {"id": "L1", "locality": 0,
     "question": "On what exact date was the Bastille stormed according to the text?"},
    {"id": "L2", "locality": 0,
     "question": "Who was executed on 21 January 1793?"},
    {"id": "L3", "locality": 0,
     "question": "What was the name of the document adopted in August 1789 that articulated liberty and equality?"},
    {"id": "L4", "locality": 0,
     "question": "Who led the Committee of Public Safety during the Reign of Terror?"},
    {"id": "L5", "locality": 0,
     "question": "What is the first sentence of the text?"},
    {"id": "L6", "locality": 0,
     "question": "Approximately how many people were officially executed during the Reign of Terror?"},
    # ---- GLOBAL (locality=1) ----
    {"id": "G1", "locality": 1,
     "question": "Please summarise the entire text in three to four sentences, covering all main phases."},
    {"id": "G2", "locality": 1,
     "question": "What were the long-term consequences of the French Revolution described in the text?"},
    {"id": "G3", "locality": 1,
     "question": "How did the French Revolution transform France's political system from beginning to end?"},
    {"id": "G4", "locality": 1,
     "question": "Based on the text, what were the causes and effects of the French Revolution?"},
    {"id": "G5", "locality": 1,
     "question": "How do the early events of the Revolution (1789) connect to its ultimate conclusion (1799)?"},
    {"id": "G6", "locality": 1,
     "question": "What is the central thesis or argument of the text as a whole?"},
]


# ---------------------------------------------------------------------------
# Row-level attention concentration metrics
# ---------------------------------------------------------------------------

def compute_row_sq_sum(sq: np.ndarray) -> float:
    """Mean Σ_j A_{t,j}² over all generation-step rows t and all heads.

    sq: (n_layers*n_heads, T, T) causal gen-only block — rows already softmax.
    High value → concentrated rows (model selects specific tokens).
    Low  value → scattered rows (non-selection of information).
    """
    H, T, _ = sq.shape
    if T < 2:
        return float("nan")
    values = []
    for h in range(H):
        for t in range(1, T):
            row = sq[h, t, :t + 1]
            values.append(float(np.sum(row * row)))
    return float(np.mean(values)) if values else float("nan")


def compute_row_entropy(sq: np.ndarray, eps: float = 1e-10) -> float:
    """Mean Shannon entropy of each attention row over all heads and steps.

    sq: (n_layers*n_heads, T, T) causal gen-only block.
    High value → scattered rows (non-selection of information).
    Low  value → concentrated rows (selective routing).
    """
    H, T, _ = sq.shape
    if T < 2:
        return float("nan")
    values = []
    for h in range(H):
        for t in range(1, T):
            row = np.clip(sq[h, t, :t + 1], eps, 1.0)
            values.append(float(-np.sum(row * np.log(row))))
    return float(np.mean(values)) if values else float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--load-4bit", action="store_true", default=False)
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--output-dir", default="outputs/exp5_locality")
    return p


def main():
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model …")
    tokenizer, model = load_text_model(args.model_path, load_4bit=args.load_4bit)
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    print(f"  {n_layers} layers × {n_heads} heads")

    results = []
    for q in QUESTIONS:
        print(f"\n[{q['id']}] {q['question'][:80]} …")
        prompt = format_qa_prompt(PASSAGE, q["question"])
        stats = collect_stats(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens)
        if stats is None:
            print("  → too few tokens, skipping")
            continue

        sq = stats["attention_all"][0]  # (n_layers*n_heads, T, T)
        ll = stats["greedy_log_likelihoods"][0]

        row_sq   = compute_row_sq_sum(sq)
        row_ent  = compute_row_entropy(sq)
        nlp_score = float(nlp(ll))
        ent_score = mean_token_entropy(stats["_token_probs"])

        gen_text = tokenizer.decode(stats["greedy_tokens"][0], skip_special_tokens=True)
        T = int(len(stats["greedy_tokens"][0]))
        print(f"  row_sq={row_sq:.5f}  row_entropy={row_ent:.4f}  "
              f"NLP={nlp_score:.4f}  T={T}")
        print(f"  Answer: {gen_text[:120]}")

        results.append({
            "id": q["id"],
            "locality": q["locality"],
            "question": q["question"],
            "answer": gen_text,
            "row_sq": row_sq,
            "row_entropy": row_ent,
            "nlp": nlp_score,
            "mean_token_entropy": ent_score,
            "n_tokens": T,
        })

    if not results:
        print("No results.")
        return

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # --- Summary ---
    local = [r for r in results if r["locality"] == 0]
    glob  = [r for r in results if r["locality"] == 1]

    def gstats(rows, key):
        vals = [r[key] for r in rows]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}

    summary: dict = {}
    main_metrics = ("row_sq", "row_entropy", "nlp", "mean_token_entropy")
    for metric in main_metrics:
        summary[metric] = {
            "local": gstats(local, metric),
            "global": gstats(glob, metric),
        }
        try:
            from scipy.stats import mannwhitneyu, spearmanr
            lv = [r[metric] for r in local]
            gv = [r[metric] for r in glob]
            _, pval = mannwhitneyu(lv, gv, alternative="two-sided")
            all_loc = [r["locality"] for r in results]
            all_met = [r[metric] for r in results]
            rho, sp = spearmanr(all_loc, all_met)
            summary[metric]["mannwhitney_p"] = float(pval)
            summary[metric]["spearman_rho"] = float(rho)
            summary[metric]["spearman_p"] = float(sp)
        except ImportError:
            pass

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        display = {
            "row_sq":           "Row Concentration\n(mean Σ A²)",
            "row_entropy":      "Row Entropy\n(mean −Σ A log A)",
            "nlp":              "Neg. Log Probability",
            "mean_token_entropy": "Token Entropy",
        }
        fig, axes = plt.subplots(1, len(display), figsize=(5 * len(display), 5))
        for ax, (metric, panel_label) in zip(axes, display.items()):
            lv = [r[metric] for r in local]
            gv = [r[metric] for r in glob]
            ax.boxplot([lv, gv], labels=["Local", "Global"])
            ax.scatter([1] * len(lv), lv, color="steelblue", alpha=0.7, zorder=3, s=50)
            ax.scatter([2] * len(gv), gv, color="tomato",   alpha=0.7, zorder=3, s=50)
            ax.set_title(panel_label)
            ax.set_ylabel("Score")
            rho = summary[metric].get("spearman_rho", float("nan"))
            pval = summary[metric].get("mannwhitney_p", float("nan"))
            ax.set_xlabel(f"Spearman ρ={rho:.2f}  MW-p={pval:.3f}")
        plt.suptitle(
            "Exp 5 — Question Locality vs Row-Level Attention Concentration",
            fontsize=12,
        )
        plt.tight_layout()
        plt.savefig(out_dir / "figure_exp5.png", dpi=150)
        print(f"\nPlot saved to {out_dir}/figure_exp5.png")
    except Exception as e:
        print(f"Plot failed: {e}")

    print(f"\n=== Exp 5 Summary ===")
    print(f"{'Metric':<22} {'Local mean':>12} {'Global mean':>12} {'Spearman ρ':>12} {'MW-p':>10}")
    for metric in main_metrics:
        ls = summary[metric]["local"]["mean"]
        gs = summary[metric]["global"]["mean"]
        rho = summary[metric].get("spearman_rho", float("nan"))
        pval = summary[metric].get("mannwhitney_p", float("nan"))
        print(f"  {metric:<20} {ls:>12.5f} {gs:>12.5f} {rho:>12.3f} {pval:>10.3f}")
    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
