"""
Generate Markdown result files from experiment output JSONs.

Usage:
    python -m experiments.generate_results_md --output-dir outputs/results_md

Reads from:
    outputs/exp1_fp16/summary.json
    outputs/exp2_fp16/summary.json
    outputs/exp4_fp16/summary.json
    outputs/exp5_locality/summary.json
    outputs/markov_attention/results.json

Writes:
    outputs/results_md/exp1_llcc.md
    outputs/results_md/exp2_tdo.md
    outputs/results_md/exp4_classification.md
    outputs/results_md/exp5_locality.md
    outputs/results_md/markov_attention.md
    outputs/results_md/index.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def fmt(v, decimals=4):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return f"{v:.{decimals}f}"


# ---------------------------------------------------------------------------
# Exp 1 — LLCC
# ---------------------------------------------------------------------------

def md_exp1(summary: dict) -> str:
    p = summary.get("paper_llcc", {})
    b = summary.get("baseline_llcc", {})
    s = summary.get("sink_corrected_llcc", {})
    n = summary.get("n_images", "?")

    lines = [
        "# Exp 1 — LLCC Hallucination Detection",
        "",
        f"**Images evaluated:** {n}  ",
        f"**Hyperparameters:** layers 19–25, topk=3, matcher=predefined",
        "",
        "## Results",
        "",
        "| Variant | Precision | Recall | F1 | TP | FP | FN |",
        "|---|---|---|---|---|---|---|",
        f"| Paper (labels replay) | {fmt(p.get('precision'))} | {fmt(p.get('recall'))} | {fmt(p.get('f1'))} | {p.get('tp','—')} | {p.get('fp','—')} | {p.get('fn','—')} |",
        f"| Baseline LLCC (re-run) | {fmt(b.get('precision'))} | {fmt(b.get('recall'))} | {fmt(b.get('f1'))} | {b.get('tp','—')} | {b.get('fp','—')} | {b.get('fn','—')} |",
        f"| Sink-corrected LLCC | {fmt(s.get('precision'))} | {fmt(s.get('recall'))} | {fmt(s.get('f1'))} | {s.get('tp','—')} | {s.get('fp','—')} | {s.get('fn','—')} |",
        "",
        "## Deltas",
        "",
        f"- ΔF1 baseline vs paper: **{fmt(summary.get('delta_f1_vs_paper'))}**",
        f"- ΔF1 sink-corrected vs baseline: **{fmt(summary.get('delta_f1_sink_vs_baseline'))}**",
        "",
        "## Notes",
        "",
        "- `Paper` = reading stored detections from `hallucination_results_val500.json`, not a re-run.",
        "- `Baseline` / `Sink-corrected` = fresh LLaVA generation + CHAIR + LLCC on our captions.",
        "- The two F1 values are **not directly comparable**: different captions → different word sets.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Exp 2 — Spectral TDO
# ---------------------------------------------------------------------------

def md_exp2(summary: dict) -> str:
    n = summary.get("n_images", "?")
    real = summary.get("real_objects", {})
    hallu = summary.get("hallucinated_objects", {})
    stats = summary.get("stats", {})

    lines = [
        "# Exp 2 — Spectral TDO (Attention Routing Geometry)",
        "",
        f"**Images evaluated:** {n}  ",
        f"**Hyperparameters:** layers 19–25",
        "",
        "## Group Means",
        "",
        "| Group | mean PR | mean leading-ratio | n |",
        "|---|---|---|---|",
        f"| Real objects | {fmt(real.get('mean_pr'))} | {fmt(real.get('mean_leading_ratio'))} | {real.get('n','—')} |",
        f"| Hallucinated objects | {fmt(hallu.get('mean_pr'))} | {fmt(hallu.get('mean_leading_ratio'))} | {hallu.get('n','—')} |",
        "",
    ]

    if stats:
        lines += [
            "## Statistical Tests",
            "",
            "| Metric | Δ(real−hallu) | Mann-Whitney p | Cohen's d |",
            "|---|---|---|---|",
        ]
        for key, label in [("pr", "PR"), ("leading_ratio", "Leading ratio")]:
            d = stats.get(key, {})
            lines.append(
                f"| {label} | {fmt(d.get('delta_mean'))} | {fmt(d.get('mannwhitney_p'), 4)} | {fmt(d.get('cohens_d'))} |"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Exp 4 — Type classification
# ---------------------------------------------------------------------------

def md_exp4(summary: dict) -> str:
    acc = summary.get("accuracy")
    n = summary.get("n_samples", "?")
    breakdown = summary.get("per_type", {})

    lines = [
        "# Exp 4 — Hallucination Type Classification",
        "",
        f"**Samples evaluated:** {n}  ",
        f"**Overall accuracy:** {fmt(acc)}",
        "",
    ]

    if breakdown:
        lines += [
            "## Per-type Accuracy",
            "",
            "| Type | Correct | Total | Accuracy |",
            "|---|---|---|---|",
        ]
        for t, v in breakdown.items():
            c = v.get("correct", 0)
            tot = v.get("total", 0)
            a = c / tot if tot else 0
            lines.append(f"| {t} | {c} | {tot} | {fmt(a)} |")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Exp 5 — Locality
# ---------------------------------------------------------------------------

def md_exp5(summary: dict) -> str:
    main_metrics = ["row_sq", "row_entropy", "nlp", "mean_token_entropy"]

    lines = [
        "# Exp 5 — Question Locality vs Row-Level Attention Concentration",
        "",
        "**Metrics:**",
        "- `row_sq` = mean Σ_j A²_{t,j} per row — high = concentrated, low = scattered",
        "- `row_entropy` = mean −Σ_j A_{t,j} log A_{t,j} — high = scattered (non-selection)",
        "",
        "**Hypothesis:** local questions → concentrated rows (model fetches a specific span);",
        "global questions → scattered rows (model integrates many positions).",
        "",
        "## Results",
        "",
        "| Metric | Local mean | Global mean | Spearman ρ | Mann-Whitney p |",
        "|---|---|---|---|---|",
    ]

    for metric in main_metrics:
        m = summary.get(metric, {})
        lm = fmt(m.get("local", {}).get("mean"), 5)
        gm = fmt(m.get("global", {}).get("mean"), 5)
        rho = fmt(m.get("spearman_rho"), 3)
        pval = fmt(m.get("mannwhitney_p"), 4)
        lines.append(f"| `{metric}` | {lm} | {gm} | {rho} | {pval} |")

    lines.append("")
    lines.append("*(ρ > 0: metric increases with locality=global questions)*")
    lines.append("")

    # Individual results table
    lines += [
        "## Per-question Results",
        "",
        "| ID | Locality | row_sq | row_entropy | Answer (truncated) |",
        "|---|---|---|---|---|",
    ]
    # These come from results.json, not summary.json — we'll leave a note
    lines.append("| — | — | — | — | see `outputs/exp5_locality/results.json` |")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markov attention
# ---------------------------------------------------------------------------

def md_markov(records: list, img_dir: Path) -> str:
    lines = [
        "# Markov Stationary Distributions of Attention Matrices",
        "",
        "For each image, the attention matrix A ∈ R^{T × n\\_patches} (T generated tokens,",
        "576 patches) is used to build a patch co-attention transition matrix:",
        "",
        "```",
        "M[i,j] = Σ_t A[t,i] · A[t,j]    (co-attention frequency)",
        "T[i,j] = M[i,j] / Σ_k M[i,k]    (row-normalised, stochastic)",
        "π T = π                           (stationary distribution, power iteration)",
        "```",
        "",
        "High π_j means patch j is a structural attractor: co-attended with most other patches.",
        "",
        "## Gallery",
        "",
    ]

    for r in records:
        img_name = r["image"]
        stem = Path(img_name).stem
        png = f"img_{stem}_stationary.png"
        caption = r.get("caption", "")[:80]
        ent = fmt(r.get("stationary_entropy"), 3)
        top5 = r.get("top5_patches", [])
        lines += [
            f"### {img_name}",
            "",
            f"![stationary heatmap]({png})",
            "",
            f"**Caption:** {caption}…  ",
            f"**π entropy:** {ent}  ",
            f"**Top-5 patches (by π):** {top5}",
            "",
        ]

    if not records:
        lines.append("*No results yet — run `experiments/viz_markov_attention.py`.*")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

def md_index(available: list) -> str:
    lines = [
        "# SADT Experiment Results",
        "",
        "| File | Description |",
        "|---|---|",
        "| [exp1_llcc.md](exp1_llcc.md) | LLCC hallucination detection (Exp 1) |",
        "| [exp2_tdo.md](exp2_tdo.md) | Spectral TDO attention routing (Exp 2) |",
        "| [exp4_classification.md](exp4_classification.md) | Type classification (Exp 4) |",
        "| [exp5_locality.md](exp5_locality.md) | Locality vs row-level attention (Exp 5) |",
        "| [markov_attention.md](markov_attention.md) | Markov stationary distributions (20 images) |",
        "",
        "## Missing results",
        "",
    ]
    missing = [x for x in ["exp1", "exp2", "exp4", "exp5", "markov"] if x not in available]
    if missing:
        for m in missing:
            lines.append(f"- `{m}` — output JSON not found, re-run the experiment.")
    else:
        lines.append("All results present.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outputs-root", default="outputs")
    p.add_argument("--output-dir", default="outputs/results_md")
    return p


def main():
    args = build_parser().parse_args()
    root = Path(args.outputs_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    available = []

    # Exp 1
    s1 = load_json(root / "exp1_fp16" / "summary.json")
    if s1:
        (out_dir / "exp1_llcc.md").write_text(md_exp1(s1))
        available.append("exp1")
        print("exp1_llcc.md written")

    # Exp 2
    s2 = load_json(root / "exp2_fp16" / "summary.json")
    if s2:
        (out_dir / "exp2_tdo.md").write_text(md_exp2(s2))
        available.append("exp2")
        print("exp2_tdo.md written")

    # Exp 4
    s4 = load_json(root / "exp4_fp16" / "summary.json")
    if s4:
        (out_dir / "exp4_classification.md").write_text(md_exp4(s4))
        available.append("exp4")
        print("exp4_classification.md written")

    # Exp 5
    s5 = load_json(root / "exp5_locality" / "summary.json")
    if s5:
        (out_dir / "exp5_locality.md").write_text(md_exp5(s5))
        available.append("exp5")
        print("exp5_locality.md written")

    # Markov
    markov_records = load_json(root / "markov_attention" / "results.json")
    markov_records = markov_records or []
    img_dir = root / "markov_attention"
    (out_dir / "markov_attention.md").write_text(md_markov(markov_records, img_dir))
    if markov_records:
        available.append("markov")
        print("markov_attention.md written")

    # Index
    (out_dir / "index.md").write_text(md_index(available))
    print("index.md written")

    print(f"\nAll files saved to {out_dir}/")


if __name__ == "__main__":
    main()
