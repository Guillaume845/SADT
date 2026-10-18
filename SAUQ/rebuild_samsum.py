"""
Rebuild SAMSum — SAUQ uncertainty vs summary quality on SAMSum.

Replicates the SAMSum experiment from:
  "Spectral Geometry of Attention: From Information Routing to Uncertainty"

For each dialogue in the SAMSum test set:
  1. Generate a summary with the LLM.
  2. Compute SAUQ, RAUQ, NLP, mean token entropy.
  3. Compute ROUGE-L against the reference summary.

Report Normalized Prediction Rejection Ratio (PRR; higher is better)
for each uncertainty method, following the LM Polygraph evaluation protocol.

Output
------
outputs/exp5b_samsum/
  results.json   — per-sample metrics
  summary.json   — PRR and Spearman ρ per uncertainty method
  figure_exp5b.png
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


# ---------------------------------------------------------------------------
# Model loading — pure text LLM (LLaMA-3.1-8b, Mistral-7B, etc.)
# ---------------------------------------------------------------------------

def load_llm(model_path: str, load_4bit: bool = False):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = dict(
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
    )
    if load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        del kwargs["torch_dtype"]

    # transformers < 4.40 rejects the Llama-3 rope_scaling format.
    # Silence validation, load config, patch the dict, then restore.
    from transformers import AutoConfig
    from transformers.models.llama.configuration_llama import LlamaConfig
    _orig_validate = LlamaConfig._rope_scaling_validation
    LlamaConfig._rope_scaling_validation = lambda self: None
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    finally:
        LlamaConfig._rope_scaling_validation = _orig_validate
    # If rope_scaling uses a type unknown to this transformers version, disable it.
    # SAMSum sequences are short enough that standard RoPE suffices.
    rs = getattr(config, "rope_scaling", None)
    KNOWN_ROPE_TYPES = {"linear", "dynamic"}
    if rs is not None:
        rope_type = rs.get("rope_type") or rs.get("type", "")
        if rope_type not in KNOWN_ROPE_TYPES:
            config.rope_scaling = None

    model = AutoModelForCausalLM.from_pretrained(model_path, config=config, **kwargs)
    model.eval()
    return tokenizer, model


# ---------------------------------------------------------------------------
# Stat collection (same logic as model_utils.collect_stats, model-agnostic)
# ---------------------------------------------------------------------------

def collect_stats(model, tokenizer, prompt: str,
                  max_new_tokens: int = 100,
                  min_new_tokens: int = 5,
                  max_prompt_tokens: int = 1024) -> Optional[dict]:
    import gc
    import torch

    gc.collect()
    torch.cuda.empty_cache()

    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                    max_length=max_prompt_tokens)
    input_ids = enc.input_ids.to(device)
    P = input_ids.shape[1]

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    NH = n_layers * n_heads

    with torch.no_grad():
        gen = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=False,
            output_attentions=True,
            output_scores=True,
            return_dict_in_generate=True,
        )

    gen_ids = gen.sequences[0, P:]
    T = int(gen_ids.shape[0])
    print(f"    [debug] P={P} T={T}", flush=True)
    if T < 2:
        return None

    square = np.zeros((NH, T, T), dtype=np.float32)
    for t in range(T):
        for l in range(n_layers):
            row = gen.attentions[t][l][0, :, 0, P:].cpu().float().numpy()
            n_gen = row.shape[1]
            hs, he = l * n_heads, (l + 1) * n_heads
            if n_gen > 0:
                square[hs:he, t, :n_gen] = row

    log_lls: List[float] = []
    token_probs: List[np.ndarray] = []
    for score, tid in zip(gen.scores, gen_ids):
        probs = score[0].float().softmax(-1).cpu().numpy()
        lp = math.log(float(probs[tid.item()]) + 1e-45)
        log_lls.append(lp)
        token_probs.append(probs)
    del gen

    import torch
    torch.cuda.empty_cache()

    return {
        "attention_all": [square],
        "attention_all_with_prompt": [],
        "attention_prefill": [],
        "greedy_log_likelihoods": [np.array(log_lls, dtype=np.float32)],
        "greedy_tokens": [gen_ids.cpu().numpy()],
        "input_tokens": [input_ids[0].cpu().numpy()],
        "_token_probs": token_probs,
    }


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_prompt(tokenizer, dialogue: str) -> str:
    system = "You are a helpful assistant that summarizes conversations."
    user = (
        "Please summarize the following conversation in one or two concise sentences.\n\n"
        f"Conversation:\n{dialogue}\n\nSummary:"
    )
    # Use chat template if available (LLaMA-3.1, Mistral-Instruct, etc.)
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    # Fallback: Vicuna-style
    return (
        f"A chat between a curious user and an assistant. "
        f"USER: {user} ASSISTANT:"
    )


# ---------------------------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------------------------

def rouge_l(hypothesis: str, reference: str) -> float:
    try:
        from rouge_score import rouge_scorer as rs
        scorer = rs.RougeScorer(["rougeL"], use_stemmer=True)
        return float(scorer.score(reference, hypothesis)["rougeL"].fmeasure)
    except Exception:
        hyp, ref = hypothesis.lower().split(), reference.lower().split()
        if not hyp or not ref:
            return 0.0
        m, n = len(ref), len(hyp)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = dp[i-1][j-1] + 1 if ref[i-1] == hyp[j-1] \
                    else max(dp[i-1][j], dp[i][j-1])
        lcs = dp[m][n]
        prec, rec = lcs / n if n else 0., lcs / m if m else 0.
        return 2 * prec * rec / (prec + rec) if prec + rec else 0.


# ---------------------------------------------------------------------------
# PRR — Normalized Prediction Rejection Ratio (LM Polygraph convention)
# ---------------------------------------------------------------------------

def compute_prr(uncertainty: np.ndarray, quality: np.ndarray) -> float:
    """
    PRR = (AUC_method - AUC_random) / (AUC_oracle - AUC_random)

    Retention curve: fraction of samples KEPT (1 → 0), sorted by uncertainty
    descending (most uncertain rejected first). Quality = mean ROUGE-L of kept samples.
    """
    n = len(uncertainty)
    if n < 2:
        return float("nan")

    # Sort by uncertainty descending → reject most uncertain first
    order = np.argsort(-uncertainty)
    quality_sorted = quality[order]

    # Retention fractions: keep all, then progressively reject
    retention = np.linspace(1.0, 1.0 / n, n)
    # Quality at each retention level = mean of kept suffix
    kept_quality = np.array([quality_sorted[i:].mean() for i in range(n)])

    auc_method = np.trapz(kept_quality, retention)
    auc_random = np.trapz(np.full(n, quality.mean()), retention)

    # Oracle: reject worst quality first
    oracle_order = np.argsort(quality)  # ascending = worst first
    oracle_quality = quality[oracle_order[::-1]]  # descending = best kept first
    oracle_kept = np.array([oracle_quality[:n - i].mean() for i in range(n)])
    auc_oracle = np.trapz(oracle_kept, retention)

    if abs(auc_oracle - auc_random) < 1e-10:
        return 0.0
    return float((auc_method - auc_random) / (auc_oracle - auc_random))


# ---------------------------------------------------------------------------
# SAUQ estimator (paper config)
# ---------------------------------------------------------------------------

def make_sauq(n_layers: int, n_heads: int) -> SpectralRAUQ:
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
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--load-4bit", action="store_true", default=False)
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--limit", type=int, default=400)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--output-dir", default="outputs/exp5b_samsum")
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

        ll = stats["greedy_log_likelihoods"][0]
        sq = stats["attention_all"][0]

        nlp_val  = float(nlp(ll))
        ent_val  = float(mean_token_entropy(stats["_token_probs"]))
        rauq_val = float(compute_rauq(sq, ll, n_layers, n_heads))
        sauq_val = float(sauq_estimator(stats)[0])

        hypothesis = tokenizer.decode(stats["greedy_tokens"][0],
                                      skip_special_tokens=True).strip()
        rl = rouge_l(hypothesis, reference)

        print(f"  SAUQ={sauq_val:.4f}  RAUQ={rauq_val:.4f}  "
              f"NLP={nlp_val:.4f}  Ent={ent_val:.4f}  ROUGE-L={rl:.4f}")
        print(f"  Ref: {reference[:100]}")
        print(f"  Gen: {hypothesis[:100]}")

        results.append({
            "id": sid, "rouge_l": rl,
            "sauq": sauq_val, "rauq": rauq_val,
            "nlp": nlp_val, "mean_token_entropy": ent_val,
            "n_tokens": int(len(stats["greedy_tokens"][0])),
        })

        with open(out_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    if not results:
        print("No results.")
        return

    rouge_vals = np.array([r["rouge_l"] for r in results])
    methods = ["sauq", "rauq", "nlp", "mean_token_entropy"]

    from scipy.stats import spearmanr
    summary: Dict = {
        "n_samples": len(results),
        "model": args.model_path,
        "rouge_l_mean": float(rouge_vals.mean()),
        "rouge_l_std":  float(rouge_vals.std()),
    }

    print(f"\n=== Exp 5b — SAMSum (n={len(results)}) ===")
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

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 5))
        for ax, method in zip(axes, methods):
            vals = np.array([r[method] for r in results])
            ax.scatter(vals, rouge_vals, alpha=0.4, s=15)
            prr = summary[method]["prr"]
            rho = summary[method]["spearman_rho"]
            ax.set_xlabel(method); ax.set_ylabel("ROUGE-L")
            ax.set_title(f"PRR={prr:.3f}\nSpearman ρ={rho:.3f}")
            ax.grid(True, alpha=0.3)
        plt.suptitle(f"Exp 5b — SAMSum uncertainty vs ROUGE-L (n={len(results)})",
                     fontsize=12)
        plt.tight_layout()
        plt.savefig(out_dir / "figure_exp5b.png", dpi=150)
        print(f"\nPlot saved to {out_dir}/figure_exp5b.png")
    except Exception as e:
        print(f"Plot failed: {e}")

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
