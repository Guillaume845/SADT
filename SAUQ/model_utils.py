"""Shared utilities for SAUQ experiments: model loading and stat collection."""
from __future__ import annotations

import sys
import math
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Model loading  (fp16, eager attention so output_attentions=True always works)
# ---------------------------------------------------------------------------

def load_text_model(model_path: str, load_4bit: bool = False):
    """Load LLaVA model for text-only use (no images).

    Default is fp16 (not 4-bit) because:
      - RTX 3090 has 24 GB VRAM; LLaMA-7B fp16 takes ~14 GB → fits fine.
      - bitsandbytes 4-bit + output_attentions=True causes generation to
        produce <2 tokens on this setup.
      - attn_implementation="eager" is set to guarantee attention weights
        are returned without SDPA fallback warnings.
    """
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    from llava.utils import disable_torch_init

    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, _, _ = load_pretrained_model(
        model_path, None, model_name,
        load_4bit=load_4bit,
        load_8bit=False,
    )
    # Force eager attention so output_attentions=True always returns weights.
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"
    for module in model.modules():
        cls_name = type(module).__name__
        if "SdpaAttention" in cls_name or "FlashAttention" in cls_name:
            # Swap to eager variant if available on this module
            pass  # transformer 4.37 handles this via config above
    model.eval()
    return tokenizer, model


# ---------------------------------------------------------------------------
# Prompt formatting (Vicuna conversation template)
# ---------------------------------------------------------------------------

VICUNA_SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def format_qa_prompt(context: str, question: str) -> str:
    user = f"Read the following text carefully.\n\n{context}\n\nQuestion: {question}"
    return f"{VICUNA_SYSTEM} USER: {user} ASSISTANT:"


def format_copy_prompt(sequence: str) -> str:
    """
    Few-shot copy prompt that works with Vicuna-style models.
    The two examples teach the format; the third is the actual task.
    """
    user = (
        "I will give you a sequence of digits. Copy it back exactly, "
        "digit by digit separated by spaces.\n\n"
        "Sequence: 1 0 1\n"
        "Copy: 1 0 1\n\n"
        "Sequence: 0 1 0 0\n"
        "Copy: 0 1 0 0\n\n"
        f"Sequence: {sequence}\n"
        "Copy:"
    )
    return f"{VICUNA_SYSTEM} USER: {user} ASSISTANT:"


# ---------------------------------------------------------------------------
# Stat collection (square mode only — no separate prefill pass)
# ---------------------------------------------------------------------------

def collect_stats(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 80,
    min_new_tokens: int = 5,
    max_prompt_tokens: int = 1024,
) -> Optional[dict]:
    """
    Run greedy generation and collect stats for SpectralRAUQ (square mode).

    Uses attn_mode="square" (gen-only T×T causal block), which avoids the
    expensive separate prefill pass and is stable with fp16 generation.

    Returns dict with:
      attention_all           — list[(n_layers*n_heads, T, T)]
      greedy_log_likelihoods  — list[(T,)]
      greedy_tokens           — list[(T,)] int
      input_tokens            — list[(P,)] int
      _token_probs            — list of T arrays of shape (vocab_size,)
      _n_layers, _n_heads

    Returns None when the model generates fewer than 2 tokens.
    """
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_prompt_tokens)
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

    # gen.attentions[t][l]: (1, n_heads, 1, k_len) where k_len = P+t or P+t+1
    # depending on whether the model includes self-attention in the output.
    # Square (T×T): gen-only causal block
    square = np.zeros((NH, T, T), dtype=np.float32)
    for t in range(T):
        for l in range(n_layers):
            row = gen.attentions[t][l][0, :, 0, P:].cpu().float().numpy()  # (n_heads, n_gen)
            n_gen = row.shape[1]                                             # t or t+1
            hs, he = l * n_heads, (l + 1) * n_heads
            if n_gen > 0:
                square[hs:he, t, :n_gen] = row

    log_lls: List[float] = []
    token_probs: List[np.ndarray] = []
    for score, tid in zip(gen.scores, gen_ids):
        probs = torch.softmax(score[0], dim=-1).cpu().float().numpy()
        lp = math.log(float(probs[tid.item()]) + 1e-45)
        log_lls.append(lp)
        token_probs.append(probs)
    del gen
    torch.cuda.empty_cache()

    return {
        "attention_all": [square],
        # square mode needs no with_prompt or prefill
        "attention_all_with_prompt": [],
        "attention_prefill": [],
        "greedy_log_likelihoods": [np.array(log_lls, dtype=np.float32)],
        "greedy_tokens": [gen_ids.cpu().numpy()],
        "input_tokens": [input_ids[0].cpu().numpy()],
        "_token_probs": token_probs,
        "_n_layers": n_layers,
        "_n_heads": n_heads,
    }


# ---------------------------------------------------------------------------
# Scalar metrics
# ---------------------------------------------------------------------------

def nlp(log_lls: np.ndarray) -> float:
    return float(-np.mean(log_lls))


def mean_token_entropy(token_probs: List[np.ndarray]) -> float:
    return float(np.mean([-np.sum(p * np.log(p + 1e-45)) for p in token_probs]))


def compute_rauq(
    square: np.ndarray,
    log_lls: np.ndarray,
    n_layers: int,
    n_heads: int,
    alpha: float = 0.2,
) -> float:
    """Previous-token attention recurrence (RAUQ baseline)."""
    T = square.shape[1]
    if T < 2:
        return 0.0

    layers_mid = list(range(n_layers // 3, int(np.ceil(n_layers / 3 * 2)) + 1))
    layer_scores = []
    for layer in layers_mid:
        la = square[layer * n_heads:(layer + 1) * n_heads]   # (H, T, T)
        prev = np.stack([la[:, t, t - 1] for t in range(1, T)], axis=1)  # (H, T-1)
        best_h = int(prev.mean(axis=1).argmax())
        w = prev[best_h]  # (T-1,)

        c = float(np.exp(log_lls[0]))
        confs = [c]
        for t in range(1, T):
            c = alpha * float(np.exp(log_lls[t])) + (1 - alpha) * float(w[t - 1]) * c
            confs.append(c)
        layer_scores.append(1.0 - float(np.log(np.clip(confs, 1e-10, None)).mean()))

    return float(np.max(layer_scores))
