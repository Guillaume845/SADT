"""Shared utilities for SADT × SAUQ experiments."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.sadt.core import (
    SADTConfig,
    _attention_for_token,
    _get_img_span,
)


def load_model(model_path: str, model_base: Optional[str] = None):
    if "qwen" in model_path.lower() or "qwen2" in model_path.lower():
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(model_path)
        return processor, model, processor, "qwen2vl"

    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    from llava.utils import disable_torch_init

    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, model_base, model_name, load_4bit=False
    )
    return tokenizer, model, image_processor, model_name


def default_config(**overrides) -> SADTConfig:
    kwargs: dict = {
        "image_attention_layers": tuple(range(14, 25)),
        "topk_patches": 3,
        "semantic_matcher": "predefined",
        "image_token_count": 144,
        "grid_size": 12,
    }
    kwargs.update(overrides)
    return SADTConfig(**kwargs)


def get_img_idx(input_ids: torch.Tensor) -> int:
    return _get_img_span(input_ids)[0]


def get_image_token_count(input_ids: torch.Tensor) -> int:
    return _get_img_span(input_ids)[1]


# ---------------------------------------------------------------------------
# Attention matrix utilities
# ---------------------------------------------------------------------------

def compute_full_attention_matrix(
    output: Any,
    img_idx: int,
    image_token_count: int,
    layers: Sequence[int],
) -> np.ndarray:
    """A[t, j] = attention from generated token t to patch j, summed over target layers.
    Shape: [n_generated_tokens, n_patches].
    """
    attentions = output["attentions"]
    rows = []
    for t in range(len(attentions)):
        att = _attention_for_token(attentions, t, img_idx, image_token_count)
        rows.append(att[list(layers)].sum(axis=0))
    return np.stack(rows)


def compute_sink_weights(
    output: Any,
    img_idx: int,
    image_token_count: int,
    layers: Sequence[int],
) -> np.ndarray:
    """μ_j = mean attention received by patch j across all generated tokens."""
    A = compute_full_attention_matrix(output, img_idx, image_token_count, layers)
    return A.mean(axis=0)  # [n_patches]


# ---------------------------------------------------------------------------
# Spectral / TDO utilities
# ---------------------------------------------------------------------------

def compute_tdo_participation_ratio(A: np.ndarray, epsilon: float = 1e-10) -> float:
    """PR of TDO D_{μ,π} = Ω_μ⁻¹ A^T Ω_π A.

    Uses symmetric factorisation B = Ω_π^{1/2} A Ω_μ^{-1/2}, so eigenvalues
    of D equal squared singular values of B.
    PR = (Σλ)² / (n · Σλ²) ∈ (0, 1].
    """
    row_sums = A.sum(axis=1)
    col_sums = A.sum(axis=0)

    row_mask = row_sums > epsilon
    col_mask = col_sums > epsilon
    if row_mask.sum() == 0 or col_mask.sum() == 0:
        return 0.0

    A_sub = A[row_mask][:, col_mask]
    rs = row_sums[row_mask]
    cs = col_sums[col_mask]

    B = (np.sqrt(rs)[:, None] * A_sub) / np.sqrt(cs)[None, :]
    try:
        _, s, _ = np.linalg.svd(B, full_matrices=False)
    except np.linalg.LinAlgError:
        return 0.0

    lambdas = s ** 2
    n = len(lambdas)
    sl = lambdas.sum()
    sl2 = (lambdas ** 2).sum()
    if n == 0 or sl2 < epsilon:
        return 0.0
    return float((sl ** 2) / (n * sl2))


def compute_spectral_leading_ratio(A: np.ndarray, epsilon: float = 1e-10) -> float:
    """λ₁ / Σλ of TDO — concentration of routing in the top direction."""
    row_sums = A.sum(axis=1)
    col_sums = A.sum(axis=0)
    row_mask = row_sums > epsilon
    col_mask = col_sums > epsilon
    if row_mask.sum() == 0 or col_mask.sum() == 0:
        return 0.0
    A_sub = A[row_mask][:, col_mask]
    rs = row_sums[row_mask]
    cs = col_sums[col_mask]
    B = (np.sqrt(rs)[:, None] * A_sub) / np.sqrt(cs)[None, :]
    try:
        _, s, _ = np.linalg.svd(B, full_matrices=False)
    except np.linalg.LinAlgError:
        return 0.0
    lambdas = s ** 2
    sl = lambdas.sum()
    if sl < epsilon:
        return 0.0
    return float(lambdas[0] / sl)


# ---------------------------------------------------------------------------
# Per-word feature extraction
# ---------------------------------------------------------------------------

def compute_logit_lens_entropy(
    output: Any,
    model: Any,
    img_idx: int,
    token_idx: int,
    top_patch_idx: int,
    layers: Sequence[int],
) -> float:
    """Mean entropy of Logit Lens softmax at the top patch, averaged over layers."""
    hidden_states = output["hidden_states"][0]
    abs_idx = img_idx + top_patch_idx
    entropies = []
    with torch.no_grad():
        for layer in layers:
            layer_index = max(0, min(layer, len(hidden_states) - 1))
            h = hidden_states[layer_index][:, abs_idx, :].to(model.device)
            logits = model.lm_head(h)  # [1, vocab]
            log_p = torch.log_softmax(logits[0], dim=-1).float().cpu().numpy()
            p = np.exp(log_p)
            entropies.append(float(-np.sum(p * log_p)))
    return float(np.mean(entropies)) if entropies else 0.0


def compute_attention_concentration(
    output: Any,
    img_idx: int,
    image_token_count: int,
    token_idx: int,
    layers: Sequence[int],
) -> float:
    """max(α) / mean(α) over patches for the target token — higher = more concentrated."""
    att = _attention_for_token(output["attentions"], token_idx, img_idx, image_token_count)
    layer_att = att[list(layers)].sum(axis=0)
    total = layer_att.sum()
    if total < 1e-10:
        return 0.0
    return float(layer_att.max() / (total / max(len(layer_att), 1)))


def compute_sink_ratio(
    output: Any,
    img_idx: int,
    image_token_count: int,
    token_idx: int,
    layers: Sequence[int],
    mu: np.ndarray,
) -> float:
    """Corrected score of raw-top patch / max corrected score.
    < 1 means the raw-top patch is an attention sink.
    """
    att = _attention_for_token(output["attentions"], token_idx, img_idx, image_token_count)
    layer_att = att[list(layers)].sum(axis=0)
    corrected = layer_att / (mu + 1e-8)
    raw_top = int(np.argmax(layer_att))
    max_corr = float(corrected.max())
    if max_corr < 1e-10:
        return 1.0
    return float(corrected[raw_top] / max_corr)
