# ============================================================================
# rauq_spectral_replication_v2.py
#
# DEPLOYMENT: this file uses a relative import (`from .estimator import
# Estimator`), same as the original rauq_spectral_replication.py, so it must
# be placed at src/lm_polygraph/estimators/rauq_spectral.py in the target
# repo (replacing the file there), NOT run as a standalone script.
#
# This is rauq_spectral_replication.py (byte-identical base -- confirmed via
# diff) plus the following additions made during rebuttal prep. None of them
# change the behavior of the paper's actual "Ours" config (mass=attn_mass,
# attn_mode=full_square, default rank_estimator="pr", default
# selection_mode="square") -- they only take effect when explicitly opted
# into via new constructor args, all defaulting to prior behavior:
#
#   1. P1 fix: rank_estimator="pr_wc_centered" was double-normalized (divided
#      by both its own (n-1) denominator and the outer `denom` in
#      _fresh_weights), collapsing the signal toward 0. Fixed via
#      _PRE_NORMALIZED_ESTIMATORS. Added a validity guard requiring
#      mass="attn_mass" for this estimator (raises ValueError otherwise).
#   2. head_selection_weighting={"euclidean","intrinsic","centered_intrinsic"}:
#      mass-aware (attn_mass-weighted) head selection in pass 1, in addition
#      to the original unweighted ("euclidean") Gram-based selection.
#      "centered_intrinsic" mirrors pr_wc_centered's centering, applied to
#      the selection score instead of the pass-2 signal.
#   3. renormalize_rows: B_tilde = diag(B 1)^{-1} B row-renormalization,
#      mainly relevant for attn_mode="square" whose rows don't sum to 1
#      (see the P3 row-sum check: gen-only block rows retain only ~12-22%
#      of softmax mass on average). Forces the fresh-Gram path when True.
#   4. Logging: __call__ now logs the resolved attn_mode="auto" outcome
#      (use_full_attn/use_prefill/selection_mode), since the previous
#      __str__ omitted mode_desc when attn_mode="auto", making the exact
#      config behind any "SAUQ (Ours)" number unrecoverable from logs alone.
#
# See the corresponding disentangled_grid_ablation.sh script for the S1-S4 x
# B1-B4 selection/signal ablation grid this enables.
# ============================================================================

import logging
import numpy as np
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from .estimator import Estimator

logger = logging.getLogger(__name__)

VALID_MASSES = {
    "uniform",
    "front_biased",
    "rear_biased",
    "attn_mass",
    "row_attn_mass",
    "zipf",
    "zipf_inv",
    "sink_smooth",
    "sink_topk",
    "unembed_bias",
    "unembed_norm",
    "unembed_pc",
}
VALID_ATTN_MODES = {
    "auto",
    "square",
    "rectangular",
    "full_square",
}
VALID_RANK_ESTIMATORS = {
    "pr",
    "erank",
    "stable_rank",
    "spectral_entropy",
    "col_entropy",
    "pr_wc_centered",
}
VALID_HEAD_SELECTIONS = {"argmax", "topk", "all"}
VALID_HEAD_AGGREGATIONS = {"mean", "weighted"}
VALID_SELECTION_MODES = {"square", "rectangular", "full_square", "match"}
VALID_SELECTION_WEIGHTINGS = {"euclidean", "intrinsic", "centered_intrinsic"}

# Masses whose per-key weights are fixed and can be precomputed before the j loop.
# Dynamic masses (attn_mass, row_attn_mass) depend on the current attention slice
# and cannot be precomputed, so they fall back to the fresh-Gram path.
_STATIC_MASSES = frozenset(
    {
        "uniform",
        "front_biased",
        "rear_biased",
        "zipf",
        "zipf_inv",
        "sink_smooth",
        "unembed_bias",
        "unembed_norm",
        "unembed_pc",
    }
)

# Only "pr" has an incremental bordered-Gram path. All other estimators require
# the full spectrum or A-matrix, so they route through the fresh-Gram path.
_INCREMENTAL_RANK_ESTIMATORS = frozenset({"pr"})

# Estimators that already return a fractional value in [0, 1] (they divide by
# their own effective dimension internally) and must NOT be divided by `denom`
# again in _fresh_weights. pr_wc_centered divides by (n-1) — the centered
# space's dimension — which is not the same quantity as `denom` (n_rows or
# cumsum_mu), so applying both would double-normalize and collapse the signal
# toward 0 (s_t -> 0 drives the recurrence to c_t ~= alpha * p_t).
_PRE_NORMALIZED_ESTIMATORS = frozenset({"pr_wc_centered"})

from scipy.sparse.linalg import eigsh


def delta_mix(A: np.ndarray, eps: float = 1e-6) -> float:
    n_rows, n_cols = A.shape
    if n_cols <= 1 or n_rows <= 1:
        return 0.0

    pi_val = 1.0 / n_rows
    mu = np.maximum(A.mean(axis=0), eps)

    B = np.sqrt(pi_val) * A / np.sqrt(mu[None, :])
    K = B.T @ B

    vals = eigsh(K, k=2, which="LA", return_eigenvectors=False)
    vals = np.sort(np.clip(vals, 0.0, None))[::-1]

    return float(np.sqrt(min(vals[1], 1.0)))


def effective_rank(A: np.ndarray, eps: float = 1e-10) -> float:
    """
    SVD-based effective rank (Roy & Vetterli, 2007).

    erank(A) = exp( -Σ σ̂ᵢ log σ̂ᵢ )   where σ̂ᵢ = σᵢ / Σσⱼ.

    Note: this takes Ã directly (not the Gram) so the normalization is over
    σ, not σ². To work on the Gram's eigenvalues (σ²) instead, see
    spectral_entropy below.
    """
    s = np.linalg.svd(A, compute_uv=False)
    s = s[s > eps]
    if len(s) == 0:
        return 1.0
    s_hat = s / s.sum()
    entropy = -np.sum(s_hat * np.log(s_hat + eps))
    return np.exp(entropy)


def participation_ratio_centered_from_attn_mass_gram(
    gram: np.ndarray,
    eps: float = 1e-10,
) -> float:
    n = gram.shape[0]

    # Existing attn_mass Gram is n * B B^T, so normalize to B B^T.
    gram = gram / n

    tr = np.trace(gram)
    frob_sq = np.sum(gram * gram)

    tr_c = tr - 1.0
    frob_sq_c = frob_sq - 1.0

    if tr_c <= eps or frob_sq_c <= eps:
        return 0.0

    pr_c = (tr_c * tr_c) / (frob_sq_c + eps)
    return float(pr_c / max(n - 1, 1))


def participation_ratio(gram: np.ndarray, eps: float = 1e-10) -> float:
    """
    Participation ratio: trace(G)² / ‖G‖_F².  O(n²) — no SVD.

    Equals n when eigenvalues are uniform (full rank) and 1 when rank-1.
    Same extremes and intuition as effective_rank; different functional form.
    Scale-invariant: multiplying G by a constant leaves PR unchanged.
    """
    tr = np.trace(gram)
    frob_sq = np.sum(gram * gram)
    return tr * tr / (frob_sq + eps)


def stable_rank(gram: np.ndarray, eps: float = 1e-10) -> float:
    """
    Stable rank of A: ‖A‖_F² / ‖A‖_2² = trace(G) / λ_max(G).

    Input is the row Gram G = ÃÃᵀ, whose eigenvalues are the squared singular
    values of Ã.  Ratio is in [1, n]; equals 1 for rank-1 G, n for isotropic G.
    O(n³) due to the largest-eigenvalue computation.
    """
    tr = np.trace(gram)
    # eigvalsh exploits symmetry; top eigenvalue is the last.
    lam_max = np.linalg.eigvalsh(gram)[-1]
    return tr / (lam_max + eps)


def spectral_entropy(gram: np.ndarray, eps: float = 1e-10) -> float:
    """
    Rescaled Shannon entropy of the Gram eigenvalues.

    Let p_i = λ_i / Σλ_j (nonneg because G is PSD). Return exp(H(p)),
    matching erank's scale so outputs land in [1, n].  O(n³).
    """
    lams = np.linalg.eigvalsh(gram)
    lams = lams[lams > eps]
    if len(lams) == 0:
        return 1.0
    p = lams / lams.sum()
    H = -np.sum(p * np.log(p + eps))
    return np.exp(H)


def col_entropy(A: np.ndarray, eps: float = 1e-10) -> float:
    """
    Rescaled entropy of the column-sum distribution of Ã.

    q_l = (col-sum of Ã at l) / Σ col-sums.  Returns exp(H(q)) so the scale
    matches the other estimators: exp(H(uniform_n)) = n, exp(H(degenerate))=1.

    O(n²) — just a sum over the matrix.  This is a routing-concentration signal
    (not a spectral quantity), serving as a cheap scalar baseline.
    """
    col_sums = A.sum(axis=0)
    col_sums = col_sums[col_sums > eps]
    if len(col_sums) == 0:
        return 1.0
    q = col_sums / col_sums.sum()
    H = -np.sum(q * np.log(q + eps))
    return np.exp(H)


# Dispatch tables. Functions taking Gram (ÃÃᵀ) vs. A-matrix (Ã itself).
_GRAM_ESTIMATORS = {
    "pr": participation_ratio,
    "pr_wc_centered": participation_ratio_centered_from_attn_mass_gram,
    "stable_rank": stable_rank,
    "spectral_entropy": spectral_entropy,
}
_A_MATRIX_ESTIMATORS = {
    "erank": effective_rank,
    "col_entropy": col_entropy,
}


class SpectralRAUQ(Estimator):
    """
    Spectral RAUQ — spectral-domain extension of RAUQ.

    RAUQ structure (what we replicate exactly):
        1. Reshape attention to (n_layers, n_heads, seq_len, seq_len)
        2. Extract a per-position scalar from attention (RAUQ: diagonal offset=-1)
        3. Select one head per layer based on that scalar (RAUQ: argmax of mean)
        4. Run recurrence:  c_j = alpha * prob_j + (1-alpha) * weight_j * c_{j-1}
        5. Uncertainty = 1 - mean(log(confidence))
        6. Max across middle-third layers

    What changes (spectral replacement):
        Step 2: Instead of extracting the previous-token attention weight
                A[l,h,j,j-1] as a scalar, we compute the fractional effective rank
                of the token difference (Gram) operator A*A on the causal prefix:

                    spectral_weight_j = rank_measure(A*A[:j+1, :j+1]) / (j + 1)

                where A* is the adjoint of A with respect to mass π.

                Value ~1 = healthy routing (spread-spectrum, full-rank Gram),
                Value ~0 = collapsed routing (rank-1 Gram, sink head).

        Step 3: Head selection = argmax of square-mode participation ratio at the
                final generation step, computed for all heads in a single batched
                matrix multiply.  Only the winning head runs the full sequence.

        Step 4: The recurrence uses spectral_weight_j instead of attention diagonal.

    Performance optimisations (vs the naive implementation):
        1. Rank estimator: "pr" (default) uses the participation ratio
           trace(G)²/‖G‖_F² instead of SVD-based effective rank — O(n²) vs O(n³).
        2. Incremental Gram update: for static mass measures the causal structure
           means G_j has a bordered form G_j = [[G_{j-1}, cross], [cross^T, c]].
           We maintain trace and ‖G‖_F² via O(j²) cross-products instead of
           rebuilding the O(j² × j_cols) Gram at each step.
        3. Mass precomputation: static mass vectors are computed once per call
           (not once per j step).
        4. Prompt block precomputation (full_square mode): G_pp = A_pp Aᵀ_pp is
           computed once per (layer, head) call; subsequent steps extend it
           incrementally rather than copying prefill_attention into a fresh matrix.
        5. Two-pass head selection: pass 1 selects the head per layer via a
           batched square-mode PR at the final step (O(n_heads × T²) BLAS call);
           only the winning head runs the full incremental computation.

    Mass measures: see original docstring — unchanged.

    Args:
        alpha: Weight for combining probability and spectral weight.
        n_layers: Number of transformer layers (auto-detected if None).
        n_heads: Number of attention heads (auto-detected if None).
        use_entropy: Whether to use entropy-based uncertainty.
        instruct: Whether the model is instruction-tuned.
        mass: Mass measure π over token positions (default: "uniform").
            Acts as both column (key) and row (query) mass.  The weighted
            Gram is G_π = Π^{1/2} A Π^{-1} Aᵀ Π^{1/2}, and the fractional
            rank is PR(G_π) / Σπ_i.  In full_square mode, π is defined
            over the full (prompt + generated) sequence, so position-based
            masses like "rear_biased" naturally upweight generation tokens.
        attn_mode: Attention matrix to use ("auto", "square", "rectangular",
            "full_square").
        rank_estimator: Spectral measure applied to the weighted Gram
            G_π = ÃÃᵀ (or directly to Ã for A-matrix estimators).
            All variants return a value in [1, n_rows]; fractional rank
            divides by the effective dimension Σμ_i.

            "pr"               (default) trace(G)² / ‖G‖_F². O(n²).
                               Only estimator with an incremental path.
            "erank"            Roy-Vetterli: exp(H(σ̂)), σ̂ = σ/Σσ on Ã.
                               O(min(m,n)² max(m,n)).
            "stable_rank"      trace(G) / λ_max(G). O(n³) top-eigenvalue.
                               Less sensitive to the spectrum's tail than PR.
            "spectral_entropy" exp(H(λ̂)), λ̂ = λ/Σλ on the Gram.  Differs
                               from erank in using σ² weights (= Gram
                               eigenvalues) instead of σ weights. O(n³).
            "col_entropy"      exp(H(col_sums(Ã))), a routing-concentration
                               signal. Not spectral — cheap O(n²) baseline.
            For any non-"pr" choice the incremental path is disabled and
            Ã (or its Gram) is rebuilt per step.
        mass_gamma: Exponent γ for positional masses (default: 1.0).
            front_biased: π_l ∝ 1/(l+1)^γ  →  γ=0 uniform, γ=1 default.
            rear_biased:  π_l ∝ (l+1)^γ    →  γ=0 uniform, γ=1 default.
            Ignored for other mass modes.
        head_selection: How many heads to use per layer (default: "argmax").
            "argmax" — top-1 head (original behavior).
            "topk"   — top-k heads by pass-1 score (uses head_top_k).
            "all"    — every head.
        head_top_k: k for head_selection="topk" (default: 4). Ignored otherwise.
        head_aggregation: How to combine selected heads' signals per layer
            (default: "mean"). Ignored when head_selection="argmax".
            "mean"     — simple average of spectral_weights signals.
            "weighted" — softmax-weighted average of signals, weights derived
                         from pass-1 PR scores with temperature head_weight_temp.
        head_weight_temp: Temperature for weighted aggregation (default: 1.0).
            Large → closer to uniform mean; small → closer to argmax.
            Ignored unless head_aggregation="weighted".
        selection_mode: Which attention matrix to use in pass 1 for head
            selection (default: "square" for backward compatibility).
            "square"      — generated-only square attention (original).
            "rectangular" — (T, P+T) rectangular slice.
            "full_square" — (P+T, P+T) block matrix with prefill.
            "match"       — mirror attn_mode.
            The signal computation (pass 2) always uses attn_mode; this
            parameter only affects which heads are chosen.
    """

    def __init__(
        self,
        alpha: Optional[float] = None,
        n_layers: Optional[int] = None,
        n_heads: Optional[int] = None,
        use_entropy: bool = False,
        instruct: bool = False,
        mass: str = "uniform",
        attn_mode: str = "auto",
        rank_estimator: str = "pr",
        mass_gamma: float = 1.0,
        head_selection: str = "argmax",
        head_top_k: int = 4,
        head_aggregation: str = "mean",
        head_weight_temp: float = 1.0,
        selection_mode: str = "square",
        head_selection_weighting: str = "euclidean",
        renormalize_rows: bool = False,
        sink_epsilon: float = 0.05,
        sink_top_k: int = 4,
    ):
        if mass not in VALID_MASSES:
            raise ValueError(f"mass must be one of {VALID_MASSES}, got {mass!r}")
        if attn_mode not in VALID_ATTN_MODES:
            raise ValueError(
                f"attn_mode must be one of {VALID_ATTN_MODES}, got {attn_mode!r}"
            )
        if rank_estimator not in VALID_RANK_ESTIMATORS:
            raise ValueError(
                f"rank_estimator must be one of {VALID_RANK_ESTIMATORS}, got {rank_estimator!r}"
            )
        if rank_estimator == "pr_wc_centered" and mass != "attn_mass":
            raise ValueError(
                f"rank_estimator='pr_wc_centered' assumes mass='attn_mass' "
                f"(it hardcodes the gram/n rescaling and subtracts the known "
                f"leading eigenvalue lambda_1 = n_rows exactly, i.e. lambda_1(G)/n = 1, "
                f"which only holds for the attn_mass row-stochastic Gram). "
                f"Got mass={mass!r}. Use rank_estimator='pr' with mass={mass!r} instead, "
                f"or generalize the function to subtract the actual leading eigenvalue."
            )
        if head_selection not in VALID_HEAD_SELECTIONS:
            raise ValueError(
                f"head_selection must be one of {VALID_HEAD_SELECTIONS}, got {head_selection!r}"
            )
        if head_aggregation not in VALID_HEAD_AGGREGATIONS:
            raise ValueError(
                f"head_aggregation must be one of {VALID_HEAD_AGGREGATIONS}, got {head_aggregation!r}"
            )
        if selection_mode not in VALID_SELECTION_MODES:
            raise ValueError(
                f"selection_mode must be one of {VALID_SELECTION_MODES}, got {selection_mode!r}"
            )
        if head_selection_weighting not in VALID_SELECTION_WEIGHTINGS:
            raise ValueError(
                f"head_selection_weighting must be one of {VALID_SELECTION_WEIGHTINGS}, "
                f"got {head_selection_weighting!r}"
            )
        if (
            head_selection_weighting == "centered_intrinsic"
            and selection_mode == "square"
        ):
            raise ValueError(
                "head_selection_weighting='centered_intrinsic' assumes row-stochastic "
                "rows (lambda_1 = n_rows exactly), which only holds for "
                "selection_mode in {'rectangular', 'full_square'} -- the gen-only "
                "'square' block's rows sum to well under 1 on average (see the P3 "
                "row-sum check). Use selection_mode='full_square' or 'rectangular'."
            )
        if head_top_k < 1:
            raise ValueError(f"head_top_k must be >= 1, got {head_top_k}")
        if head_weight_temp <= 0:
            raise ValueError(f"head_weight_temp must be > 0, got {head_weight_temp}")

        dependencies = ["attention_all", "greedy_log_likelihoods"]
        if use_entropy:
            dependencies.append("entropy")
        dependencies.extend(["greedy_tokens", "input_tokens"])
        super().__init__(dependencies, "sequence")

        self.use_entropy = use_entropy
        self.instruct = instruct
        self.mass = mass
        self.attn_mode = attn_mode
        self.rank_estimator = rank_estimator

        if rank_estimator == "delta_mix":
            self._rank_fn = delta_mix
            self._rank_takes_gram = False
        elif rank_estimator in _GRAM_ESTIMATORS:
            self._rank_fn = _GRAM_ESTIMATORS[rank_estimator]
            self._rank_takes_gram = True
        else:
            self._rank_fn = _A_MATRIX_ESTIMATORS[rank_estimator]
            self._rank_takes_gram = False
        self.mass_gamma = mass_gamma
        self.head_selection = head_selection
        self.head_top_k = head_top_k
        self.head_aggregation = head_aggregation
        self.head_weight_temp = head_weight_temp
        self.selection_mode = selection_mode
        self.head_selection_weighting = head_selection_weighting
        self.renormalize_rows = renormalize_rows
        self.alpha = alpha if alpha is not None else self.get_alpha()

        self.n_layers = n_layers
        self.n_heads = n_heads

        # Cache for the per-vocab frequency proxy vector (unembed_bias / unembed_norm).
        # Computed lazily on the first __call__ once we have access to the model.
        self._vocab_freq_weights: Optional[np.ndarray] = None

        if self.n_layers is not None:
            self.layers = list(
                range(self.n_layers // 3, int(np.ceil(self.n_layers / 3 * 2) + 1))
            )
        else:
            self.layers = None
        if not (0.0 < sink_epsilon < 1.0):
            raise ValueError(f"sink_epsilon must be in (0, 1), got {sink_epsilon}")
        if sink_top_k < 1:
            raise ValueError(f"sink_top_k must be >= 1, got {sink_top_k}")
        self.sink_epsilon = sink_epsilon
        self.sink_top_k = sink_top_k

    def __str__(self) -> str:
        method_desc = " (entropy)" if self.use_entropy else ""
        if self.mass == "uniform":
            mass_desc = ""
        elif self.mass == "sink_smooth":
            mass_desc = f", mass=sink_smooth(ε={self.sink_epsilon})"
        elif self.mass == "sink_topk":
            mass_desc = f", mass=sink_topk(k={self.sink_top_k},ε={self.sink_epsilon})"
        else:
            mass_desc = f", mass={self.mass}"

        mode_desc = f", mode={self.attn_mode}" if self.attn_mode != "auto" else ""
        rank_desc = (
            f", rank={self.rank_estimator}" if self.rank_estimator != "pr" else ""
        )
        gamma_desc = f", g={self.mass_gamma}" if self.mass_gamma != 1.0 else ""
        if self.head_selection == "argmax":
            head_desc = ""
        elif self.head_selection == "topk":
            head_desc = f", heads=top{self.head_top_k}/{self.head_aggregation}"
        else:  # all
            head_desc = f", heads=all/{self.head_aggregation}"
        if (
            self.head_selection != "argmax"
            and self.head_aggregation == "weighted"
            and self.head_weight_temp != 1.0
        ):
            head_desc += f"@T{self.head_weight_temp}"
        sel_desc = (
            f", sel={self.selection_mode}" if self.selection_mode != "square" else ""
        )
        weight_desc = (
            f", selw={self.head_selection_weighting}"
            if self.head_selection_weighting != "euclidean"
            else ""
        )
        renorm_desc = ", renorm_rows" if self.renormalize_rows else ""
        return f"SpectralRAUQ(a={self.alpha}{mass_desc}{mode_desc}{rank_desc}{gamma_desc}{head_desc}{sel_desc}{weight_desc}{renorm_desc}){method_desc}"

    def get_alpha(self) -> float:
        if self.instruct:
            return 0.9 if self.use_entropy else 0.5
        return 0.8 if self.use_entropy else 0.2

    def _compute_vocab_freq_weights(self, model_obj) -> np.ndarray:
        """
        Build a per-vocab "frequency proxy" w[t] from the model's output head.

        mass='unembed_bias': w ∝ exp(b_U).  The LM head bias is the model's
            learned unigram log-prob correction.  Requires a bias term —
            LLaMA / Mistral / most recent LLMs do NOT have one; use
            'unembed_norm' for those architectures.

        mass='unembed_norm': w ∝ exp(-‖W_U[t]‖).  Always available.
            Smaller row norm → higher mass (empirically correlated with
            frequency for tied-embedding models, but treat as a heuristic).

        Result is cached on the estimator instance; computed once per object.
        PR is scale-invariant, so no normalisation is needed here.
        """
        if self._vocab_freq_weights is not None:
            return self._vocab_freq_weights

        output_emb = model_obj.get_output_embeddings()
        if output_emb is None:
            raise ValueError(
                f"mass={self.mass!r} requires model.get_output_embeddings() to "
                f"return a module, but got None."
            )

        if self.mass == "unembed_bias":
            bias = getattr(output_emb, "bias", None)
            if bias is None:
                raise ValueError(
                    f"mass='unembed_bias' requires the LM head to have a bias "
                    f"term, but {type(output_emb).__name__} has none. "
                    f"Most modern LLMs (LLaMA, Mistral, Qwen, ...) don't -- "
                    f"use mass='unembed_norm' instead."
                )
            b_U = bias.detach().cpu().float().numpy()
            w = np.exp(b_U - b_U.max())
        elif self.mass == "unembed_norm":
            W_U = output_emb.weight.detach().cpu().float().numpy()
            norms = np.linalg.norm(W_U, axis=1)
            w = np.exp(-(norms - norms.min()))
        elif self.mass == "unembed_pc":
            W_U = output_emb.weight.detach().cpu().float().numpy()
            # Center: the top PC of an uncentered matrix is dominated by the mean,
            # which would just give us "average row" rather than the frequency axis.
            W_centered = W_U - W_U.mean(axis=0, keepdims=True)
            # Top right singular vector via a thin SVD on the (vocab, d) matrix.
            # We only need v_1, so use full_matrices=False; cost is O(vocab · d²).
            _, _, Vt = np.linalg.svd(W_centered, full_matrices=False)
            v1 = Vt[0]  # (d,)
            proj = W_centered @ v1  # (vocab,)

            # Sign convention: choose the sign so common tokens (low-id BPE merges,
            # which are typically frequent) get positive projection. This is a
            # heuristic — for a robust check, correlate with token id and flip
            # if needed.
            if np.corrcoef(proj, np.arange(len(proj)))[0, 1] > 0:
                proj = -proj
            # Map projection to a positive mass via exp, with a temperature so the
            # weighting isn't degenerate. PR is scale-invariant so we don't normalise.
            proj = proj - proj.max()  # numerical stability
            w = np.exp(proj)
        else:
            raise AssertionError(
                f"_compute_vocab_freq_weights called with non-vocab mass {self.mass!r}"
            )

        self._vocab_freq_weights = w
        return w

    # ------------------------------------------------------------------
    # Mass helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_pi(
        mode: str,
        size: int,
        token_ids: Optional[np.ndarray] = None,
        attention_prefix: Optional[np.ndarray] = None,
        context_token_ids: Optional[np.ndarray] = None,
        query_row: Optional[np.ndarray] = None,
        gamma: float = 1.0,
        sink_epsilon: float = 0.05,
        sink_top_k: int = 4,
        vocab_freq_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute π of the given size (used for dynamic masses at each step)."""
        if mode == "uniform":
            pi = np.ones(size)
        elif mode == "front_biased":
            pi = 1.0 / (np.arange(size) + 1) ** gamma
        elif mode == "rear_biased":
            pi = (np.arange(size) + 1.0) ** gamma
        elif mode == "attn_mass":
            col_sums = attention_prefix.sum(axis=0)
            pi = col_sums + 1e-10
        elif mode == "row_attn_mass":
            pi = query_row + 1e-10
        elif mode in ("zipf", "zipf_inv"):
            freq_source = (
                context_token_ids if context_token_ids is not None else token_ids[:size]
            )
            counts = Counter(freq_source.tolist())
            freqs = np.array(
                [counts[int(tid)] for tid in token_ids[:size]], dtype=float
            )
            pi = freqs if mode == "zipf" else 1.0 / (freqs + 1e-10)
        elif mode == "sink_smooth":
            # Smoothed sink: π_0 = 1-ε, π_l = ε/(size-1) for l>0.
            # In L²(π), concentrating mass on the sink downweights it in the
            # spectral analysis (tokens with large π divide down in the Gram).
            # ε → 0 makes the sink quotient sharper (riskier numerically);
            # ε → (size-1)/size gives uniform.
            if size == 1:
                pi = np.ones(1)
            else:
                pi = np.full(size, sink_epsilon / (size - 1), dtype=float)
                pi[0] = 1.0 - sink_epsilon
        elif mode == "sink_topk":
            # Adaptive multi-sink: identify top-k tokens by column sum
            # (the empirical sinks), concentrate mass on them.
            # k=1 recovers sink_smooth behavior when BOS is the sole sink;
            # k>1 handles multi-sink settings (e.g. chat models with
            # BOS + first-user-turn token both acting as sinks).
            col_sums = attention_prefix.sum(axis=0)
            k_eff = min(sink_top_k, size)
            if k_eff >= size:
                pi = np.ones(size) / size
            else:
                top_idx = np.argpartition(col_sums, -k_eff)[-k_eff:]
                pi = np.full(size, sink_epsilon / (size - k_eff), dtype=float)
                pi[top_idx] = (1.0 - sink_epsilon) / k_eff

        elif mode in ("unembed_bias", "unembed_norm"):
            if vocab_freq_weights is None:
                raise ValueError(f"mass={mode!r} requires vocab_freq_weights")
            if token_ids is None:
                raise ValueError(f"mass={mode!r} requires token_ids")
            pi = vocab_freq_weights[token_ids[:size]] + 1e-10
        else:
            raise ValueError(f"Unknown mass mode: {mode}")
        return pi / (pi.sum() + 1e-10)

    @staticmethod
    def _make_inv_sq_pi_static(
        mode: str,
        max_size: int,
        token_ids: Optional[np.ndarray] = None,
        context_token_ids: Optional[np.ndarray] = None,
        gamma: float = 1.0,
        sink_epsilon: float = 0.05,
        vocab_freq_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Precompute 1/π for the full key range for static masses.

        Returns inv_sq_pi[l] = 1/π_l (unnormalized is fine — PR is scale-invariant).
        """
        if mode == "uniform":
            return np.ones(max_size)
        elif mode == "front_biased":
            # π_l ∝ 1/(l+1)^γ  →  1/π_l ∝ (l+1)^γ
            return (np.arange(max_size, dtype=float) + 1.0) ** gamma
        elif mode == "rear_biased":
            # π_l ∝ (l+1)^γ  →  1/π_l ∝ 1/(l+1)^γ
            return 1.0 / (np.arange(max_size, dtype=float) + 1.0) ** gamma
        elif mode in ("zipf", "zipf_inv"):
            freq_source = (
                context_token_ids
                if context_token_ids is not None
                else token_ids[:max_size]
            )
            counts = Counter(freq_source.tolist())
            freqs = np.array(
                [counts[int(tid)] for tid in token_ids[:max_size]], dtype=float
            )
            if mode == "zipf":
                # π_l ∝ freq  →  1/π_l ∝ 1/freq
                return 1.0 / (freqs + 1e-10)
            else:
                # π_l ∝ 1/freq  →  1/π_l ∝ freq
                return freqs + 1e-10
        elif mode == "sink_smooth":
            # 1/π with π as in the sink_smooth branch of _make_pi.
            # Scale-invariance of PR means we don't need to normalize.
            if max_size == 1:
                return np.ones(1)
            inv_pi = np.full(
                max_size, (max_size - 1) / (sink_epsilon + 1e-12), dtype=float
            )
            inv_pi[0] = 1.0 / (1.0 - sink_epsilon + 1e-12)
            return inv_pi

        elif mode in ("unembed_bias", "unembed_norm", "unembed_pc"):
            if vocab_freq_weights is None:
                raise ValueError(f"mass={mode!r} requires vocab_freq_weights")
            if token_ids is None:
                raise ValueError(f"mass={mode!r} requires token_ids")
            return 1.0 / (vocab_freq_weights[token_ids[:max_size]] + 1e-10)

        else:
            raise ValueError(
                f"_make_inv_sq_pi_static called with dynamic mass {mode!r}"
            )

    # ------------------------------------------------------------------
    # Spectral weight extraction
    # ------------------------------------------------------------------

    def _extract_spectral_weights(
        self,
        attention_map: np.ndarray,
        token_ids: Optional[np.ndarray] = None,
        context_token_ids: Optional[np.ndarray] = None,
        full_attention_map: Optional[np.ndarray] = None,
        full_key_token_ids: Optional[np.ndarray] = None,
        prefill_attention: Optional[np.ndarray] = None,
        vocab_freq_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute fractional rank signal per generation step.

        Returns np.ndarray of shape (gen_len-1,).

        For static masses the computation is incremental (O(j²) per step via
        bordered Gram updates).  For dynamic masses the Gram is rebuilt at each
        step (O(j² × j_cols)) but the O(n³) SVD is replaced by the O(n²) PR.
        """
        seq_len = attention_map.shape[0]
        use_full = full_attention_map is not None
        use_square_full = use_full and prefill_attention is not None
        prompt_len = full_attention_map.shape[1] - seq_len if use_full else 0

        # Incremental bordered-Gram path is only valid when:
        #   (1) the mass is static (precomputable),
        #   (2) the rank estimator is PR (closed-form update of trace + frob²), and
        #   (3) row renormalization is off -- renormalize_rows divides each row by
        #       its own (step-dependent) sum, which the bordered-Gram update does
        #       not account for. Route through the fresh-Gram path instead, which
        #       rebuilds A explicitly at each step and applies renormalization there.
        if (
            self.mass in _STATIC_MASSES
            and self.rank_estimator in _INCREMENTAL_RANK_ESTIMATORS
            and not self.renormalize_rows
        ):
            return self._incremental_weights(
                attention_map,
                token_ids,
                context_token_ids,
                full_attention_map,
                full_key_token_ids,
                prefill_attention,
                seq_len,
                prompt_len,
                use_full,
                use_square_full,
                vocab_freq_weights,
            )
        else:
            return self._fresh_weights(
                attention_map,
                token_ids,
                context_token_ids,
                full_attention_map,
                full_key_token_ids,
                prefill_attention,
                seq_len,
                prompt_len,
                use_full,
                use_square_full,
                vocab_freq_weights,
            )

    # --- Incremental path (static masses) ---

    def _incremental_weights(
        self,
        attention_map: np.ndarray,
        token_ids: Optional[np.ndarray],
        context_token_ids: Optional[np.ndarray],
        full_attention_map: Optional[np.ndarray],
        full_key_token_ids: Optional[np.ndarray],
        prefill_attention: Optional[np.ndarray],
        seq_len: int,
        prompt_len: int,
        use_full: bool,
        use_square_full: bool,
        vocab_freq_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Incremental bordered-Gram PR computation for static mass measures.

        At each step j, the causal structure ensures G_j has the bordered form:

            G_j = [ G_{j-1}  |  cross_j ]
                  [ cross_jᵀ |  c_j     ]

        where cross_j = A_pi_prev @ a_pi_j[:prev_cols]  (matrix-vector, O(j²))
        and   c_j     = ‖a_pi_j‖²                       (O(j)).

        PR is then updated as:
            trace_j  = trace_{j-1} + c_j
            frob²_j  = frob²_{j-1} + 2‖cross_j‖² + c_j²
            PR_j     = trace_j² / frob²_j
        """
        max_key = (prompt_len + seq_len) if use_full else seq_len
        t_ids = full_key_token_ids if use_full else token_ids
        inv_sq_pi = self._make_inv_sq_pi_static(
            self.mass,
            max_key,
            token_ids=t_ids,
            context_token_ids=(full_key_token_ids if use_full else context_token_ids),
            gamma=self.mass_gamma,
            sink_epsilon=self.sink_epsilon,
            vocab_freq_weights=vocab_freq_weights,
        )
        inv_sqrt_pi = np.sqrt(inv_sq_pi)  # used to scale attention rows once

        # Row mass μ = π (same measure on both dims).
        # Since inv_sq_pi = 1/π (unnormalized), μ = 1/inv_sq_pi.
        # For "uniform": inv_sq_pi = ones → μ = ones (all rows equal).
        # For "rear_biased": inv_sq_pi = 1/(i+1) → μ = (i+1) (gen tokens upweighted).
        # PR is scale-invariant so unnormalized μ is fine; Σμ_i is the
        # effective dimension for the fractional rank.
        if use_square_full:
            # Rows index the full (P+T) sequence — μ has length P+T.
            mu_full = 1.0 / inv_sq_pi  # length = P + T
            sqrt_mu = np.sqrt(mu_full)
            cumsum_mu = np.cumsum(mu_full)
        elif use_full:
            # Rectangular: rows are gen tokens (positions P..P+T-1 in the
            # full sequence).  Take the generation-portion of μ.
            mu_gen = 1.0 / inv_sq_pi[prompt_len:]  # length = T
            sqrt_mu = np.sqrt(mu_gen)
            cumsum_mu = np.cumsum(mu_gen)
        else:
            # Square: rows index the T generation tokens.
            mu_gen = 1.0 / inv_sq_pi  # length = T
            sqrt_mu = np.sqrt(mu_gen)
            cumsum_mu = np.cumsum(mu_gen)

        if use_square_full:
            return self._incremental_full_square(
                attention_map,
                full_attention_map,
                prefill_attention,
                seq_len,
                prompt_len,
                inv_sq_pi,
                inv_sqrt_pi,
                sqrt_mu,
                cumsum_mu,
            )
        elif use_full:
            return self._incremental_rectangular(
                full_attention_map,
                seq_len,
                prompt_len,
                inv_sq_pi,
                sqrt_mu,
                cumsum_mu,
            )
        else:
            return self._incremental_square(
                attention_map,
                seq_len,
                inv_sq_pi,
                sqrt_mu,
                cumsum_mu,
            )

    def _incremental_square(
        self,
        attention_map: np.ndarray,
        seq_len: int,
        inv_sq_pi: np.ndarray,
        sqrt_mu: np.ndarray,
        cumsum_mu: np.ndarray,
    ) -> np.ndarray:
        """Square mode incremental PR.  A is (T, T) causal.

        Gram with row mass: G_μ[i,j] = √μ_i √μ_j Σ_l A[i,l] A[j,l] / π_l.
        """
        # Init: j=0, matrix is 1×1
        c = sqrt_mu[0] ** 2 * attention_map[0, 0] ** 2 * inv_sq_pi[0]
        trace_val = c
        frob_sq_val = c * c

        spectral_weights = []
        for j in range(1, seq_len):
            a_j = attention_map[j, : j + 1]
            # Unweighted cross (without row mass): raw_cross[i] = Σ_l A[i,l] A[j,l] / π_l
            a_weighted = a_j[:j] * inv_sq_pi[:j]  # O(j)
            raw_cross = attention_map[:j, :j] @ a_weighted  # O(j²)
            # Apply row mass: cross[i] = √μ_i · √μ_j · raw_cross[i]
            cross = sqrt_mu[:j] * sqrt_mu[j] * raw_cross  # O(j)
            c = sqrt_mu[j] ** 2 * np.dot(a_j * a_j, inv_sq_pi[: j + 1])  # O(j)

            trace_val += c
            frob_sq_val += 2.0 * np.dot(cross, cross) + c * c
            pr = trace_val * trace_val / (frob_sq_val + 1e-10)
            spectral_weights.append(pr / cumsum_mu[j])

        return np.array(spectral_weights)

    def _incremental_rectangular(
        self,
        full_attention_map: np.ndarray,
        seq_len: int,
        prompt_len: int,
        inv_sq_pi: np.ndarray,
        sqrt_mu: np.ndarray,
        cumsum_mu: np.ndarray,
    ) -> np.ndarray:
        """
        Rectangular mode incremental PR.
        A is (T, P+T): generated rows, all (prompt+generated) key columns.
        Row mass μ is length T (generation tokens only).
        """
        # Init: j=0, one generated row
        a0 = full_attention_map[0, : prompt_len + 1]
        c = sqrt_mu[0] ** 2 * np.dot(a0 * a0, inv_sq_pi[: prompt_len + 1])
        trace_val = c
        frob_sq_val = c * c

        spectral_weights = []
        for j in range(1, seq_len):
            n_keys = prompt_len + j
            a_j = full_attention_map[j, : prompt_len + j + 1]
            a_weighted = a_j[:n_keys] * inv_sq_pi[:n_keys]  # O(P+j)
            raw_cross = full_attention_map[:j, :n_keys] @ a_weighted  # O(j(P+j))
            cross = sqrt_mu[:j] * sqrt_mu[j] * raw_cross  # O(j)
            c = sqrt_mu[j] ** 2 * np.dot(a_j * a_j, inv_sq_pi[: prompt_len + j + 1])

            trace_val += c
            frob_sq_val += 2.0 * np.dot(cross, cross) + c * c
            pr = trace_val * trace_val / (frob_sq_val + 1e-10)
            spectral_weights.append(pr / cumsum_mu[j])

        return np.array(spectral_weights)

    def _incremental_full_square(
        self,
        attention_map: np.ndarray,
        full_attention_map: np.ndarray,
        prefill_attention: np.ndarray,
        seq_len: int,
        prompt_len: int,
        inv_sq_pi: np.ndarray,
        inv_sqrt_pi: np.ndarray,
        sqrt_mu: np.ndarray,
        cumsum_mu: np.ndarray,
    ) -> np.ndarray:
        """
        Full-square mode incremental PR with consistent mass π on rows and cols.

        The full square matrix at step j is (P+j+1) × (P+j+1):

            A_j = [ A_pp  |  0               ]   (prompt_len rows)
                  [ A_gp  |  A_gg[:j+1,:j+1] ]   (j+1 generated rows)

        The weighted Gram is G_π = Π^{1/2} A Π^{-1} Aᵀ Π^{1/2}, where Π acts
        on both rows and columns.  Row i is scaled by √μ_i = √π_i (same mass),
        and the fractional rank normalizes by Σμ_i.

        sqrt_mu has length (P+T), cumsum_mu has length (P+T).
        sqrt_mu[:P] covers prompt rows, sqrt_mu[P:] covers gen rows.
        """
        P = prompt_len

        # Precompute column-scaled prompt block, then apply row mass
        # prefill_pi_mu[i,l] = √μ_i · prefill_attention[i,l] · inv_sqrt_pi[l]
        prefill_pi = prefill_attention * inv_sqrt_pi[:P][None, :]  # (P, P)
        prefill_pi_mu = prefill_pi * sqrt_mu[:P, None]  # row mass
        G_pp = prefill_pi_mu @ prefill_pi_mu.T  # (P, P)
        trace_pp = np.trace(G_pp)
        frob_sq_pp = np.sum(G_pp * G_pp)

        # Column-scaled gen-attention: full_attn_pi[k,l] = full_attention_map[k,l] · inv_sqrt_pi[l]
        full_attn_pi = (
            full_attention_map * inv_sqrt_pi[: P + seq_len][None, :]
        )  # (T, P+T)

        # Init: j=0, first generated token (row index P in full matrix)
        a0_pi = full_attn_pi[0, : P + 1]  # column-scaled
        a0_pi_mu = a0_pi * sqrt_mu[P]  # + row mass
        # Cross with prompt rows
        cross_0 = prefill_pi_mu @ a0_pi[:P] * sqrt_mu[P]  # (P,)
        c0 = np.dot(a0_pi_mu, a0_pi_mu)
        trace_val = trace_pp + c0
        frob_sq_val = frob_sq_pp + 2.0 * np.dot(cross_0, cross_0) + c0 * c0

        spectral_weights = []
        for j in range(1, seq_len):
            n_keys = P + j
            row_idx = P + j  # index in the full (P+T) sequence
            a_j_pi = full_attn_pi[j, : P + j + 1]  # column-scaled

            # Cross from prompt rows (P of them):
            #   cross_p[i] = √μ_i · √μ_{row_idx} · (prefill_pi[i,:P] · a_j_pi[:P])
            cross_p = prefill_pi_mu @ a_j_pi[:P] * sqrt_mu[row_idx]  # (P,)
            # Cross from previously generated rows (j of them, row indices P..P+j-1):
            raw_cross_g = full_attn_pi[:j, :n_keys] @ a_j_pi[:n_keys]  # (j,)
            cross_g = sqrt_mu[P : P + j] * sqrt_mu[row_idx] * raw_cross_g  # (j,)
            cross = np.concatenate([cross_p, cross_g])  # (P+j,)

            c = sqrt_mu[row_idx] ** 2 * np.dot(a_j_pi, a_j_pi)

            trace_val += c
            frob_sq_val += 2.0 * np.dot(cross, cross) + c * c
            pr = trace_val * trace_val / (frob_sq_val + 1e-10)
            spectral_weights.append(pr / cumsum_mu[row_idx])

        return np.array(spectral_weights)

    # --- Fresh-Gram path (dynamic masses or non-incremental rank estimators) ---

    def _fresh_weights(
        self,
        attention_map: np.ndarray,
        token_ids: Optional[np.ndarray],
        context_token_ids: Optional[np.ndarray],
        full_attention_map: Optional[np.ndarray],
        full_key_token_ids: Optional[np.ndarray],
        prefill_attention: Optional[np.ndarray],
        seq_len: int,
        prompt_len: int,
        use_full: bool,
        use_square_full: bool,
        vocab_freq_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Fresh-Gram path: rebuild Ã at each step and apply the rank estimator.

        Used when the mass is dynamic (attn_mass, row_attn_mass) or when the
        rank estimator has no incremental update (everything except PR).

        Row mass handling:
          - Static mass: μ_i = π_i is precomputable, so we apply √μ to rows
            of Ã for consistency with the incremental path (same G as the
            incremental PR up to the estimator).
          - Dynamic mass: row mass stays uniform (we can't define a static
            row mass from a column mass that depends on the current slice).

        Normalization: the fractional rank is rank(Ã)/Σμ_i when row mass is
        nontrivial, rank(Ã)/n_rows otherwise.  This matches the incremental
        path's `cumsum_mu[row_idx]` normalization.
        """
        is_static_mass = self.mass in _STATIC_MASSES

        # Precompute static row mass μ once (PR-incremental style).  For
        # dynamic masses μ is effectively uniform -> sqrt_mu = 1, cumsum = row count.
        if is_static_mass:
            max_key = (prompt_len + seq_len) if use_full else seq_len
            t_ids_static = full_key_token_ids if use_full else token_ids
            ctx_ids_static = full_key_token_ids if use_full else context_token_ids
            inv_sq_pi_static = self._make_inv_sq_pi_static(
                self.mass,
                max_key,
                token_ids=t_ids_static,
                context_token_ids=ctx_ids_static,
                gamma=self.mass_gamma,
                vocab_freq_weights=vocab_freq_weights,
            )
            mu_full = 1.0 / inv_sq_pi_static
            # Row indices we'll write into at step j:
            #   square:        rows 0..j                (length = seq_len = T)
            #   rectangular:   gen rows 0..j → use μ[prompt_len:]
            #   full_square:   rows 0..P+j             (length = P+T)
            if use_square_full:
                sqrt_mu_static = np.sqrt(mu_full)  # length P+T
                cumsum_mu_static = np.cumsum(mu_full)
            elif use_full:
                sqrt_mu_static = np.sqrt(mu_full[prompt_len:])  # length T
                cumsum_mu_static = np.cumsum(mu_full[prompt_len:])
            else:
                sqrt_mu_static = np.sqrt(mu_full)  # length T
                cumsum_mu_static = np.cumsum(mu_full)
        else:
            sqrt_mu_static = None
            cumsum_mu_static = None

        spectral_weights = []
        for j in range(1, seq_len):
            if use_square_full:
                n_rows = prompt_len + j + 1
                A = np.zeros((n_rows, n_rows))
                A[:prompt_len, :prompt_len] = prefill_attention
                A[prompt_len:, : prompt_len + j + 1] = full_attention_map[
                    : j + 1, : prompt_len + j + 1
                ]
            elif use_full:
                A = full_attention_map[: j + 1, : prompt_len + j + 1]
            else:
                A = attention_map[: j + 1, : j + 1]

            if self.renormalize_rows:
                # B_tilde = diag(B 1)^{-1} B: renormalize each row to sum to 1.
                # Mainly relevant for attn_mode="square", whose rows only cover
                # already-generated tokens and sum to well under 1 on average
                # (see the P3 row-sum check). Rows with ~0 mass (e.g. row 0,
                # which has no previous generated tokens to attend to) are left
                # untouched rather than divided by ~0.
                row_sums = A.sum(axis=1, keepdims=True)
                nonzero = row_sums[:, 0] > 1e-10
                A = A.copy()
                A[nonzero] = A[nonzero] / row_sums[nonzero]

            key_size = A.shape[1]
            n_rows = A.shape[0]
            if self.rank_estimator == "delta_mix":
                spectral_weights.append(delta_mix(A))
                continue
            if self.mass == "row_attn_mass":
                current_row = (prompt_len + j) if use_square_full else j
                query_row = A[current_row]
            else:
                query_row = None

            if self.mass in ("zipf", "zipf_inv", "unembed_bias", "unembed_norm"):
                t_ids = full_key_token_ids[:key_size] if use_full else token_ids
                ctx_ids = full_key_token_ids if use_full else context_token_ids
            else:
                t_ids = token_ids
                ctx_ids = context_token_ids

            pi = self._make_pi(
                self.mass,
                key_size,
                token_ids=t_ids,
                attention_prefix=A,
                context_token_ids=ctx_ids,
                query_row=query_row,
                gamma=self.mass_gamma,
                sink_epsilon=self.sink_epsilon,
                sink_top_k=self.sink_top_k,
                vocab_freq_weights=vocab_freq_weights,
            )
            inv_sqrt_pi = 1.0 / np.sqrt(pi + 1e-10)
            A_pi = A * inv_sqrt_pi[None, :]  # column mass applied

            # Apply static row mass so the Gram matches the incremental path.
            if is_static_mass:
                A_pi = A_pi * sqrt_mu_static[:n_rows, None]
                denom = cumsum_mu_static[n_rows - 1] + 1e-10
            else:
                denom = n_rows  # uniform row mass

            # Dispatch on whether the estimator takes the Gram or Ã directly.
            if self._rank_takes_gram:
                gram = A_pi @ A_pi.T
                rank_val = self._rank_fn(gram)
            else:
                rank_val = self._rank_fn(A_pi)

            if self.rank_estimator in _PRE_NORMALIZED_ESTIMATORS:
                spectral_weights.append(rank_val)
            else:
                spectral_weights.append(rank_val / denom)

        return np.array(spectral_weights)

    # ------------------------------------------------------------------
    # Confidence recurrence (unchanged from RAUQ)
    # ------------------------------------------------------------------

    def _calculate_confidence_scores(
        self,
        log_probabilities: np.ndarray,
        spectral_weights: np.ndarray,
    ) -> List[float]:
        confidence_scores = [np.exp(log_probabilities[0])]
        for j in range(1, len(log_probabilities)):
            confidence = (
                self.alpha * np.exp(log_probabilities[j])
                + (1 - self.alpha) * spectral_weights[j - 1] * confidence_scores[-1]
            )
            confidence_scores.append(confidence)
        return confidence_scores

    # ------------------------------------------------------------------
    # Head selection scoring (pass 1)
    # ------------------------------------------------------------------

    def _compute_head_selection_scores(
        self,
        A_layer: np.ndarray,
        A_layer_full: Optional[np.ndarray],
        prefill_layer: Optional[np.ndarray],
        selection_mode: str,
    ) -> np.ndarray:
        """
        Compute an unweighted PR score per head at the final step, batched.

        Args:
            A_layer:      (n_heads, T, T) — gen-only square attention for this layer.
            A_layer_full: (n_heads, T, P+T) — gen rows with prompt+gen keys, or None.
            prefill_layer:(n_heads, P, P) — prompt × prompt attention block, or None.
            selection_mode: "square", "rectangular", or "full_square".

        Returns:
            head_scores: (n_heads,) PR score per head.

        The PR is computed on the unweighted Gram (no mass π). Mass-aware
        selection would require rebuilding per-head, which defeats the purpose
        of pass-1 batching; the current signal computation (pass 2) still
        honors `self.mass`.
        """
        n_heads = A_layer.shape[0]
        if selection_mode == "square":
            M = A_layer  # (n_heads, T, T)
        elif selection_mode == "rectangular":
            if A_layer_full is None:
                raise ValueError("selection_mode='rectangular' requires full attention")
            M = A_layer_full  # (n_heads, T, P+T)
        else:  # full_square
            if A_layer_full is None or prefill_layer is None:
                raise ValueError(
                    "selection_mode='full_square' requires full + prefill attention"
                )
            P = prefill_layer.shape[-1]
            T_plus_keys = A_layer_full.shape[-1]  # = P + T
            # Stitch: M has (n_heads, P+T, P+T) with prefill on top, zero top-right, gen below.
            M = np.zeros((n_heads, T_plus_keys, T_plus_keys), dtype=A_layer_full.dtype)
            M[:, :P, :P] = prefill_layer
            M[:, P:, :] = A_layer_full

        if self.head_selection_weighting == "euclidean":
            # Row Gram for all heads: (n_heads, n_rows, n_rows)
            # For rectangular, n_rows = T, inner dim = P+T.
            grams = M @ M.transpose(0, 2, 1)
            traces = np.einsum("hii->h", grams)
            frob_sqs = np.einsum("hij,hij->h", grams, grams)
            return traces * traces / (frob_sqs + 1e-10)

        # intrinsic / centered_intrinsic: mass-weight columns by per-head attn_mass
        # (column sums of M, i.e. the same pi = attn_mass definition used elsewhere,
        # just computed per head at this single final step instead of per (l,h) call).
        n_rows = M.shape[1]
        pi_h = M.sum(axis=1) + 1e-10  # (n_heads, n_cols)
        pi_h = pi_h / pi_h.sum(axis=1, keepdims=True)
        B = M * (1.0 / np.sqrt(pi_h))[:, None, :]
        grams = B @ B.transpose(0, 2, 1)
        traces = np.einsum("hii->h", grams)
        frob_sqs = np.einsum("hij,hij->h", grams, grams)

        if self.head_selection_weighting == "intrinsic":
            return traces * traces / (frob_sqs + 1e-10)

        # centered_intrinsic: subtract the known leading eigenvalue lambda_1 = n_rows
        # (valid because M's rows are row-stochastic in {rectangular, full_square} —
        # see participation_ratio_centered_from_attn_mass_gram for the same derivation).
        # Only the head ranking matters here (argmax/topk), so we skip the final
        # /(n_rows - 1) normalization -- it's a constant across heads in a layer.
        traces_n = traces / n_rows
        frob_sqs_n = frob_sqs / (n_rows * n_rows)
        traces_c = traces_n - 1.0
        frob_sqs_c = frob_sqs_n - 1.0
        valid = (traces_c > 1e-10) & (frob_sqs_c > 1e-10)
        scores = np.zeros_like(traces_c)
        scores[valid] = (traces_c[valid] ** 2) / (frob_sqs_c[valid] + 1e-10)
        return scores

    def _select_heads(self, head_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pick heads and derive aggregation weights from PR scores.

        Returns:
            heads:   (k,) int array of selected head indices.
            weights: (k,) float array summing to 1 (aggregation weights).
        """
        n_heads = head_scores.shape[0]
        if self.head_selection == "argmax":
            heads = np.array([int(head_scores.argmax())], dtype=int)
        elif self.head_selection == "topk":
            k = min(self.head_top_k, n_heads)
            # argpartition is O(n); then sort the top-k slice for determinism
            top_idx = np.argpartition(head_scores, -k)[-k:]
            heads = top_idx[np.argsort(-head_scores[top_idx])]
        else:  # "all"
            heads = np.arange(n_heads, dtype=int)

        if len(heads) == 1 or self.head_aggregation == "mean":
            weights = np.ones(len(heads)) / len(heads)
        else:  # weighted
            s = head_scores[heads] / self.head_weight_temp
            s = s - s.max()  # numerical stability
            weights = np.exp(s)
            weights = weights / weights.sum()
        return heads, weights

    # ------------------------------------------------------------------
    # Main __call__
    # ------------------------------------------------------------------

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate uncertainty scores for each sequence in a batch.

        Two-pass head selection:
          Pass 1 — batched PR at the final generation step for all heads in
                   the `selection_mode` attention view.  One big matmul per
                   layer.  Picks k heads per layer (argmax / top-k / all).
          Pass 2 — full incremental spectral weights (honoring `attn_mode`
                   and `mass`) for each selected head, then aggregated
                   (mean or softmax-weighted by PR) into one per-layer
                   signal of length seq_len-1.
        """
        if self.n_layers is None:
            self.n_layers = stats["model"].model.config.num_hidden_layers
            self.layers = list(
                range(self.n_layers // 3, int(np.ceil(self.n_layers / 3 * 2) + 1))
            )
        if self.n_heads is None:
            self.n_heads = stats["model"].model.config.num_attention_heads

        if self.mass in ("unembed_bias", "unembed_norm", "unembed_pc"):
            vocab_freq_weights = self._compute_vocab_freq_weights(stats["model"].model)
        else:
            vocab_freq_weights = None

        all_token_ids = [np.array(tids) for tids in stats["greedy_tokens"]]
        all_context_ids = [
            np.concatenate([stats["input_tokens"][i], stats["greedy_tokens"][i]])
            for i in range(len(stats["greedy_tokens"]))
        ]
        all_full_key_ids = all_context_ids

        # ---- Detect available data (independent of attn_mode) ----
        has_full_attn_data = (
            "attention_all_with_prompt" in stats
            and len(stats["attention_all_with_prompt"]) > 0
        )
        has_prefill_data = (
            "attention_prefill" in stats and len(stats["attention_prefill"]) > 0
        )

        # ---- Determine attention mode for signal computation (pass 2) ----
        if self.attn_mode == "auto":
            use_full_attn = has_full_attn_data
            use_prefill = has_prefill_data
        elif self.attn_mode == "square":
            use_full_attn = False
            use_prefill = False
        elif self.attn_mode == "rectangular":
            if not has_full_attn_data:
                raise ValueError(
                    "attn_mode='rectangular' requires 'attention_all_with_prompt' in stats"
                )
            use_full_attn = True
            use_prefill = False
        else:  # full_square
            if not (has_full_attn_data and has_prefill_data):
                raise ValueError(
                    "attn_mode='full_square' requires both "
                    "'attention_all_with_prompt' and 'attention_prefill' in stats"
                )
            use_full_attn = True
            use_prefill = True

        # ---- Resolve selection_mode for head scoring (pass 1) ----
        # Independent of attn_mode: pass 1 can use richer attention views
        # than pass 2 if the data is in stats.
        if self.selection_mode == "match":
            if use_prefill:
                selection_mode = "full_square"
            elif use_full_attn:
                selection_mode = "rectangular"
            else:
                selection_mode = "square"
        else:
            selection_mode = self.selection_mode

        if selection_mode == "rectangular" and not has_full_attn_data:
            raise ValueError(
                "selection_mode='rectangular' requires 'attention_all_with_prompt' in stats"
            )
        if selection_mode == "full_square" and not (
            has_full_attn_data and has_prefill_data
        ):
            raise ValueError(
                "selection_mode='full_square' requires both "
                "'attention_all_with_prompt' and 'attention_prefill' in stats"
            )

        # attn_mode="auto" resolves at runtime based on what's in `stats`, so the
        # estimator's __str__ alone (which omits mode_desc when attn_mode="auto")
        # doesn't tell you what actually ran. Log the resolution explicitly so the
        # exact config behind any "SAUQ (Ours)" number is recoverable from the log.
        logger.info(
            f"{self}: resolved attn_mode={self.attn_mode!r} -> "
            f"use_full_attn={use_full_attn}, use_prefill={use_prefill}, "
            f"selection_mode={selection_mode!r}"
        )

        # Whether pass 1 needs the full/prefill arrays (so we always load them
        # when selection_mode requires, even if attn_mode=square).
        need_full_attn_for_pass1 = selection_mode in ("rectangular", "full_square")
        need_prefill_for_pass1 = selection_mode == "full_square"
        load_full_attn = use_full_attn or need_full_attn_for_pass1
        load_prefill = use_prefill or need_prefill_for_pass1

        # ---- Build per-layer aggregated spectral weights ----
        # After pre-aggregation we only need shape (n_layers_used, seq_len-1),
        # kept as dict[layer] -> np.ndarray(seq_len-1) for sparsity.
        spectral_weights_all: List[Dict[int, np.ndarray]] = []

        for attn_idx, attention_weight in enumerate(stats["attention_all"]):
            reshaped = attention_weight.reshape(
                self.n_layers,
                self.n_heads,
                attention_weight.shape[-2],
                attention_weight.shape[-1],
            )
            seq_len = reshaped.shape[-1]

            if load_full_attn:
                attn_full = stats["attention_all_with_prompt"][attn_idx]
                reshaped_full = attn_full.reshape(
                    self.n_layers,
                    self.n_heads,
                    attn_full.shape[-2],
                    attn_full.shape[-1],
                )
            else:
                reshaped_full = None

            if load_prefill:
                prefill_block = stats["attention_prefill"][attn_idx].reshape(
                    self.n_layers,
                    self.n_heads,
                    stats["attention_prefill"][attn_idx].shape[-2],
                    stats["attention_prefill"][attn_idx].shape[-1],
                )
            else:
                prefill_block = None

            per_layer_signal: Dict[int, np.ndarray] = {}

            for layer in self.layers:
                # ---- Pass 1: head scoring in the chosen selection_mode ----
                A_layer = reshaped[layer]  # (n_heads, T, T)
                A_layer_full = (
                    reshaped_full[layer] if reshaped_full is not None else None
                )
                prefill_layer = (
                    prefill_block[layer] if prefill_block is not None else None
                )

                head_scores = self._compute_head_selection_scores(
                    A_layer,
                    A_layer_full,
                    prefill_layer,
                    selection_mode,
                )
                heads, weights = self._select_heads(head_scores)

                # ---- Pass 2: incremental weights per selected head, aggregated.
                # Pass 2 views are governed by attn_mode (use_full_attn/use_prefill),
                # NOT selection_mode — we deliberately pass None when attn_mode
                # wants a smaller view even if we loaded the larger one for pass 1.
                agg_signal = np.zeros(seq_len - 1)
                for h, w in zip(heads, weights):
                    full_head = (
                        reshaped_full[layer, h]
                        if (use_full_attn and reshaped_full is not None)
                        else None
                    )
                    prefill_head = (
                        prefill_block[layer, h]
                        if (use_prefill and prefill_block is not None)
                        else None
                    )
                    sig = self._extract_spectral_weights(
                        reshaped[layer, h],
                        token_ids=all_token_ids[attn_idx],
                        context_token_ids=all_context_ids[attn_idx],
                        full_attention_map=full_head,
                        full_key_token_ids=all_full_key_ids[attn_idx],
                        prefill_attention=prefill_head,
                        vocab_freq_weights=vocab_freq_weights,
                    )
                    agg_signal += w * sig

                per_layer_signal[layer] = agg_signal

            spectral_weights_all.append(per_layer_signal)

        greedy_log_likelihoods = stats["greedy_log_likelihoods"]

        if self.use_entropy:
            entropy = stats["entropy"]
            vocab_size = len(stats["greedy_log_probs"][0][0])
            max_entropy = np.log(vocab_size)

        uncertainty_scores = []

        for idx in range(len(greedy_log_likelihoods)):
            if self.use_entropy:
                log_probabilities = np.log(max_entropy - np.array(entropy[idx]) + 1e-10)
            else:
                log_probabilities = greedy_log_likelihoods[idx]

            layer_scores = []
            for layer in self.layers:
                confidence_scores = self._calculate_confidence_scores(
                    log_probabilities,
                    spectral_weights_all[idx][layer],
                )
                uncertainty = 1 - np.log(np.clip(confidence_scores, 1e-10, None)).mean()
                layer_scores.append(uncertainty)

            uncertainty_scores.append(np.max(layer_scores))

        return np.array(uncertainty_scores)
