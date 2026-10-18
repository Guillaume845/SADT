"""
Markov stationary distribution of the image-patch attention matrix.

For each image:
  1. Run LLaVA and collect the full attention matrix A ∈ R^{T × n_patches}
     where A[t, j] = attention from generated token t to image patch j
     (summed over target layers, averaged over heads).

  2. Build the patch co-attention matrix:
       M = A^T A  ∈ R^{n_patches × n_patches}
     M[i,j] = sum_t A[t,i] * A[t,j] — how often are patch i and j jointly attended.

  3. Row-normalise M to get a stochastic transition matrix T.

  4. Compute the stationary distribution π via power iteration:
       π T = π  →  iterate π_{n+1} = π_n T until convergence.

  5. Reshape π to a 24×24 grid and overlay as a heatmap on the original image.

Interpretation: patches with high π_j are structurally central attractors of
the attention flow — they are consistently co-attended with the majority of
other patches regardless of which token generates the query.

Output
------
outputs/markov_attention/
  img_<name>_stationary.png   — heatmap overlay per image
  results.json                — per-image statistics
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.sadt.core import (
    SADTConfig,
    generate_with_trace,
)
from tools.sadt.pipeline_utils import collect_images
from experiments.utils import (
    default_config,
    get_img_idx,
    load_model,
    compute_full_attention_matrix,
)


# ---------------------------------------------------------------------------
# Markov chain construction and stationary distribution
# ---------------------------------------------------------------------------

def build_transition_matrix(A: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Row-normalised co-attention matrix.

    A : (T, n_patches) — full attention matrix, rows sum to 1.
    Returns T : (n_patches, n_patches) row-stochastic.
    """
    M = A.T @ A                          # (n_patches, n_patches)
    row_sums = M.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < eps, 1.0, row_sums)
    return M / row_sums


def stationary_distribution(T: np.ndarray, n_iter: int = 2000, eps: float = 1e-9) -> np.ndarray:
    """Power iteration: π_{n+1} = π_n T until ||Δπ||_1 < eps.

    Returns π : (n_patches,) normalised to sum to 1.
    """
    n = T.shape[0]
    pi = np.ones(n, dtype=np.float64) / n
    for _ in range(n_iter):
        pi_new = pi @ T
        pi_new /= pi_new.sum()
        if np.abs(pi_new - pi).sum() < eps:
            break
        pi = pi_new
    return pi_new


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def overlay_heatmap(image_path: str, pi: np.ndarray, grid_size: int = 24) -> "Image":
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    # Make square (same as LLaVA preprocessing)
    size = max(W, H)
    square = Image.new("RGB", (size, size), (0, 0, 0))
    square.paste(img, ((size - W) // 2, (size - H) // 2))
    square = square.resize((384, 384), Image.BICUBIC)

    heatmap = pi.reshape(grid_size, grid_size)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-10)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Original image
    axes[0].imshow(square)
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis("off")

    # Heatmap alone
    im = axes[1].imshow(heatmap, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("Stationary π (24×24)", fontsize=10)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # Overlay
    axes[2].imshow(square)
    axes[2].imshow(
        heatmap,
        cmap="jet",
        alpha=0.55,
        extent=[0, 384, 384, 0],
        vmin=0, vmax=1,
        interpolation="bilinear",
    )
    axes[2].set_title("Overlay", fontsize=10)
    axes[2].axis("off")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--model-base", default=None)
    p.add_argument("--image-dir", default="data/test_imgs")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--prompt", default="Describe this image.")
    p.add_argument("--image-attention-layers", default="19:25")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--output-dir", default="outputs/markov_attention")
    return p


def main():
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    from llava.utils import disable_torch_init
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, args.model_base, model_name, load_4bit=True
    )

    from tools.sadt.pipeline_utils import parse_layers
    layers = parse_layers(args.image_attention_layers)
    config = default_config(image_attention_layers=layers)

    images = collect_images(None, args.image_dir, args.limit)
    records = []

    for image_file in images:
        name = Path(image_file).stem
        print(f"\nProcessing {name} …")

        output, input_ids, response = generate_with_trace(
            model, tokenizer, image_processor, model_name, image_file, args.prompt,
            max_new_tokens=args.max_new_tokens, temperature=0.0,
        )
        print(f"  Caption: {response[:80]}")

        img_idx = get_img_idx(input_ids)
        A = compute_full_attention_matrix(output, img_idx, config.image_token_count, layers)
        # A : (T, n_patches)

        T_mat = build_transition_matrix(A)
        pi = stationary_distribution(T_mat)

        # Stats
        top5 = np.argsort(pi)[::-1][:5].tolist()
        entropy = float(-np.sum(pi * np.log(pi + 1e-12)))
        gini = float(1 - np.sum(pi ** 2) / (np.sum(pi) ** 2 / len(pi)) / len(pi))

        print(f"  π: entropy={entropy:.3f}  top-5 patches={top5}")

        fig = overlay_heatmap(image_file, pi, grid_size=config.grid_size)
        out_path = out_dir / f"img_{name}_stationary.png"
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
        print(f"  Saved → {out_path}")

        records.append({
            "image": Path(image_file).name,
            "caption": response,
            "stationary_entropy": entropy,
            "stationary_gini": gini,
            "top5_patches": top5,
            "n_generated_tokens": int(A.shape[0]),
        })

    with open(out_dir / "results.json", "w") as f:
        json.dump(records, f, indent=2)

    print(f"\n=== Done. {len(records)} images saved to {out_dir}/ ===")


if __name__ == "__main__":
    main()
