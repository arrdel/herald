"""Figure for E0.5: the KL16 decoder has no compact band to keep.

Two spectra of the *same* ImageNet samples, each sorted descending and truncated to the top 64
components, so they are directly comparable:

  * the 2D DCT coefficient magnitudes of the decoded image, which is what transform coding exploits;
  * the singular values of the KL16 decoder's local Jacobian, which is what projected verification
    would need to exploit.

The first decays steeply (energy compaction, the premise of JPEG). The second is nearly flat, so
there is no perceptually cheap complement to leave unverified.

Usage:
    python scripts/plot_e05_spectrum.py \
        --results experiments/E0.5-kl16-jacobian/results.json \
        --samples experiments/E0.1-mar-base/samples.png \
        --out submission/figures/e05_kl16_spectrum.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.fft import dctn

TOP_K = 64
# torchvision.make_grid with nrow=3, padding=2 over 256px tiles gives a 776px canvas.
TILE = 256
PAD = 2


def image_tiles(path: Path, n: int) -> list[np.ndarray]:
    """Split the sample grid back into individual images, in the order they were generated."""
    grid = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    per_row = (grid.shape[1] - PAD) // (TILE + PAD)
    tiles = []
    for i in range(n):
        r, c = divmod(i, per_row)
        y = PAD + r * (TILE + PAD)
        x = PAD + c * (TILE + PAD)
        tiles.append(grid[y : y + TILE, x : x + TILE])
    return tiles


def tensor_tiles(path: Path, n: int) -> list[np.ndarray]:
    """Same images run_e05.py consumed, as HWC arrays in [0, 1] on the displayed scale."""
    import torch

    x = torch.load(path, map_location="cpu")[:n].float()
    x = (x.clamp(-1, 1) + 1) / 2
    return list(x.permute(0, 2, 3, 1).numpy().astype(np.float64))


def dct_spectrum(tile: np.ndarray) -> np.ndarray:
    """Top-K sorted DCT coefficient magnitudes of the luminance channel, normalised to max 1."""
    lum = tile @ np.array([0.299, 0.587, 0.114])
    coef = np.abs(dctn(lum, norm="ortho")).ravel()
    coef.sort()
    top = coef[::-1][:TOP_K]
    return top / top[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--samples", type=Path, help="E0.1 montage PNG")
    ap.add_argument("--images", type=Path, help="tensor of samples, as passed to run_e05.py")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if (args.samples is None) == (args.images is None):
        ap.error("pass exactly one of --samples or --images")

    res = json.loads(args.results.read_text())
    per_latent = res["per_latent"]
    sv = np.array([lat["singular_values"] for lat in per_latent])  # (n_latents, TOP_K)
    tiles = (image_tiles(args.samples, len(per_latent)) if args.images is None
             else tensor_tiles(args.images, len(per_latent)))
    dct = np.array([dct_spectrum(t) for t in tiles])

    rank = np.arange(1, TOP_K + 1)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 4.1))

    # ---- Panel A: the contrast, both normalised to their own leading component.
    for i, row in enumerate(dct):
        ax0.plot(rank, row, color="tab:blue", alpha=0.75, lw=1.4,
                 label="image DCT coefficients" if i == 0 else None)
    sv_norm = sv / sv[:, :1]
    for i, row in enumerate(sv_norm):
        ax0.plot(rank, row, color="tab:red", alpha=0.75, lw=1.4,
                 label="KL16 decoder Jacobian" if i == 0 else None)

    ax0.set_xlabel("component, sorted descending")
    ax0.set_ylabel("magnitude relative to the largest")
    ax0.set_ylim(0, 1.02)
    ax0.set_xlim(1, TOP_K)
    ax0.set_title("(a) Signal energy compacts; decoder sensitivity does not", fontsize=10)
    ax0.legend(frameon=False, fontsize=9)
    ax0.grid(alpha=0.25)

    med_dct = np.median(dct[:, TOP_K - 1])
    med_sv = np.median(sv_norm[:, TOP_K - 1])
    ax0.annotate(f"median at rank {TOP_K}: {med_sv:.3f}", xy=(TOP_K, med_sv),
                 xytext=(20, 0.72), fontsize=8.5, color="tab:red",
                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
                 arrowprops=dict(arrowstyle="->", color="tab:red", lw=0.8))
    ax0.annotate(f"median at rank {TOP_K}: {med_dct:.3f}", xy=(TOP_K, med_dct),
                 xytext=(20, 0.16), fontsize=8.5, color="tab:blue",
                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
                 arrowprops=dict(arrowstyle="->", color="tab:blue", lw=0.8))

    # ---- Panel B: absolute gains, which is where the between-image spread shows up.
    label_each = len(sv) <= 6
    for i, row in enumerate(sv):
        ax1.plot(rank, row, lw=1.4 if label_each else 0.7, alpha=0.85 if label_each else 0.35,
                 color=None if label_each else "tab:purple",
                 label=f"latent {i} ($\\sigma_{{\\max}}$={row[0]:.1f})" if label_each else None)
    if not label_each:
        ax1.plot(rank, np.median(sv, axis=0), lw=2.2, color="black",
                 label=f"median of {len(sv)} latents")
    ax1.set_xlabel("component, sorted descending")
    ax1.set_ylabel("perceptual gain $\\sigma_i$ (pixels per unit latent)")
    ax1.set_xlim(1, TOP_K)
    ax1.set_ylim(bottom=0)
    spread = sv[:, 0].max() / sv[:, 0].min()
    ax1.set_title(f"(b) The same tolerance buys ${spread:.1f}\\times$ different pixel error",
                  fontsize=10)
    ax1.legend(frameon=False, fontsize=8.5)
    ax1.grid(alpha=0.25)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"wrote {args.out}")
    def q(a: np.ndarray) -> str:
        return f"{np.median(a):.4f} (IQR {np.percentile(a, 25):.4f}-{np.percentile(a, 75):.4f})"

    print(f"  n latents: {len(sv)}")
    print(f"  DCT magnitude at rank {TOP_K}:      {q(dct[:, TOP_K - 1])}")
    print(f"  Jacobian magnitude at rank {TOP_K}: {q(sv_norm[:, TOP_K - 1])}")
    print(f"  ratio of the two medians: {med_sv / med_dct:.1f}x shallower")
    print(f"  sigma_max across latents: {q(sv[:, 0])}, "
          f"range {sv[:, 0].min():.2f}-{sv[:, 0].max():.2f}")


if __name__ == "__main__":
    main()
