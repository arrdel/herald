#!/usr/bin/env python
"""Feasibility map: when is exact continuous speculative verification worth attempting?

Two panels over the E0.8 analytic results.

(a) Acceptance against token dimension for several fixed per-coordinate gaps, with the trained
    testbed's measured points on top. The testbed is a noisy sample of one of these curves, which is
    the cleanest way to show that the collapse is the analytic law rather than a training artefact.

(b) The engineering read: acceptance contours over ``(d, delta/sigma)``. A model pair is a point on
    this plane and the contours say what it buys. Because the contours are lines of
    ``delta*sqrt(d) = const`` they have slope ``-1/2`` on log-log axes, so every halving of the
    tolerable gap costs a factor of four in token width.

Example:
    python scripts/plot_e08_feasibility.py --results experiments/E0.8-analytic/results.json \
        --out submission/figures/e08_feasibility.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import norm  # noqa: E402

# Measured in E1.1: acceptance of the exact rule in the trained testbed at each token width.
TESTBED_D = np.array([4, 8, 16, 32, 64, 128])
TESTBED_A = np.array([0.6425, 0.4069, 0.1743, 0.0736, 0.0178, 0.0035])


def acceptance(M):
    return 2.0 - 2.0 * norm.cdf(np.asarray(M) / 2.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("experiments/E0.8-analytic/results.json"))
    ap.add_argument("--out", type=Path, default=Path("submission/figures/e08_feasibility.png"))
    args = ap.parse_args()
    json.loads(args.results.read_text())  # fail early if the run is missing

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.0, 4.0))

    # ---------------- (a) collapse curves, with the trained sweep on top ----------------
    d_grid = np.logspace(0, np.log10(4096), 400)
    for gap, style in [(0.1, ":"), (0.2, "-."), (0.4, "--"), (0.578, "-"), (1.0, "--")]:
        lw = 2.2 if gap == 0.578 else 1.2
        col = "C3" if gap == 0.578 else "0.45"
        axA.plot(d_grid, acceptance(gap * np.sqrt(d_grid)), style, color=col, lw=lw,
                 label=rf"$\delta/\sigma={gap}$" + (" (testbed)" if gap == 0.578 else ""))
    axA.plot(TESTBED_D, TESTBED_A, "o", color="C0", ms=7, zorder=5,
             markeredgecolor="white", markeredgewidth=1.0, label="measured (trained testbed)")
    axA.set_xscale("log", base=2)
    axA.set_yscale("log")
    axA.set_ylim(1e-4, 1.3)
    axA.set_xlabel("token dimension $d$")
    axA.set_ylabel("exact acceptance rate")
    axA.set_title("(a) The collapse is the analytic law", fontsize=11, loc="left")
    axA.grid(alpha=0.25, which="both")
    axA.legend(fontsize=7.5, loc="lower left", framealpha=0.92)

    # ---------------- (b) feasibility map ----------------
    dd = np.logspace(0, np.log10(8192), 300)
    gg = np.logspace(np.log10(3e-3), np.log10(2.0), 300)
    D, G = np.meshgrid(dd, gg)
    A = acceptance(G * np.sqrt(D))

    levels = [0.05, 0.2, 0.5, 0.8]
    filled = axB.contourf(D, G, A, levels=[0] + levels + [1.0], cmap="RdYlGn", alpha=0.55)
    cs = axB.contour(D, G, A, levels=levels, colors="black", linewidths=0.9)
    # Place each label on its own contour near d=2^9, where the four curves are well separated.
    axB.clabel(cs, fmt=lambda v: f"A={v:g}", fontsize=7.5, inline=True,
               manual=[(512, 2.0 * norm.ppf((2 - lv) / 2) / np.sqrt(512)) for lv in levels])
    fig.colorbar(filled, ax=axB, label="acceptance", pad=0.02)

    # The two operating points an image pipeline actually has.
    for d_op, lbl, ha in [(16, "one KL16\ntoken", "left"), (4096, "a $16{\\times}16$\ngrid of them", "right")]:
        axB.axvline(d_op, color="C0", lw=1.0, ls="--", alpha=0.85)
        axB.text(d_op * (1.25 if ha == "left" else 0.8), 4.2e-3, lbl, fontsize=7.5, color="C0",
                 va="bottom", ha=ha)
    axB.axhline(0.578, color="C3", lw=1.1, ls=":", alpha=0.95)
    axB.text(1.2, 0.578 * 1.13, "testbed pair, $\\delta/\\sigma=0.58$", fontsize=7.5, color="C3")

    axB.set_xscale("log", base=2)
    axB.set_yscale("log")
    axB.set_xlabel("token dimension $d$")
    axB.set_ylabel(r"per-coordinate gap $\delta/\sigma$")
    axB.set_title("(b) What a pair must achieve to be worth verifying", fontsize=11, loc="left")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
