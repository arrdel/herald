#!/usr/bin/env python
"""E1.2b analysis -- is the acceptance/fidelity frontier a property of the rule, or of the pair?

E1.2b measured five ways of choosing which ``k`` of ``d`` dimensions to verify: uniformly at random,
the top principal directions, the directions the decoder Jacobian says move the waveform most, the
same restricted to the low-frequency DCT band, and an anti-control that deliberately verifies the
*least* important directions. Plotted against ``k`` these look like five different methods. Plotted
against acceptance they should, if the conservation argument is right, be one curve.

The argument, for the tractable case. Take ``p`` and ``q`` Gaussian with shared covariance ``σ²I``
and mean gap ``δ``. Verifying the statistic ``Ax`` for orthonormal ``A`` pushes both forward to
Gaussians separated by ``‖Aδ‖``, so acceptance is a function of ``‖Aδ‖²`` alone; what leaks into the
output is governed by the complement, ``‖A⊥δ‖²``. Because ``A`` is orthonormal these satisfy

    ‖Aδ‖² + ‖A⊥δ‖² = ‖δ‖²,

a quantity that does not depend on ``A``. Acceptance and fidelity are therefore two readings of one
conserved budget, and every choice of subspace -- clever, random, or adversarially bad -- can only
pick a point on a single curve fixed by ``‖δ‖``. Choosing *which* directions to verify cannot beat
choosing *how many*, and the anti-control is not bad because it is unlucky but because it spends the
budget at the wrong end.

This script tests that prediction on the real mixture models, where the identity is not exact. It
scores every structured point against the random-projection curve interpolated to the *same
acceptance*, so the comparison is like for like, and reports the residuals against the dynamic range
of the cost. Residuals small compared with the range mean the rules have collapsed.

Offline, no GPU, no model loading.

Run:  python scripts/analyze_e12b.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RULES = ("random", "pca", "signal_top", "dct_low", "signal_bottom")
COLORS = {"random": "#2b6cb0", "pca": "#38a169", "signal_top": "#d69e2e",
          "dct_low": "#805ad5", "signal_bottom": "#e53e3e"}
METRICS = (("excess_nll", "excess NLL under the target (nats/token)"),
           ("signal_energy", "signal-domain energy distance"))


def baseline_at(rows: list[dict], acceptance: float, key: str) -> float | None:
    """The random-projection curve at a given acceptance, without extrapolating past its range."""
    pts = sorted((r["acceptance"], r[key]) for r in rows if r["rule"] == "random")
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    if not xs or not (xs[0] <= acceptance <= xs[-1]):
        return None
    return float(np.interp(acceptance, xs, ys))


def residuals(rows: list[dict], key: str) -> list[dict]:
    """Every non-random point's cost minus the random curve's cost at the same acceptance."""
    out = []
    for row in rows:
        if row["rule"] in ("exact", "random"):
            continue
        base = baseline_at(rows, row["acceptance"], key)
        if base is not None:
            out.append({"rule": row["rule"], "k": row["k"], "acceptance": row["acceptance"],
                        "value": row[key], "baseline": base, "residual": row[key] - base})
    return out


def summarize(results: list[dict]) -> list[str]:
    lines = []
    for r in results:
        rows = r["rows"]
        lines.append(f"\n{'=' * 78}\nd={r['d']}")
        for key, label in METRICS:
            res = residuals(rows, key)
            if not res:
                continue
            costs = [row[key] for row in rows if row["rule"] != "exact"]
            span = max(costs) - min(costs)
            dev = np.abs([x["residual"] for x in res])
            lines.append(f"\n  {label}")
            lines.append(f"    cost ranges over {span:.3f} across the sweep "
                         f"({min(costs):+.3f} to {max(costs):+.3f})")
            lines.append(f"    deviation from the random curve at matched acceptance: "
                         f"median {np.median(dev):.3f}, max {dev.max():.3f}")
            lines.append(f"    i.e. every rule sits within {100 * dev.max() / span:.1f}% of the "
                         "range of a single curve")

            best = min(res, key=lambda x: x["residual"])
            worst = max(res, key=lambda x: x["residual"])
            for tag, x in (("best", best), ("worst", worst)):
                verdict = "below" if x["residual"] < 0 else "above"
                lines.append(f"    {tag:<5s}: {x['rule']} k={x['k']} at acceptance "
                             f"{x['acceptance']:.3f} sits {abs(x['residual']):.3f} {verdict} "
                             "the random curve")
    return lines


def plot(results: list[dict], path: Path, metrics=METRICS) -> Path:
    """Cost against acceptance for every rule. One panel per (metric, dimension).

    Args:
        metrics: which cost axes to draw. The paper's main text uses excess NLL alone, where the
            collapse is tightest and a single row fits the page budget; the appendix uses both.
    """
    n = len(results)
    # Beyond three panels a single row is too wide to be legible once scaled to the text block, so
    # wrap into a grid and stack each metric's rows underneath one another.
    ncols = min(3, n)
    blocks = -(-n // ncols)
    fig, axes = plt.subplots(len(metrics) * blocks, ncols,
                             figsize=(5.8 * ncols, 4.6 * len(metrics) * blocks), squeeze=False)
    for ax in (axes[len(metrics) * blocks - 1][i] for i in range(n % ncols or ncols, ncols)):
        ax.set_visible(False)

    for col, r in enumerate(results):
        for row_idx, (key, label) in enumerate(metrics):
            ax = axes[(col // ncols) * len(metrics) + row_idx][col % ncols]
            for rule in RULES:
                pts = sorted((row["acceptance"], row[key]) for row in r["rows"]
                             if row["rule"] == rule)
                if not pts:
                    continue
                style = "o--" if rule == "signal_bottom" else "o-"
                ax.plot([p[0] for p in pts], [p[1] for p in pts], style, color=COLORS[rule],
                        label=rule.replace("_", " "), alpha=0.85, markersize=5)
            exact = next(row for row in r["rows"] if row["rule"] == "exact")
            ax.scatter([exact["acceptance"]], [exact[key]], marker="*", s=210, color="black",
                       zorder=6, label="exact (lossless)")
            ax.axhline(0.0, ls=":", color="grey", lw=1.2)
            ax.set_xlabel("acceptance rate")
            ax.set_ylabel(label)
            # The claim is the overlap: five different subspace rules tracing one curve.
            ax.set_title(f"d={r['d']}: five subspace rules, one frontier", fontsize=10)
            ax.legend(fontsize=8, loc="upper left")
            ax.grid(alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="experiments/E1.2b-structured/results.json")
    ap.add_argument("--figure", default="docs/figures/e12b_collapse.png")
    ap.add_argument("--compact-figure", default="docs/figures/e12b_collapse_compact.png",
                    help="single-row version, excess NLL only, for the page-limited main text")
    ap.add_argument("--compact-dims", type=int, nargs="+", default=None,
                    help="dimensions to draw in the compact figure; all of them by default. Six "
                         "panels do not survive being scaled to the text width, so the main text "
                         "shows the largest few and the appendix carries the rest.")
    args = ap.parse_args()

    results = json.loads(Path(args.results).read_text())["results"]
    print("\n".join(summarize(results)))
    print(f"\nwrote {plot(results, Path(args.figure))}")
    compact = results if args.compact_dims is None else \
        [r for r in results if r["d"] in set(args.compact_dims)]
    print(f"wrote {plot(compact, Path(args.compact_figure), metrics=METRICS[:1])}")


if __name__ == "__main__":
    main()
