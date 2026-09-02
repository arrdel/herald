#!/usr/bin/env python
"""Offline analysis of E1.1b -- no training, no GPU, reads only the saved trace.

Three things E1.1 could not say on its own:

1. **The undertraining confound is refuted.** Acceptance falls monotonically with training budget at
   both d=16 and d=128, so E1.1's fixed 4,000 steps was not depressing large-d acceptance. The
   confound predicted the opposite.

2. **Why it falls: validation NLL is non-monotone.** Both models overfit well before 4,000 steps,
   and they overfit to *different* solutions, so per-coordinate disagreement grows with budget. The
   convergence trace is therefore partly a divergence trace, which is worth saying out loud.

3. **The matched-delta comparison**, which is the real prize. Because delta drifts over training,
   the two dimensions each sweep out a *range* of model agreement. Plotting acceptance against
   delta rather than against step puts them on a common axis, so the vertical gap between the two
   curves is what dimension costs at *identical* per-coordinate agreement. E1.1 argued this from
   delta merely being flat across the sweep; here the matching is direct.

Run:  python scripts/analyze_e11b.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

E11_BUDGET = 4000


def interp_log_acceptance(delta_query: float, deltas: np.ndarray, accs: np.ndarray) -> float | None:
    """Acceptance at ``delta_query``, interpolated in log space; None if outside the measured range.

    Log space because acceptance spans orders of magnitude and is roughly exponential in the
    squared divergence; linear interpolation across a 100x range would be dominated by the top end.
    Extrapolation is refused rather than guessed -- the whole point is a like-for-like comparison.
    """
    order = np.argsort(deltas)
    x, y = deltas[order], np.log(np.clip(accs[order], 1e-12, None))
    if delta_query < x[0] or delta_query > x[-1]:
        return None
    return float(np.exp(np.interp(delta_query, x, y)))


def matched_delta_table(results: list[dict]) -> list[dict]:
    """For each checkpoint of the lowest d, the acceptance the highest d achieves at the same delta."""
    lo, hi = results[0], results[-1]
    hi_delta = np.array([t["delta_median"] for t in hi["trace"]])
    hi_acc = np.array([t["acceptance"] for t in hi["trace"]])

    rows = []
    for t in lo["trace"]:
        matched = interp_log_acceptance(t["delta_median"], hi_delta, hi_acc)
        if matched is None:
            continue
        rows.append({
            "step": t["step"],
            "delta": t["delta_median"],
            f"acceptance_d{lo['d']}": t["acceptance"],
            f"acceptance_d{hi['d']}_at_same_delta": matched,
            "ratio": t["acceptance"] / max(matched, 1e-12),
        })
    return rows


def plot(results: list[dict], path: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    colors = plt.cm.viridis(np.linspace(0.15, 0.8, len(results)))

    ax = axes[0]
    for r, c in zip(results, colors, strict=True):
        steps = [t["step"] for t in r["trace"]]
        ax.plot(steps, [t["acceptance"] for t in r["trace"]], "o-", color=c, label=f"d={r['d']}")
    ax.axvline(E11_BUDGET, ls=":", color="crimson", lw=1.4)
    ax.text(E11_BUDGET, ax.get_ylim()[1], " E1.1's budget", color="crimson", va="top", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("training steps")
    ax.set_ylabel("exact acceptance rate")
    ax.set_title("Acceptance vs training budget\n(undertraining would show acceptance still climbing at 4k)",
                 fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    for r, c in zip(results, colors, strict=True):
        for name, ls in (("target", "-"), ("draft", "--")):
            hist = r["val_nll"][name]
            ax.plot([e["step"] for e in hist], [e["nll"] for e in hist], ls, color=c,
                    label=f"d={r['d']} {name}")
    ax.set_xlabel("training steps")
    ax.set_ylabel("validation NLL (nats/token)")
    ax.set_title("Validation NLL vs training budget\n(non-monotone: compare only within a model, not across d)",
                 fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[2]
    # Scatter, not a line: delta is not monotone in training step, so joining the points in training
    # order draws a path that doubles back and reads as structure that is not there.
    for r, c in zip(results, colors, strict=True):
        ax.scatter([t["delta_median"] for t in r["trace"]], [t["acceptance"] for t in r["trace"]],
                   color=c, s=34, label=f"d={r['d']}", edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel(r"per-coordinate disagreement  $\delta$")
    ax.set_ylabel("exact acceptance rate")
    ax.set_title("At matched model agreement, dimension still costs ~2 orders\n(vertical gap = the dimensional effect alone)",
                 fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="experiments/E1.1b-convergence/results.json")
    p.add_argument("--figure", default="docs/figures/e11b_convergence.png")
    args = p.parse_args()

    payload = json.loads(Path(args.results).read_text())
    results = sorted(payload["results"], key=lambda r: r["d"])
    print(f"{payload.get('experiment', 'E1.1b')}  commit {payload['commit'][:8]}  "
          f"host {payload['host']}  budget {payload['config']['mar_steps']} steps\n")

    for r in results:
        acc = [t["acceptance"] for t in r["trace"]]
        direction = "falls" if acc[-1] < acc[0] else "rises"
        print(f"d={r['d']}: acceptance {acc[0]:.4f} -> {acc[-1]:.4f} ({direction} "
              f"{acc[0] / max(acc[-1], 1e-12):.1f}x) as budget goes "
              f"{r['trace'][0]['step']} -> {r['trace'][-1]['step']}")
        for name in ("target", "draft"):
            hist = r["val_nll"][name]
            best = min(hist, key=lambda e: e["nll"])
            print(f"    {name:>6} val NLL {hist[0]['nll']:.2f} -> {hist[-1]['nll']:.2f}, "
                  f"best {best['nll']:.2f} at step {best['step']}")
        if r.get("best_val"):
            b = r["best_val"]
            print(f"    at each model's own best-val checkpoint "
                  f"(draft@{b['draft_step']}, target@{b['target_step']}): "
                  f"acceptance {b['acceptance']:.4f}, delta {b['delta_median']:.4f}")

    rows = matched_delta_table(results)
    if rows:
        lo, hi = results[0]["d"], results[-1]["d"]
        print("\nmatched-delta comparison: what dimension costs at identical model agreement")
        print(f"{'delta':>8}{f'acc d={lo}':>12}{f'acc d={hi}':>12}{'ratio':>9}")
        for r in rows:
            print(f"{r['delta']:>8.4f}{r[f'acceptance_d{lo}']:>12.4f}"
                  f"{r[f'acceptance_d{hi}_at_same_delta']:>12.5f}{r['ratio']:>9.1f}x")
        ratios = [r["ratio"] for r in rows]
        print(f"\nAcross {len(rows)} matched-delta pairs the d={lo} -> d={hi} penalty is "
              f"{min(ratios):.0f}x to {max(ratios):.0f}x (median {np.median(ratios):.0f}x), "
              f"with per-coordinate agreement held equal.")
    else:
        print("\nno overlapping delta range between the two dimensions; matched comparison skipped")

    out = Path(args.results).parent / "matched_delta.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")
    print(f"wrote {plot(results, Path(args.figure))}")


if __name__ == "__main__":
    main()
