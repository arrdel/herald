#!/usr/bin/env python
"""E1.4 -- moving the frontier instead of sliding along it.

E1.2b is a negative result with a constructive corollary. Every acceptance rule tested there --
random subspaces, principal subspaces, decoder-Jacobian subspaces, DCT-band subspaces, and the
anti-control that verifies the least important directions -- lands on the *same* acceptance/fidelity
curve. Choosing which ``k`` directions to verify only slides a point along that curve; it never moves
the curve. The reason is a conservation identity: for distributions that factorise across an
orthonormal basis the total draft/target divergence splits into a verified part, which suppresses
acceptance, and an unverified part, which is exactly what leaks into the output. Their sum does not
depend on where the split is drawn.

So the frontier is a property of the *pair*, not of the rule, and the only way to beat it is to make
the pair closer. This script does that: it distils the draft against the frozen target with a
mode-covering objective and re-measures the frontier at several points along the way.

The comparison to make is at matched fidelity. Exact verification is lossless at every stage, so any
acceptance it gains through distillation is free -- whereas the projection rules of E1.2b buy
acceptance only by spending nats. The headline number is therefore how many nats of excess NLL a
projection would have had to spend to reach the acceptance that distillation delivers for zero.

Runs on E1.1's saved checkpoints. Training is small and CPU-only.

Run:  python scripts/run_e14.py --threads 8
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from herald.data.timeseries import build_bundle, load_etth1, synthetic_ar
from herald.eval import measure_fidelity
from herald.eval.fidelity import TokensWithSignal
from herald.train.timeseries import (
    CachedTokenDataset,
    TrainConfig,
    distillation_loss,
    load_checkpoint,
    set_seed,
    train_module,
)
from herald.verifiers import random_projection


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def frontier(draft, target, paired, tokenizer, d: int, args) -> list[dict]:
    """Exact verification plus the random-projection curve, for one draft."""
    common = {"device": args.device, "batch_size": args.batch_size,
              "max_batches": args.eval_batches, "seed": args.seed, "tokenizer": tokenizer}

    exact = measure_fidelity(draft, target, paired, projection=None, **common)
    exact.update(k=d, rule="exact")
    rows = [exact]

    for k in [k for k in (1, 2, 4, 8, 16, 32) if k < d]:
        draws = []
        for r in range(args.n_projections):
            gen = torch.Generator(device=args.device).manual_seed(args.seed + 1000 * r)
            a = random_projection(k, d, generator=gen, device=args.device)
            draws.append(measure_fidelity(draft, target, paired, projection=a, **common))
        row = {"k": k, "rule": "random"}
        for key in ("acceptance", "excess_nll", "excess_nll_sem", "signal_energy",
                    "signal_energy_sem", "residual_convergence"):
            row[key] = float(np.mean([m[key] for m in draws]))
        rows.append(row)
    return rows


def nats_to_reach(rows: list[dict], acceptance: float) -> float | None:
    """What the projection frontier would have to spend to reach a given acceptance.

    Exact verification is included as the curve's low-acceptance endpoint, because it *is* the
    ``k = d`` projection: the identity is an orthonormal ``A``, and leaving it out would truncate the
    curve exactly where distillation operates and make the comparison impossible to draw.

    Interpolating inside the measured range only. Outside it the honest answer is that the baseline
    was never observed there, which is itself the strongest form of the result: distillation reaches
    an acceptance that projection cannot buy at any price on the measured curve.
    """
    pts = sorted((r["acceptance"], r["excess_nll"]) for r in rows
                 if r["rule"] in ("random", "exact"))
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    if not xs or not (xs[0] <= acceptance <= xs[-1]):
        return None
    return float(np.interp(acceptance, xs, ys))


def run_one_dim(d: int, bundle, args) -> dict:
    device = args.device
    dim_dir = Path(args.checkpoints) / f"d{d}"
    t0 = time.time()

    tokenizer = load_checkpoint(dim_dir / "tokenizer.pt").to(device)
    tok_train = CachedTokenDataset(bundle.train, tokenizer, device=device)
    tok_test = CachedTokenDataset(bundle.test, tokenizer, device=device)
    paired = TokensWithSignal(tok_test, bundle.test)
    target = load_checkpoint(dim_dir / "target.pt").to(device)
    draft = load_checkpoint(dim_dir / "draft.pt").to(device)

    baseline = frontier(draft, target, paired, tokenizer, d, args)
    base_exact = next(r for r in baseline if r["rule"] == "exact")
    print(f"  d={d:<4d} step {0:<6d} exact acceptance {base_exact['acceptance']:.4f}  "
          f"nats {base_exact['excess_nll']:+.4f}  energy {base_exact['signal_energy']:+.4f}",
          flush=True)

    stages = [{"step": 0, "rows": baseline}]
    student = copy.deepcopy(draft)
    loss_fn = distillation_loss(target, seed=args.seed)

    for stage in range(args.stages):
        config = TrainConfig(steps=args.steps_per_stage, batch_size=args.batch_size,
                             lr=args.lr, device=device, seed=args.seed + stage,
                             num_workers=0, log_every=0, eval_every=0)
        student = train_module(student, tok_train, loss_fn, config, tag=f"distilled_d{d}")
        step = (stage + 1) * args.steps_per_stage
        rows = frontier(student, target, paired, tokenizer, d, args)
        stages.append({"step": step, "rows": rows})
        ex = next(r for r in rows if r["rule"] == "exact")
        print(f"  d={d:<4d} step {step:<6d} exact acceptance {ex['acceptance']:.4f}  "
              f"nats {ex['excess_nll']:+.4f}  energy {ex['signal_energy']:+.4f}", flush=True)

    final_exact = next(r for r in stages[-1]["rows"] if r["rule"] == "exact")
    equivalent = nats_to_reach(baseline, final_exact["acceptance"])
    return {
        "d": d,
        "stages": stages,
        "baseline_acceptance": base_exact["acceptance"],
        "final_acceptance": final_exact["acceptance"],
        "speedup_ratio": final_exact["acceptance"] / max(base_exact["acceptance"], 1e-9),
        "projection_nats_for_same_acceptance": equivalent,
        "seconds": time.time() - t0,
    }


def summarize(results: list[dict]) -> list[str]:
    lines = ["", "E1.4: distillation buys acceptance at zero fidelity cost."]
    for r in results:
        lines.append(
            f"  d={r['d']}: exact acceptance {r['baseline_acceptance']:.4f} -> "
            f"{r['final_acceptance']:.4f} ({r['speedup_ratio']:.1f}x), both lossless."
        )
        cost = r["projection_nats_for_same_acceptance"]
        if cost is None:
            lines.append("    a random projection never reaches that acceptance on the measured "
                         "curve, at any fidelity cost.")
        else:
            lines.append(f"    a random projection would have to spend {cost:+.2f} nats/token of "
                         "excess NLL to reach the same acceptance.")
    return lines


def plot(results: list[dict], path: Path, ncols: int = 3) -> Path:
    n = len(results)
    ncols = min(ncols, n)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.8 * ncols, 4.8 * nrows), squeeze=False)
    flat = [ax for row in axes for ax in row]
    for ax in flat[n:]:
        ax.set_visible(False)

    for ax, r in zip(flat[:n], results, strict=True):
        base = r["stages"][0]["rows"]
        proj = sorted((row["acceptance"], row["excess_nll"]) for row in base
                       if row["rule"] == "random")
        ax.plot([p[0] for p in proj], [p[1] for p in proj], "o-", color="#2b6cb0",
                label="projection frontier (undistilled)")

        shades = plt.cm.Reds(np.linspace(0.45, 0.95, len(r["stages"])))
        for stage, color in zip(r["stages"], shades, strict=True):
            ex = next(row for row in stage["rows"] if row["rule"] == "exact")
            ax.scatter([ex["acceptance"]], [ex["excess_nll"]], marker="*", s=200, color=color,
                       zorder=5, label=f"exact, {stage['step']} distillation steps")

        ax.axhline(0.0, ls=":", color="grey", lw=1.2)
        ax.set_xlabel("acceptance rate")
        ax.set_ylabel("excess NLL under the target (nats/token)")
        # Distillation moves the lossless point rightwards along zero; projection can only climb.
        ax.set_title(f"d={r['d']}: projection slides along the frontier,\n"
                     "distillation moves it", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_headline(results: list[dict], path: Path) -> Path:
    """Two panels for the main text: the mechanism at one dimension, then its trend across all.

    The right panel is the claim the single-dimension view cannot make: the gain is not a fixed
    bonus, it grows with the dimension, so distillation helps most exactly where the collapse of
    Section 3 bites hardest.
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.2))

    worst = max(results, key=lambda r: r["d"])
    base = worst["stages"][0]["rows"]
    undistilled = next(row for row in base if row["rule"] == "exact")
    # Exact verification is the k=d endpoint of the projection curve, so it belongs on the frontier;
    # without it the curve appears to start in mid-air well to the right of where distillation works.
    proj = sorted([(row["acceptance"], row["excess_nll"]) for row in base if row["rule"] == "random"]
                  + [(undistilled["acceptance"], undistilled["excess_nll"])])
    left.plot([p[0] for p in proj], [p[1] for p in proj], "o-", color="#2b6cb0",
              label="projection frontier (undistilled)")

    stars = []
    shades = plt.cm.Reds(np.linspace(0.45, 0.95, len(worst["stages"])))
    for stage, color in zip(worst["stages"], shades, strict=True):
        ex = next(row for row in stage["rows"] if row["rule"] == "exact")
        stars.append(ex["acceptance"])
        left.scatter([ex["acceptance"]], [ex["excess_nll"]], marker="*", s=210, color=color,
                     zorder=5, label=f"exact, {stage['step']} steps")

    # At high d the distilled points occupy a sliver of the acceptance range while the frontier runs
    # to ~0.9, so a linear axis stacks every star on the origin. Acceptance spans two decades here.
    left.set_xscale("log")

    # The vertical gap at the final distilled acceptance is the quantity the paper reports: what the
    # frontier charges for the rate distillation reached for nothing.
    cost = worst["projection_nats_for_same_acceptance"]
    if cost is not None:
        left.annotate("", xy=(max(stars), cost), xytext=(max(stars), 0.0),
                      arrowprops={"arrowstyle": "<->", "color": "#2f855a", "lw": 1.6})
        left.text(max(stars) * 1.25, cost * 0.55, f"{cost:.2f} nats\nsaved", fontsize=8,
                  color="#2f855a", va="center")
        # The frontier runs to tens of nats at high acceptance, which would flatten the gap being
        # annotated; clip to the region where the two are actually being compared.
        top = 4.0 * cost
        left.set_ylim(-0.28 * cost, top)
        onscreen = [p[0] for p in proj if p[1] <= top] + stars
        left.set_xlim(min(onscreen) * 0.7, max(onscreen) * 2.0)
    left.axhline(0.0, ls=":", color="grey", lw=1.2)
    left.set_xlabel("acceptance rate (log scale)")
    left.set_ylabel("excess NLL under the target (nats/token)")
    left.set_title(f"$d={worst['d']}$: projection climbs, distillation\nadvances along zero",
                   fontsize=10)
    left.legend(fontsize=7, loc="upper left")
    left.grid(alpha=0.3)

    dims = [r["d"] for r in results]
    gains = [r["speedup_ratio"] for r in results]
    right.plot(dims, gains, "o-", color="#c53030", lw=2, ms=7)
    for d, g in zip(dims, gains, strict=True):
        right.annotate(f"{g:.2f}$\\times$", (d, g), textcoords="offset points", xytext=(0, 8),
                       ha="center", fontsize=8)
    right.axhline(1.0, ls=":", color="grey", lw=1.2, label="no gain")
    right.set_xscale("log", base=2)
    right.set_xticks(dims)
    right.set_xticklabels([str(d) for d in dims])
    right.set_xlabel("token dimension $d$")
    right.set_ylabel("lossless acceptance gain from distillation")
    right.set_title("The gain grows with dimension:\ndistillation helps most where collapse is worst",
                    fontsize=10)
    right.set_ylim(0.0, max(gains) * 1.22)
    right.legend(fontsize=8, loc="upper left")
    right.grid(alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="+", default=[16, 64])
    ap.add_argument("--checkpoints", default="experiments/E1.1-dsweep")
    ap.add_argument("--out", default="experiments/E1.4-alignment")
    ap.add_argument("--figure", default="docs/figures/e14_alignment.png")
    ap.add_argument("--dataset", default="etth1")
    ap.add_argument("--window", type=int, default=96)
    ap.add_argument("--stages", type=int, default=3)
    ap.add_argument("--steps-per-stage", type=int, default=600)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-projections", type=int, default=3)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        payload = json.loads((out / "results.json").read_text())
        figure = Path(args.figure)
        print(plot(payload["results"], figure))
        print(plot_headline(payload["results"],
                            figure.with_name(figure.stem + "_headline" + figure.suffix)))
        print("\n".join(summarize(payload["results"])))
        return

    set_seed(args.seed)
    series = load_etth1() if args.dataset == "etth1" else synthetic_ar()
    bundle = build_bundle(series, window=args.window)

    t0 = time.time()
    results = [run_one_dim(d, bundle, args) for d in args.dims]
    payload = {
        "experiment": "E1.4-alignment",
        "commit": git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "args": vars(args),
        "results": results,
        "seconds": time.time() - t0,
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out / 'results.json'}")
    print(f"wrote {plot(results, Path(args.figure))}")
    print("\n".join(summarize(results)))


if __name__ == "__main__":
    main()
