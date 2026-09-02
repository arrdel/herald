#!/usr/bin/env python
"""E1.2a -- the null control: how much acceptance does a *random* projection buy?

E1.1 showed exact acceptance collapsing with token dimension. The obvious remedy is to stop
verifying the raw token and verify a low-dimensional statistic of it instead. Before crediting any
*structured* statistic (E1.2b's transform-domain features, E1.3's learned JEPA projection) with that
gain, we need the curve a projection with **no structure at all** already achieves.

This is not a formality. Fearnhead & Prangle (JRSS-B 2012) prove that ABC acceptance scales as
``h^d`` in the dimension of the matched statistic, so *any* dimension reduction must raise
acceptance; Grouped Speculative Decoding (ICCV 2025) proves the discrete analogue via a
TV-reduction argument. A structured statistic that beats raw verification has therefore demonstrated
nothing. The question E1.2b and E1.3 have to answer is whether they beat **this** curve at matched
``k``, and that is the number this script produces.

Runs entirely on E1.1's saved checkpoints -- no training, no GPU. The models, tokenizers and test
split are exactly E1.1's, so the ``k = d`` column must reproduce E1.1's acceptance to within Monte
Carlo error, which doubles as an end-to-end check that the projected density is right.

Run:  python scripts/run_e12a.py
"""

from __future__ import annotations

import argparse
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
from herald.eval import measure_acceptance
from herald.train.timeseries import CachedTokenDataset, load_checkpoint, set_seed
from herald.verifiers import random_projection


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def ks_for(d: int) -> list[int]:
    """Powers of two up to ``d``, always including ``d`` itself as the no-op control."""
    ks = [k for k in (1, 2, 4, 8, 16, 32, 64, 128) if k < d]
    return [*ks, d]


def run_one_dim(d: int, bundle, args) -> dict:
    device = args.device
    dim_dir = Path(args.checkpoints) / f"d{d}"
    t0 = time.time()

    tokenizer = load_checkpoint(dim_dir / "tokenizer.pt").to(device)
    tok_test = CachedTokenDataset(bundle.test, tokenizer, device=device)
    draft = load_checkpoint(dim_dir / "draft.pt").to(device)
    target = load_checkpoint(dim_dir / "target.pt").to(device)

    rows = []
    for k in ks_for(d):
        # Several draws of A per k: a single random projection is one sample from a distribution
        # over subspaces, and the spread across draws is part of the result -- if it is wide, "a
        # random projection gives X" is not a well-defined baseline to compare against.
        per_draw = []
        for r in range(args.n_projections):
            gen = torch.Generator(device=device).manual_seed(args.seed + 1000 * r)
            a = random_projection(k, d, generator=gen, device=device)
            per_draw.append(measure_acceptance(
                draft, target, tok_test, device=device, batch_size=args.batch_size,
                max_batches=args.eval_batches, seed=args.seed, projection=a,
            ))
        acc = np.array([m["acceptance"] for m in per_draw])
        rows.append({
            "k": k,
            "acceptance_mean": float(acc.mean()),
            "acceptance_min": float(acc.min()),
            "acceptance_max": float(acc.max()),
            "acceptance_std_across_projections": float(acc.std()),
            "delta_median": float(np.mean([m["delta_median"] for m in per_draw])),
            "log_sigma_gap_median": float(np.mean([m["log_sigma_gap_median"] for m in per_draw])),
            "n_projections": args.n_projections,
            "n_decisions": per_draw[0]["n_decisions"],
        })
        print(f"  d={d:<4d} k={k:<4d} acceptance {acc.mean():.4f} "
              f"[{acc.min():.4f}, {acc.max():.4f}]  delta {rows[-1]['delta_median']:.4f}")

    raw = measure_acceptance(draft, target, tok_test, device=device, batch_size=args.batch_size,
                             max_batches=args.eval_batches, seed=args.seed)
    return {"d": d, "raw_acceptance": raw["acceptance"], "raw_delta_median": raw["delta_median"],
            "sweep": rows, "seconds": time.time() - t0}


def plot(results: list[dict], path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(results)))

    ax = axes[0]
    for r, c in zip(results, colors, strict=True):
        ks = [row["k"] for row in r["sweep"]]
        acc = [row["acceptance_mean"] for row in r["sweep"]]
        lo = [row["acceptance_min"] for row in r["sweep"]]
        hi = [row["acceptance_max"] for row in r["sweep"]]
        ax.plot(ks, acc, "o-", color=c, label=f"d={r['d']}")
        ax.fill_between(ks, lo, hi, color=c, alpha=0.2)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("verified dimension $k$")
    ax.set_ylabel("acceptance rate")
    ax.set_title("Acceptance is set by the verified dimension $k$, not the token dimension $d$\n"
                 "(six values of $d$ collapse onto one curve; band = spread over draws of $A$)",
                 fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    for r, c in zip(results, colors, strict=True):
        ks = [row["k"] for row in r["sweep"]]
        gain = [row["acceptance_mean"] / max(r["raw_acceptance"], 1e-12) for row in r["sweep"]]
        ax.plot(ks, gain, "o-", color=c, label=f"d={r['d']}")
    ax.axhline(1.0, ls=":", color="crimson", lw=1.3)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("verified dimension $k$")
    ax.set_ylabel("acceptance / raw acceptance")
    ax.set_title("The bar E1.2b and E1.3 must clear\n(gain that is free, from dimension alone)",
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
    p.add_argument("--plot-only", action="store_true", help="regenerate the figure from results.json")
    p.add_argument("--dims", type=int, nargs="+", default=[16, 32, 64, 128])
    p.add_argument("--checkpoints", default="experiments/E1.1-dsweep",
                   help="E1.1's output directory; this experiment retrains nothing")
    p.add_argument("--dataset", default="etth1", choices=["etth1", "synthetic"])
    p.add_argument("--window", type=int, default=96)
    p.add_argument("--patch", type=int, default=8)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--n-projections", type=int, default=5)
    p.add_argument("--eval-batches", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="experiments/E1.2a-random-projection")
    p.add_argument("--figure", default="docs/figures/e12a_random_projection.png")
    args = p.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    set_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        saved = json.loads((out_dir / "results.json").read_text())["results"]
        print(f"wrote {plot(saved, Path(args.figure))}")
        return

    series = load_etth1() if args.dataset == "etth1" else synthetic_ar(n_steps=20_000, n_channels=7)
    bundle = build_bundle(series, window=args.window, stride=args.stride)
    print(f"{args.dataset}: {bundle.describe()}")
    print(f"reusing checkpoints from {args.checkpoints}, {args.n_projections} projections per k\n")

    results = []
    for d in args.dims:
        results.append(run_one_dim(d, bundle, args))
        (out_dir / "results.json").write_text(json.dumps({
            "experiment": "E1.2a",
            "commit": git_commit(),
            "host": platform.node(),
            "config": vars(args),
            "results": results,
        }, indent=2))

    print(f"\n{'d':>5}{'raw (k=d)':>12}{'k=d check':>12}{'k=4':>10}{'k=1':>10}{'gain@k=4':>11}")
    for r in results:
        by_k = {row["k"]: row["acceptance_mean"] for row in r["sweep"]}
        k4, k1 = by_k.get(4, float("nan")), by_k.get(1, float("nan"))
        print(f"{r['d']:>5}{r['raw_acceptance']:>12.4f}{by_k[r['d']]:>12.4f}"
              f"{k4:>10.4f}{k1:>10.4f}{k4 / max(r['raw_acceptance'], 1e-12):>11.1f}x")

    print("\nThe 'k=d check' column must match 'raw': a square orthogonal projection is a bijection,")
    print("so any discrepancy there is an error in the projected density, not a finding.")
    print(f"\nwrote {out_dir / 'results.json'}")
    print(f"wrote {plot(results, Path(args.figure))}")


if __name__ == "__main__":
    main()
