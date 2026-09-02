#!/usr/bin/env python
"""E1.2-frontier -- the acceptance/fidelity trade for random projections.

E1.2a showed acceptance is a function of the verified dimension ``k`` essentially alone, so it can no
longer tell the rungs of the ladder apart: at matched ``k`` a random projection reaches the same rate
a structured one will. What separates them has to be *what they discard*.

This traces the other axis. For each ``k`` it runs a full speculative step -- accept on the projected
statistic, and on rejection draw from the true residual ``(p - q)₊`` -- and measures how many nats
less likely the emitted tokens are under the target than genuine target samples.

The curve it produces is the deliverable: **the acceptance/fidelity frontier a structureless
projection already achieves.** E1.2b and E1.3 earn their place only by sitting below and to the right
of it -- more acceptance for fewer nats. Exact token-space verification is included as the anchor and
must report zero excess NLL, since with a correct residual sampler it emits the target exactly.

Runs on E1.1's saved checkpoints. No training, no GPU.

Run:  python scripts/run_e12_frontier.py
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
from herald.eval import measure_fidelity
from herald.train.timeseries import CachedTokenDataset, load_checkpoint, set_seed
from herald.verifiers import random_projection


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def run_one_dim(d: int, bundle, args) -> dict:
    device = args.device
    dim_dir = Path(args.checkpoints) / f"d{d}"
    t0 = time.time()

    tokenizer = load_checkpoint(dim_dir / "tokenizer.pt").to(device)
    tok_test = CachedTokenDataset(bundle.test, tokenizer, device=device)
    draft = load_checkpoint(dim_dir / "draft.pt").to(device)
    target = load_checkpoint(dim_dir / "target.pt").to(device)

    common = {"device": device, "batch_size": args.batch_size, "max_batches": args.eval_batches,
              "seed": args.seed}
    rows = []

    anchor = measure_fidelity(draft, target, tok_test, projection=None, **common)
    anchor["k"] = d
    anchor["rule"] = "exact"
    rows.append(anchor)
    print(f"  d={d:<4d} exact      acceptance {anchor['acceptance']:.4f}  "
          f"excess NLL {anchor['excess_nll']:+.4f} +- {anchor['excess_nll_sem']:.4f}")

    for k in [k for k in (1, 2, 4, 8, 16, 32, 64) if k < d]:
        per_draw = []
        for r in range(args.n_projections):
            gen = torch.Generator(device=device).manual_seed(args.seed + 1000 * r)
            a = random_projection(k, d, generator=gen, device=device)
            per_draw.append(measure_fidelity(draft, target, tok_test, projection=a, **common))
        acc = np.array([m["acceptance"] for m in per_draw])
        exc = np.array([m["excess_nll"] for m in per_draw])
        rows.append({
            "k": k,
            "rule": "random_projection",
            "acceptance": float(acc.mean()),
            "excess_nll": float(exc.mean()),
            "excess_nll_min": float(exc.min()),
            "excess_nll_max": float(exc.max()),
            "excess_nll_sem": float(np.mean([m["excess_nll_sem"] for m in per_draw])),
            "residual_convergence": float(np.mean([m["residual_convergence"] for m in per_draw])),
            "n_projections": args.n_projections,
            "n_decisions": per_draw[0]["n_decisions"],
        })
        print(f"  d={d:<4d} k={k:<4d}     acceptance {acc.mean():.4f}  "
              f"excess NLL {exc.mean():+.4f} [{exc.min():+.4f}, {exc.max():+.4f}]")

    return {"d": d, "rows": rows, "seconds": time.time() - t0}


def plot(results: list[dict], path: Path) -> Path:
    fig, axes = plt.subplots(1, len(results), figsize=(5.6 * len(results), 4.8), squeeze=False)

    for ax, r in zip(axes[0], results, strict=True):
        proj = [row for row in r["rows"] if row["rule"] == "random_projection"]
        exact = next(row for row in r["rows"] if row["rule"] == "exact")

        acc = [row["acceptance"] for row in proj]
        exc = [row["excess_nll"] for row in proj]
        lo = [row["excess_nll"] - row["excess_nll_min"] for row in proj]
        hi = [row["excess_nll_max"] - row["excess_nll"] for row in proj]

        ax.errorbar(acc, exc, yerr=[lo, hi], fmt="o-", color="#2b6cb0", capsize=3,
                    label="random projection")
        for row in proj:
            ax.annotate(f"k={row['k']}", (row["acceptance"], row["excess_nll"]),
                        textcoords="offset points", xytext=(6, -10), fontsize=8, color="#2b6cb0")
        ax.scatter([exact["acceptance"]], [exact["excess_nll"]], marker="*", s=190,
                   color="crimson", zorder=5, label=f"exact (k={exact['k']}), lossless anchor")
        ax.axhline(0.0, ls=":", color="grey", lw=1.2)

        ax.set_xlabel("acceptance rate")
        ax.set_ylabel("excess NLL under the target (nats/token)")
        # Better = more acceptance for fewer nats, i.e. down and to the right.
        ax.set_title(f"d={r['d']}: what acceptance costs\n"
                     "E1.2b and E1.3 must land below-right of this curve", fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--dims", type=int, nargs="+", default=[16, 64])
    p.add_argument("--checkpoints", default="experiments/E1.1-dsweep")
    p.add_argument("--dataset", default="etth1", choices=["etth1", "synthetic"])
    p.add_argument("--window", type=int, default=96)
    p.add_argument("--patch", type=int, default=8)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--n-projections", type=int, default=3)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="experiments/E1.2-frontier")
    p.add_argument("--figure", default="docs/figures/e12_frontier.png")
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
    print("exact verification is the anchor: with a correct residual sampler it emits the target,")
    print("so its excess NLL must be zero. Anything else is the harness, not the rule.\n")

    results = []
    for d in args.dims:
        results.append(run_one_dim(d, bundle, args))
        (out_dir / "results.json").write_text(json.dumps({
            "experiment": "E1.2-frontier",
            "commit": git_commit(),
            "host": platform.node(),
            "config": vars(args),
            "results": results,
        }, indent=2))

    print(f"\n{'d':>5}{'rule':>20}{'acceptance':>12}{'excess NLL':>13}{'+-':>9}")
    for r in results:
        for row in r["rows"]:
            name = "exact" if row["rule"] == "exact" else f"random k={row['k']}"
            print(f"{r['d']:>5}{name:>20}{row['acceptance']:>12.4f}"
                  f"{row['excess_nll']:>13.4f}{row['excess_nll_sem']:>9.4f}")

    print(f"\nwrote {out_dir / 'results.json'}")
    print(f"wrote {plot(results, Path(args.figure))}")


if __name__ == "__main__":
    main()
