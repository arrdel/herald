#!/usr/bin/env python
"""E1.2b -- does *which* ``k`` directions you verify matter, or only how many?

E1.2a is the null control that makes this experiment necessary. It showed acceptance is a function of
the verified dimension ``k`` essentially alone: a structureless random projection reaches the same
rate at matched ``k`` as anything else could. Acceptance therefore cannot separate a clever statistic
from a stupid one, and the entire claim has to be carried by the other axis -- what the rule
*discards*. E1.2-frontier measured that axis for random projections and found a cost of roughly
``(d - k) * c(k)``, with ``c(k)`` around 0.4 nats per discarded dimension.

So the ladder reduces to one number. At matched ``k``, a structured statistic earns its place only by
having a **lower ``c(k)``** than a random one. This script measures ``c(k)`` for four structured
choices of subspace against the random baseline:

``pca``            top-``k`` principal directions of the token distribution -- knows the data, not the decoder.
``signal_top``     top-``k`` eigenvectors of ``E[JᵀJ]`` for the decoder Jacobian: the directions that
                   move the reconstructed waveform most.
``dct_low``        the same restricted to the low-frequency DCT band of the decoded patch. This is the
                   transform-domain statistic, made linear by composing the DCT with the Jacobian so
                   the pushforward stays exactly Gaussian and the projected test stays exact with
                   respect to the statistic.
``signal_bottom``  the anti-control: verify the directions that matter *least*. It must come out worse
                   than random. If it does not, the metric is not measuring what it claims to and no
                   result above it can be believed.

Two costs are reported. Excess NLL answers "are we still sampling from the model?"; the signal-domain
energy distance answers "does the reconstructed waveform still have the right distribution?". They
can disagree, and a method that reports only one is easy to fool, so every rule is scored on both.

One thing this experiment inherits from E1.2a and then overturns: acceptance is *not* matched across
these rules at fixed ``k``. E1.2a's finding holds for random subspaces, which are exchangeable by
rotational symmetry, but a structured subspace can deliberately sit where ``p`` and ``q`` agree --
``signal_bottom`` does exactly that and buys a much higher rate at fixed ``k``. So the rules cannot be
compared at matched ``k``; they are compared on the (acceptance, cost) plane, and a rule wins only by
lying below the random-projection curve at the same acceptance.

Runs on E1.1's saved checkpoints. No training, no GPU.

Run:  python scripts/run_e12b.py --threads 8
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
from herald.eval.fidelity import TokensWithSignal
from herald.train.timeseries import CachedTokenDataset, load_checkpoint, set_seed
from herald.verifiers import random_projection
from herald.verifiers.structured import (
    pca_projection,
    signal_impact_matrix,
    top_eigen_projection,
)

RULES = ("random", "pca", "signal_top", "dct_low", "signal_bottom")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def ks_for(d: int) -> list[int]:
    return [k for k in (1, 2, 4, 8, 16, 32) if k < d]


def bases_for(tokenizer, tok_test, args, device) -> dict[str, torch.Tensor]:
    """Full ``d x d`` orthonormal bases, ordered so the top ``k`` rows are the rule's subspace.

    Each basis is built once per dimension and then simply truncated: the eigendecomposition does not
    depend on ``k``, and recomputing it per ``k`` would make the Jacobian pass the dominant cost for
    no change in the answer.
    """
    d = tok_test.mean.shape[-1]
    sample = tok_test.mean[: args.basis_samples].to(device)

    bases = {"pca": pca_projection(sample, d)}
    for name, band, largest in (
        ("signal_top", "all", True),
        ("dct_low", "low", True),
        ("signal_bottom", "all", False),
    ):
        m = signal_impact_matrix(tokenizer, sample, band=band, n_samples=args.jacobian_samples)
        bases[name] = top_eigen_projection(m, d, largest=largest)
    return bases


def run_one_dim(d: int, bundle, args) -> dict:
    device = args.device
    dim_dir = Path(args.checkpoints) / f"d{d}"
    t0 = time.time()

    tokenizer = load_checkpoint(dim_dir / "tokenizer.pt").to(device)
    tok_test = CachedTokenDataset(bundle.test, tokenizer, device=device)
    paired = TokensWithSignal(tok_test, bundle.test)
    draft = load_checkpoint(dim_dir / "draft.pt").to(device)
    target = load_checkpoint(dim_dir / "target.pt").to(device)

    common = {
        "device": device,
        "batch_size": args.batch_size,
        "max_batches": args.eval_batches,
        "seed": args.seed,
        "tokenizer": tokenizer,
    }

    print(f"  d={d}: building structured bases", flush=True)
    bases = bases_for(tokenizer, tok_test, args, device)

    rows = []
    anchor = measure_fidelity(draft, target, paired, projection=None, **common)
    anchor.update(k=d, rule="exact")
    rows.append(anchor)
    print(f"  d={d:<4d} exact          acc {anchor['acceptance']:.4f}  "
          f"nats {anchor['excess_nll']:+.4f}  energy {anchor['signal_energy']:+.4f}", flush=True)

    for k in ks_for(d):
        for rule in RULES:
            if rule == "random":
                draws = []
                for r in range(args.n_projections):
                    gen = torch.Generator(device=device).manual_seed(args.seed + 1000 * r)
                    a = random_projection(k, d, generator=gen, device=device)
                    draws.append(measure_fidelity(draft, target, paired, projection=a, **common))
                row = {"k": k, "rule": rule, "n_projections": args.n_projections,
                       "n_decisions": draws[0]["n_decisions"]}
                for key in ("acceptance", "excess_nll", "excess_nll_sem", "signal_energy",
                            "signal_energy_sem", "excess_mse", "residual_convergence"):
                    row[key] = float(np.mean([m[key] for m in draws]))
                # Spread over draws of A, kept separate from the within-draw SEM: they answer
                # different questions and collapsing them would hide which one dominates.
                for key in ("excess_nll", "signal_energy"):
                    row[f"{key}_min"] = float(np.min([m[key] for m in draws]))
                    row[f"{key}_max"] = float(np.max([m[key] for m in draws]))
            else:
                a = bases[rule][:k].contiguous()
                row = measure_fidelity(draft, target, paired, projection=a, **common)
                row.update(k=k, rule=rule)
            rows.append(row)
            print(f"  d={d:<4d} k={k:<3d} {rule:<14s} acc {row['acceptance']:.4f}  "
                  f"nats {row['excess_nll']:+.4f}  energy {row['signal_energy']:+.4f}", flush=True)

    return {"d": d, "rows": rows, "seconds": time.time() - t0}


def random_cost_at(rows: list[dict], acceptance: float, key: str) -> float | None:
    """The random-projection curve's cost at a given acceptance, by linear interpolation.

    Returns ``None`` outside the measured range rather than extrapolating: the whole point of the
    comparison is that the baseline was actually observed there, and an extrapolated baseline would
    let a structured rule "win" against a number nobody measured.
    """
    pts = sorted((r["acceptance"], r[key]) for r in rows if r["rule"] == "random")
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    if not xs or not (xs[0] <= acceptance <= xs[-1]):
        return None
    return float(np.interp(acceptance, xs, ys))


def summarize(results: list[dict]) -> list[str]:
    lines = []
    for r in results:
        d = r["d"]
        lines.append(f"\nd={d}: cost at matched acceptance (lower is better; "
                     "'vs rand' is the random-projection curve interpolated to the same rate)")
        lines.append(f"  {'k':>3s}  {'rule':<14s} {'accept':>7s} {'nats':>8s} {'vs rand':>9s} "
                     f"{'energy':>8s} {'vs rand':>9s}")
        for row in r["rows"]:
            if row["rule"] == "exact":
                continue
            cells = []
            for key in ("excess_nll", "signal_energy"):
                base = random_cost_at(r["rows"], row["acceptance"], key)
                ratio = "  --" if base is None or row["rule"] == "random" else f"{row[key] - base:+9.4f}"
                cells.append(f"{row[key]:>8.4f} {ratio:>9s}")
            lines.append(f"  {row['k']:>3d}  {row['rule']:<14s} {row['acceptance']:>7.4f} "
                         + " ".join(cells))
    return lines


def plot(results: list[dict], path: Path) -> Path:
    colors = {"random": "#2b6cb0", "pca": "#38a169", "signal_top": "#d69e2e",
              "dct_low": "#805ad5", "signal_bottom": "#e53e3e"}
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(5.6 * n, 9.0), squeeze=False)

    for col, r in enumerate(results):
        d = r["d"]
        for metric, ax in (("excess_nll", axes[0][col]), ("signal_energy", axes[1][col])):
            for rule in RULES:
                pts = [row for row in r["rows"] if row["rule"] == rule]
                if not pts:
                    continue
                ax.plot([p["acceptance"] for p in pts], [p[metric] for p in pts],
                        "o-", color=colors[rule], label=rule, alpha=0.9)
            exact = next(row for row in r["rows"] if row["rule"] == "exact")
            ax.scatter([exact["acceptance"]], [exact[metric]], marker="*", s=190,
                       color="black", zorder=5, label="exact (lossless)")
            ax.axhline(0.0, ls=":", color="grey", lw=1.2)
            nll = metric == "excess_nll"
            ax.set_xlabel("acceptance rate")
            ax.set_ylabel("excess NLL under the target (nats/token)" if nll
                          else "signal-domain energy distance")
            # Lower is better on both axes' vertical direction; better rules sit down and to the right.
            ax.set_title(f"d={d}: {'likelihood' if nll else 'signal'} cost of acceptance",
                         fontsize=10)
            ax.legend(fontsize=8, loc="upper left")
            ax.grid(alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="+", default=[16, 64])
    ap.add_argument("--checkpoints", default="experiments/E1.1-dsweep")
    ap.add_argument("--out", default="experiments/E1.2b-structured")
    ap.add_argument("--figure", default="docs/figures/e12b_structured.png")
    ap.add_argument("--dataset", default="etth1")
    ap.add_argument("--window", type=int, default=96)
    ap.add_argument("--n-projections", type=int, default=3)
    ap.add_argument("--basis-samples", type=int, default=2048)
    ap.add_argument("--jacobian-samples", type=int, default=32)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    figure = Path(args.figure)

    if args.plot_only:
        payload = json.loads((out / "results.json").read_text())
        print(plot(payload["results"], figure))
        print("\n".join(summarize(payload["results"])))
        return

    set_seed(args.seed)
    series = load_etth1() if args.dataset == "etth1" else synthetic_ar()
    bundle = build_bundle(series, window=args.window)

    t0 = time.time()
    results = [run_one_dim(d, bundle, args) for d in args.dims]
    payload = {
        "experiment": "E1.2b-structured",
        "commit": git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "args": vars(args),
        "results": results,
        "seconds": time.time() - t0,
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out / 'results.json'}")
    print(f"wrote {plot(results, figure)}")
    print("\n".join(summarize(results)))


if __name__ == "__main__":
    main()
