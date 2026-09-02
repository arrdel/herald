#!/usr/bin/env python
"""E1.1 -- exact speculative acceptance as a function of token dimension d.

The plot claim C1 is built around. ImageNet cannot produce it: d is fixed at 16 by whichever VAE
the checkpoint was trained with. Here d is a free parameter, so the curve can actually be swept.

For each d: fit a patch VAE with latent dimension d, freeze it, then train two masked autoregressive
models on its latents -- a large *target* and a small *draft*. Condition both on the identical
context and mask, draw x from the draft, and evaluate the exact ratio ``min(1, p(x)/q(x))``. The
mixture head makes that ratio exact rather than approximate, which is the whole reason the testbed
uses one (see ``herald/models/ts_mar.py``).

**What makes this more than a curve.** Acceptance falling with d proves little on its own, because
raising d changes the tokenizer, the latent distribution, and how hard the tokens are to model --
so a decay could be any of those rather than dimension. Section 1b of the plan predicts a specific
mechanism: acceptance should track ``2 - 2Phi(delta*sqrt(d)/2)`` where delta is the *per-coordinate*
disagreement between draft and target. So this script measures delta directly and reports the
analytic prediction beside the measurement. If acceptance decays while delta holds roughly constant,
the dimensional mechanism is confirmed. If delta shrinks as d grows and acceptance stays flat, C1's
dimensional half is wrong -- and that is the result, reported as such.

Run:  python scripts/run_e11.py --dims 4 8 16 32 64 128
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import torch

from herald.data.timeseries import build_bundle, load_etth1, synthetic_ar
from herald.eval import DRAFT_ARCH, TARGET_ARCH, measure_acceptance
from herald.models.ts_mar import MARConfig, TimeSeriesMAR
from herald.models.ts_tokenizer import TimeSeriesTokenizer, TokenizerConfig
from herald.train.timeseries import (
    CachedTokenDataset,
    TrainConfig,
    load_checkpoint,
    mar_loss,
    set_seed,
    tokenizer_loss,
    train_module,
)
from herald.verifiers import analytic_acceptance_shared_covariance


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def measure_from_checkpoints(d: int, bundle, args, out_dir: Path) -> dict:
    """Re-measure a dimension from its saved checkpoints, without retraining.

    Added after the acceptance estimator was found to be drawing x from the global RNG, which left
    the reported rates carrying an uncontrolled Monte Carlo component. Re-measuring is inference
    only, so the committed numbers can be checked in minutes rather than by repeating the sweep.
    """
    device = args.device
    dim_dir = out_dir / f"d{d}"
    tokenizer = load_checkpoint(dim_dir / "tokenizer.pt").to(device)
    tok_test = CachedTokenDataset(bundle.test, tokenizer, device=device)
    draft = load_checkpoint(dim_dir / "draft.pt").to(device)
    target = load_checkpoint(dim_dir / "target.pt").to(device)

    result = measure_acceptance(
        draft, target, tok_test, device=device, batch_size=args.batch_size, seed=args.seed
    )
    result["d"] = d
    result["n_tokens"] = args.window // args.patch
    result["analytic_at_measured_delta"] = analytic_acceptance_shared_covariance(
        result["delta_median"], d
    )
    result["target_params"] = sum(p.numel() for p in target.parameters())
    result["draft_params"] = sum(p.numel() for p in draft.parameters())
    result["seconds"] = 0.0
    return result


def run_one_dim(d: int, bundle, args, out_dir: Path) -> dict:
    device = args.device
    window = args.window
    n_tokens = window // args.patch
    t0 = time.time()

    tok_cfg = TokenizerConfig(
        n_channels=bundle.n_channels, patch=args.patch, latent_dim=d, hidden=args.tok_hidden
    )
    tokenizer = TimeSeriesTokenizer(tok_cfg)
    train_module(
        tokenizer,
        bundle.train,
        tokenizer_loss,
        TrainConfig(
            steps=args.tok_steps, batch_size=args.batch_size, device=device, seed=args.seed,
            num_workers=0, out_dir=out_dir / f"d{d}", eval_every=0, log_every=args.tok_steps // 4,
        ),
        tag="tokenizer",
    )

    tok_train = CachedTokenDataset(bundle.train, tokenizer, device=device)
    tok_val = CachedTokenDataset(bundle.val, tokenizer, device=device)
    tok_test = CachedTokenDataset(bundle.test, tokenizer, device=device)

    models = {}
    for name, arch in (("target", TARGET_ARCH), ("draft", DRAFT_ARCH)):
        model = TimeSeriesMAR(MARConfig(latent_dim=d, n_tokens=n_tokens, **arch))
        train_module(
            model,
            tok_train,
            mar_loss,
            TrainConfig(
                steps=args.mar_steps, batch_size=args.batch_size, device=device,
                # Different seeds so draft and target are not the same run at two widths; their
                # disagreement should come from capacity, not from a shared initialisation.
                seed=args.seed + (0 if name == "target" else 1),
                num_workers=0, out_dir=out_dir / f"d{d}",
                eval_every=args.mar_steps // 2, log_every=args.mar_steps // 4,
            ),
            val_dataset=tok_val,
            tag=name,
        )
        models[name] = model

    result = measure_acceptance(
        models["draft"], models["target"], tok_test, device=device,
        batch_size=args.batch_size, seed=args.seed,
    )
    result["d"] = d
    result["n_tokens"] = n_tokens
    result["analytic_at_measured_delta"] = analytic_acceptance_shared_covariance(
        result["delta_median"], d
    )
    result["target_params"] = sum(p.numel() for p in models["target"].parameters())
    result["draft_params"] = sum(p.numel() for p in models["draft"].parameters())
    result["seconds"] = time.time() - t0
    return result


def plot_results(rows: list[dict], path: Path) -> Path:
    """The acceptance-vs-d curve, with delta plotted beside it.

    Both panels are needed together. Acceptance alone cannot distinguish a dimensional effect from
    the draft simply getting worse at larger d, and the delta panel is what rules that out.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dims = [r["d"] for r in rows]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax.plot(dims, [r["acceptance"] for r in rows], "o-", lw=2, label="measured (exact rule)")
    ax.plot(dims, [r["analytic_at_measured_delta"] for r in rows], "s--", lw=1.5, alpha=0.75,
            label=r"analytic $2-2\Phi(\delta\sqrt{d}/2)$, equal covariance")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("token dimension $d$")
    ax.set_ylabel("exact acceptance rate")
    ax.set_title("Acceptance collapses with token dimension")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    ax2.plot(dims, [r["delta_median"] for r in rows], "o-", lw=2,
             label=r"$\delta$: per-coordinate mean gap")
    ax2.plot(dims, [r["log_sigma_gap_median"] for r in rows], "^-", lw=2,
             label=r"$|\log\sigma_p/\sigma_q|$: scale gap")
    ax2.set_xscale("log", base=2)
    ax2.set_ylim(bottom=0)
    ax2.set_xlabel("token dimension $d$")
    ax2.set_ylabel("per-coordinate disagreement")
    ax2.set_title("Per-coordinate disagreement is flat for $d\\geq16$")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plot-only", action="store_true", help="regenerate the figure from results.json")
    p.add_argument("--measure-only", action="store_true",
                   help="re-measure acceptance from saved checkpoints without retraining")
    p.add_argument("--figure", default="docs/figures/e11_acceptance_vs_d.png")
    p.add_argument("--dims", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128])
    p.add_argument("--dataset", default="etth1", choices=["etth1", "synthetic"])
    p.add_argument("--window", type=int, default=96)
    p.add_argument("--patch", type=int, default=8)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--tok-steps", type=int, default=2000)
    p.add_argument("--tok-hidden", type=int, default=128)
    p.add_argument("--mar-steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="experiments/E1.1-dsweep")
    args = p.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        rows = json.loads((out_dir / "results.json").read_text())["rows"]
        print(f"wrote {plot_results(rows, Path(args.figure))}")
        return

    series = load_etth1() if args.dataset == "etth1" else synthetic_ar(n_steps=20_000, n_channels=7)
    bundle = build_bundle(series, window=args.window, stride=args.stride)
    print(f"{args.dataset}: {bundle.describe()}")
    print(f"device {args.device}, sweeping d = {args.dims}\n")

    rows = []
    for d in args.dims:
        row = measure_from_checkpoints(d, bundle, args, out_dir) if args.measure_only \
            else run_one_dim(d, bundle, args, out_dir)
        rows.append(row)
        print(
            f"d={d:4d}  acceptance {row['acceptance']:.4f}  "
            f"delta {row['delta_median']:.4f}  "
            f"|log sigma gap| {row['log_sigma_gap_median']:.4f}  "
            f"analytic {row['analytic_at_measured_delta']:.4f}  "
            f"({row['seconds']:.0f}s)"
        )
        # Never overwrite the committed sweep from a re-measurement: the point of re-measuring is to
        # compare the two, which is impossible if the second run silently replaces the first.
        results_path = out_dir / ("results_remeasured.json" if args.measure_only else "results.json")
        results_path.write_text(json.dumps({
            "experiment": "E1.1-remeasured" if args.measure_only else "E1.1",
            "commit": git_commit(),
            "host": platform.node(),
            "config": vars(args),
            "target_arch": TARGET_ARCH,
            "draft_arch": DRAFT_ARCH,
            "rows": rows,
        }, indent=2))

    print(f"\n{'d':>5} {'acceptance':>11} {'delta':>8} {'logsig':>8} {'analytic':>9} {'certain':>8}")
    for r in rows:
        print(f"{r['d']:>5} {r['acceptance']:>11.4f} {r['delta_median']:>8.4f} "
              f"{r['log_sigma_gap_median']:>8.4f} "
              f"{r['analytic_at_measured_delta']:>9.4f} {r['fraction_certain_accept']:>8.3f}")
    print(f"\nwrote {results_path}")
    if not args.measure_only:
        print(f"wrote {plot_results(rows, Path(args.figure))}")


if __name__ == "__main__":
    main()
