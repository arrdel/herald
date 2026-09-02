#!/usr/bin/env python
"""E1.1b -- is E1.1's acceptance collapse an artifact of a fixed training budget?

E1.1 trained every token dimension for the same 4,000 steps. Larger-d models have more to learn on
that budget, so part of the observed decay could be relative undertraining rather than dimension.
Flat per-coordinate disagreement argued against it but did not settle it, and it is the one serious
hole in the result.

Rather than repeat the whole sweep at matched validation NLL -- awkward, because NLL is not
comparable across d without a per-dimension normalisation that is itself arguable -- this attacks
the confound head on: train the two extremes far longer than E1.1 did, checkpointing along the way,
and plot acceptance against training budget.

Prediction if E1.1 is sound: acceptance plateaus well before 4,000 steps at *both* d=16 and d=128,
so the gap between them is not a gap in convergence.
Prediction if the confound is real: d=128 keeps climbing past 4,000 steps and the curves converge.

Draft and target are always compared at matched step counts. A draft trained longer than its target
would change their disagreement for reasons unrelated to either hypothesis.

Run:  python scripts/run_e11b.py --dims 16 128 --mar-steps 16000
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

E11_BUDGET = 4000


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def run_one_dim(d: int, bundle, args, out_dir: Path) -> dict:
    device = args.device
    n_tokens = args.window // args.patch
    dim_dir = out_dir / f"d{d}"
    t0 = time.time()

    tokenizer = TimeSeriesTokenizer(
        TokenizerConfig(n_channels=bundle.n_channels, patch=args.patch, latent_dim=d,
                        hidden=args.tok_hidden)
    )
    train_module(
        tokenizer, bundle.train, tokenizer_loss,
        TrainConfig(steps=args.tok_steps, batch_size=args.batch_size, device=device,
                    seed=args.seed, num_workers=0, out_dir=dim_dir, eval_every=0,
                    log_every=max(1, args.tok_steps // 2)),
        tag="tokenizer",
    )

    tok_train = CachedTokenDataset(bundle.train, tokenizer, device=device)
    tok_val = CachedTokenDataset(bundle.val, tokenizer, device=device)
    tok_test = CachedTokenDataset(bundle.test, tokenizer, device=device)

    val_nll: dict[str, list] = {}
    for name, arch in (("target", TARGET_ARCH), ("draft", DRAFT_ARCH)):
        cfg = TrainConfig(
            steps=args.mar_steps, batch_size=args.batch_size, device=device,
            seed=args.seed + (0 if name == "target" else 1), num_workers=0, out_dir=dim_dir,
            eval_every=args.checkpoint_every, log_every=args.checkpoint_every,
            checkpoint_every=args.checkpoint_every,
        )
        train_module(
            TimeSeriesMAR(MARConfig(latent_dim=d, n_tokens=n_tokens, **arch)),
            tok_train, mar_loss, cfg, val_dataset=tok_val, tag=name,
        )
        val_nll[name] = [
            {"step": h["step"], "nll": h.get("nll", h.get("loss"))}
            for h in cfg.history if h.get("split") == "val"
        ]
        print(f"  d={d} {name}: trained {args.mar_steps} steps")

    # The final budget is saved as ``{tag}.pt`` rather than ``{tag}_step{N}.pt``, so it has to be
    # asked for by name; leaving it out would cut the plateau off exactly where it matters most.
    steps = list(range(args.checkpoint_every, args.mar_steps, args.checkpoint_every))

    def acceptance_at(draft_step: int, target_step: int) -> dict:
        def ckpt(tag: str, step: int) -> Path:
            return dim_dir / (f"{tag}.pt" if step == args.mar_steps else f"{tag}_step{step}.pt")

        return measure_acceptance(
            load_checkpoint(ckpt("draft", draft_step)).to(device),
            load_checkpoint(ckpt("target", target_step)).to(device),
            tok_test, device=device, batch_size=args.batch_size,
            max_batches=args.eval_batches, seed=args.seed,
        )

    trace = []
    for step in [*steps, args.mar_steps]:
        row = acceptance_at(step, step)
        row["step"] = step
        trace.append(row)
        print(f"    d={d:<4d} step {step:>6d}  acceptance {row['acceptance']:.4f}  "
              f"delta {row['delta_median']:.4f}")

    # Validation NLL turns out to be non-monotone, so the trace above is partly a trace over how far
    # each model has overfit rather than over how well it has converged. Each model's own
    # best-validation checkpoint is the one that would actually be deployed, so measure there too.
    # NLL is compared only within a single model: across d it is a density in a different dimension.
    best = {}
    for name in ("target", "draft"):
        usable = [e for e in val_nll[name] if e["step"] in {*steps, args.mar_steps}]
        best[name] = min(usable, key=lambda e: e["nll"]) if usable else None

    best_row = None
    if best["draft"] and best["target"]:
        best_row = acceptance_at(best["draft"]["step"], best["target"]["step"])
        best_row.update(
            draft_step=best["draft"]["step"], target_step=best["target"]["step"],
            draft_nll=best["draft"]["nll"], target_nll=best["target"]["nll"],
        )
        print(f"    d={d:<4d} best-val draft@{best['draft']['step']} "
              f"target@{best['target']['step']}  acceptance {best_row['acceptance']:.4f}  "
              f"delta {best_row['delta_median']:.4f}")

    return {"d": d, "trace": trace, "best_val": best_row, "val_nll": val_nll,
            "seconds": time.time() - t0}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dims", type=int, nargs="+", default=[16, 128])
    p.add_argument("--dataset", default="etth1", choices=["etth1", "synthetic"])
    p.add_argument("--window", type=int, default=96)
    p.add_argument("--patch", type=int, default=8)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--tok-steps", type=int, default=2000)
    p.add_argument("--tok-hidden", type=int, default=128)
    p.add_argument("--mar-steps", type=int, default=16000)
    p.add_argument("--checkpoint-every", type=int, default=2000)
    p.add_argument("--eval-batches", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    # The host runs ~64 cores at a load average north of 60. Torch's default of one thread per core
    # then spends its time contending rather than computing: measured on this box, these models step
    # 22x slower at 32 threads than at 8. This is a property of the machine, not of the experiment.
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--out", default="experiments/E1.1b-convergence")
    args = p.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    set_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    series = load_etth1() if args.dataset == "etth1" else synthetic_ar(n_steps=20_000, n_channels=7)
    bundle = build_bundle(series, window=args.window, stride=args.stride)
    print(f"{args.dataset}: {bundle.describe()}")
    print(f"device {args.device}, dims {args.dims}, budget {args.mar_steps} steps "
          f"({args.mar_steps / E11_BUDGET:.0f}x E1.1's {E11_BUDGET})\n")

    results = []
    for d in args.dims:
        results.append(run_one_dim(d, bundle, args, out_dir))
        (out_dir / "results.json").write_text(json.dumps({
            "experiment": "E1.1b",
            "commit": git_commit(),
            "host": platform.node(),
            "config": vars(args),
            "e11_budget_for_reference": E11_BUDGET,
            "results": results,
        }, indent=2))

    print("\nacceptance vs training budget")
    print(f"{'step':>8}" + "".join(f"{'d=' + str(r['d']):>12}" for r in results))
    for i, row in enumerate(results[0]["trace"]):
        cells = "".join(f"{r['trace'][i]['acceptance']:>12.4f}" for r in results)
        marker = "   <- E1.1's budget" if row["step"] == E11_BUDGET else ""
        print(f"{row['step']:>8}{cells}{marker}")
    if all(r.get("best_val") for r in results):
        cells = "".join(f"{r['best_val']['acceptance']:>12.4f}" for r in results)
        print(f"{'best-val':>8}{cells}   <- each model at its own best validation NLL")
    print(f"\nwrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
