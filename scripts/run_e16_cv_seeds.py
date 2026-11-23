#!/usr/bin/env python
# Copyright (C) 2026 Lenovo
# All Rights Reserved.
# Lenovo Confidential Restricted.

"""E1.6 -- Multi-seed and expanding chronological cross-validation sweep.

Runs the full dimension-indexed pipeline (tokenizer -> target & draft MAR -> exact acceptance ->
mode-covering KL distillation -> KL/M/exact-fidelity audit) for one (fold, seed, d) tuple or
aggregates completed runs.

Run a single task:
  python scripts/run_e16_cv_seeds.py --fold 3 --seed 0 --dim 16 --device cuda

Aggregate all completed tasks:
  python scripts/run_e16_cv_seeds.py --aggregate
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm
from torch.utils.data import DataLoader

from herald.data.timeseries import expanding_cv_bundles, load_etth1
from herald.eval import DRAFT_ARCH, TARGET_ARCH, measure_acceptance, measure_fidelity
from herald.eval.fidelity import TokensWithSignal
from herald.models.ts_mar import MARConfig, TimeSeriesMAR, random_masking
from herald.models.ts_tokenizer import TimeSeriesTokenizer, TokenizerConfig
from herald.train.timeseries import (
    CachedTokenDataset,
    TrainConfig,
    distillation_loss,
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


@torch.no_grad()
def measure_kl_and_m(
    draft: TimeSeriesMAR,
    target: TimeSeriesMAR,
    dataset: CachedTokenDataset,
    device: str,
    batch_size: int,
    max_batches: int,
    seed: int,
) -> dict:
    """Measure KL(p||q), pooled separation M, and Bernoulli acceptance rate on test batches."""
    draft.eval()
    target.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    kl_terms, m_vals, accepts = [], [], []
    devices = [torch.device(device)] if torch.device(device).type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        gen = torch.Generator(device=device).manual_seed(seed)
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            tokens = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
            mask = random_masking(tokens.shape[0], tokens.shape[1], torch.device(device), generator=gen)
            if not mask.any():
                continue

            q = draft.token_distribution(draft.forward_predictor(tokens, mask))
            p = target.token_distribution(target.forward_predictor(tokens, mask))

            x = p.sample()
            kl_terms.append((p.log_prob(x) - q.log_prob(x))[mask].cpu())

            gap = p.mean - q.mean
            pooled = torch.sqrt(0.5 * (p.variance + q.variance)).clamp_min(1e-8)
            m = torch.linalg.vector_norm(gap / pooled, dim=-1)
            m_vals.append(m[mask].cpu())

            y = q.sample()
            ratio = (p.log_prob(y) - q.log_prob(y)).exp().clamp(max=1.0)
            accepts.append(torch.bernoulli(ratio, generator=gen)[mask].float().cpu())

    kl = torch.cat(kl_terms).double()
    m = torch.cat(m_vals).double()
    acc = torch.cat(accepts).double()
    return {
        "n_decisions": int(kl.numel()),
        "kl_pq": float(kl.mean()),
        "M_mean": float(m.mean()),
        "M_median": float(m.median()),
        "acceptance_kl_probe": float(acc.mean()),
        "acceptance_gaussian_pred": float(np.mean(2.0 - 2.0 * norm.cdf(m.numpy() / 2.0))),
    }


def run_task(args: argparse.Namespace) -> dict:
    t0 = time.time()
    set_seed(args.seed)
    device = args.device
    d = args.dim
    n_tokens = args.window // args.patch

    series = load_etth1()
    bundles = expanding_cv_bundles(series, window=args.window, stride=args.stride, n_folds=args.n_folds)
    bundle = bundles[args.fold - 1]

    tok_cfg = TokenizerConfig(
        n_channels=bundle.n_channels, patch=args.patch, latent_dim=d, hidden=args.tok_hidden
    )
    tokenizer = TimeSeriesTokenizer(tok_cfg)
    train_module(
        tokenizer,
        bundle.train,
        tokenizer_loss,
        TrainConfig(
            steps=args.tok_steps,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed,
            num_workers=0,
            eval_every=0,
            log_every=0,
        ),
        tag=f"tok_f{args.fold}_s{args.seed}_d{d}",
    )

    tok_train = CachedTokenDataset(bundle.train, tokenizer, device=device)
    tok_val = CachedTokenDataset(bundle.val, tokenizer, device=device)
    tok_test = CachedTokenDataset(bundle.test, tokenizer, device=device)
    paired = TokensWithSignal(tok_test, bundle.test)

    target = TimeSeriesMAR(MARConfig(latent_dim=d, n_tokens=n_tokens, **TARGET_ARCH))
    train_module(
        target,
        tok_train,
        mar_loss,
        TrainConfig(
            steps=args.mar_steps,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed,
            num_workers=0,
            eval_every=args.mar_steps // 2,
            log_every=0,
        ),
        val_dataset=tok_val,
        tag=f"target_f{args.fold}_s{args.seed}_d{d}",
    )

    draft = TimeSeriesMAR(MARConfig(latent_dim=d, n_tokens=n_tokens, **DRAFT_ARCH))
    train_module(
        draft,
        tok_train,
        mar_loss,
        TrainConfig(
            steps=args.mar_steps,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed + 1,
            num_workers=0,
            eval_every=args.mar_steps // 2,
            log_every=0,
        ),
        val_dataset=tok_val,
        tag=f"draft_f{args.fold}_s{args.seed}_d{d}",
    )

    collapse = measure_acceptance(
        draft, target, tok_test, device=device, batch_size=args.batch_size, seed=args.seed
    )
    collapse["analytic_at_measured_delta"] = analytic_acceptance_shared_covariance(
        collapse["delta_median"], d
    )

    fid_cfg = {
        "device": device,
        "batch_size": args.batch_size,
        "max_batches": args.eval_batches,
        "seed": args.seed,
        "tokenizer": tokenizer,
    }
    fid_before = measure_fidelity(draft, target, paired, projection=None, **fid_cfg)
    kl_before = measure_kl_and_m(
        draft, target, tok_test, device=device, batch_size=args.batch_size,
        max_batches=args.eval_batches, seed=args.seed,
    )

    student = copy.deepcopy(draft)
    loss_fn = distillation_loss(target, seed=args.seed)
    for stage in range(args.stages):
        cfg = TrainConfig(
            steps=args.steps_per_stage,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            seed=args.seed + stage,
            num_workers=0,
            log_every=0,
            eval_every=0,
        )
        student = train_module(student, tok_train, loss_fn, cfg, tag=f"distill_f{args.fold}_s{args.seed}_d{d}")

    fid_after = measure_fidelity(student, target, paired, projection=None, **fid_cfg)
    kl_after = measure_kl_and_m(
        student, target, tok_test, device=device, batch_size=args.batch_size,
        max_batches=args.eval_batches, seed=args.seed,
    )

    gain = fid_after["acceptance"] / max(fid_before["acceptance"], 1e-12)
    kl_drop_frac = 1.0 - kl_after["kl_pq"] / max(kl_before["kl_pq"], 1e-12)
    return {
        "fold": args.fold,
        "seed": args.seed,
        "d": d,
        "split_windows": {
            "train": len(bundle.train),
            "val": len(bundle.val),
            "test": len(bundle.test),
        },
        "collapse": collapse,
        "distill": {
            "before": {**fid_before, **kl_before},
            "after": {**fid_after, **kl_after},
            "gain": float(gain),
            "kl_drop_frac": float(kl_drop_frac),
        },
        "seconds": time.time() - t0,
    }


def _stats(vals: list[float]) -> dict:
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(len(arr)),
    }


def aggregate(out_dir: Path) -> None:
    files = sorted(out_dir.glob("fold*_seed*_d*.json"))
    records = [json.loads(f.read_text()) for f in files]
    if not records:
        raise FileNotFoundError(f"no fold*_seed*_d*.json files in {out_dir}")

    dims = sorted({r["d"] for r in records})

    # 1) Multi-seed summary on fold 3 (primary 70/10/20 split)
    multiseed = {}
    for d in dims:
        rs = sorted([r for r in records if r["fold"] == 3 and r["d"] == d], key=lambda x: x["seed"])
        if not rs:
            continue
        multiseed[str(d)] = {
            "seeds": [r["seed"] for r in rs],
            "collapse_acc": _stats([r["collapse"]["acceptance"] for r in rs]),
            "delta_median": _stats([r["collapse"]["delta_median"] for r in rs]),
            "before_acc": _stats([r["distill"]["before"]["acceptance"] for r in rs]),
            "after_acc": _stats([r["distill"]["after"]["acceptance"] for r in rs]),
            "gain": _stats([r["distill"]["gain"] for r in rs]),
            "kl_before": _stats([r["distill"]["before"]["kl_pq"] for r in rs]),
            "kl_after": _stats([r["distill"]["after"]["kl_pq"] for r in rs]),
            "kl_drop_pct": _stats([100.0 * r["distill"]["kl_drop_frac"] for r in rs]),
            "M_before": _stats([r["distill"]["before"]["M_mean"] for r in rs]),
            "M_after": _stats([r["distill"]["after"]["M_mean"] for r in rs]),
            "excess_nll_after": _stats([r["distill"]["after"]["excess_nll"] for r in rs]),
        }

    # 2) Cross-validation summary across folds 1..3 (seed 0)
    cv = {}
    for d in dims:
        rs = sorted([r for r in records if r["seed"] == 0 and r["d"] == d], key=lambda x: x["fold"])
        if not rs:
            continue
        cv[str(d)] = {
            "folds": [r["fold"] for r in rs],
            "collapse_acc": _stats([r["collapse"]["acceptance"] for r in rs]),
            "delta_median": _stats([r["collapse"]["delta_median"] for r in rs]),
            "before_acc": _stats([r["distill"]["before"]["acceptance"] for r in rs]),
            "after_acc": _stats([r["distill"]["after"]["acceptance"] for r in rs]),
            "gain": _stats([r["distill"]["gain"] for r in rs]),
            "kl_before": _stats([r["distill"]["before"]["kl_pq"] for r in rs]),
            "kl_after": _stats([r["distill"]["after"]["kl_pq"] for r in rs]),
            "kl_drop_pct": _stats([100.0 * r["distill"]["kl_drop_frac"] for r in rs]),
        }

    summary = {
        "experiment": "E1.6-cv-seeds",
        "commit": git_commit(),
        "host": platform.node(),
        "records": records,
        "multiseed_fold3": multiseed,
        "cv_seed0": cv,
    }
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2))

    print("\n=== Multi-seed (Fold 3, mean +/- std) ===")
    print(f"{'d':>4} {'acc_col':>14} {'delta':>12} {'before':>14} {'after':>14} {'gain':>13} {'KL drop%':>12}")
    for d in dims:
        s = multiseed.get(str(d))
        if not s:
            continue
        print(
            f"{d:4d} "
            f"{s['collapse_acc']['mean']:.4f}+/-{s['collapse_acc']['std']:.4f} "
            f"{s['delta_median']['mean']:.3f}+/-{s['delta_median']['std']:.3f} "
            f"{s['before_acc']['mean']:.4f}+/-{s['before_acc']['std']:.4f} "
            f"{s['after_acc']['mean']:.4f}+/-{s['after_acc']['std']:.4f} "
            f"{s['gain']['mean']:.2f}+/-{s['gain']['std']:.2f}x "
            f"{s['kl_drop_pct']['mean']:.1f}+/-{s['kl_drop_pct']['std']:.1f}%"
        )

    print("\n=== 3-Fold Expanding Chronological CV (Seed 0, mean +/- std) ===")
    print(f"{'d':>4} {'acc_col':>14} {'delta':>12} {'before':>14} {'after':>14} {'gain':>13} {'KL drop%':>12}")
    for d in dims:
        s = cv.get(str(d))
        if not s:
            continue
        print(
            f"{d:4d} "
            f"{s['collapse_acc']['mean']:.4f}+/-{s['collapse_acc']['std']:.4f} "
            f"{s['delta_median']['mean']:.3f}+/-{s['delta_median']['std']:.3f} "
            f"{s['before_acc']['mean']:.4f}+/-{s['before_acc']['std']:.4f} "
            f"{s['after_acc']['mean']:.4f}+/-{s['after_acc']['std']:.4f} "
            f"{s['gain']['mean']:.2f}+/-{s['gain']['std']:.2f}x "
            f"{s['kl_drop_pct']['mean']:.1f}+/-{s['kl_drop_pct']['std']:.1f}%"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", type=int, default=3, choices=[1, 2, 3])
    ap.add_argument("--n-folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--window", type=int, default=96)
    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--tok-steps", type=int, default=2000)
    ap.add_argument("--tok-hidden", type=int, default=128)
    ap.add_argument("--mar-steps", type=int, default=4000)
    ap.add_argument("--stages", type=int, default=5)
    ap.add_argument("--steps-per-stage", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-batches", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="experiments/E1.6-cv-seeds")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.aggregate:
        aggregate(out_dir)
        return

    torch.set_num_threads(args.threads)
    res = run_task(args)
    out_path = out_dir / f"fold{args.fold}_seed{args.seed}_d{args.dim}.json"
    out_path.write_text(json.dumps(res, indent=2))
    print(
        f"[fold={args.fold} seed={args.seed} d={args.dim}] "
        f"col={res['collapse']['acceptance']:.4f} delta={res['collapse']['delta_median']:.3f} "
        f"distill={res['distill']['before']['acceptance']:.4f}->{res['distill']['after']['acceptance']:.4f} "
        f"({res['distill']['gain']:.2f}x) kl_drop={100*res['distill']['kl_drop_frac']:.1f}% "
        f"({res['seconds']:.1f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
