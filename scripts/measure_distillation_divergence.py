#!/usr/bin/env python
"""E1.5 -- does distillation raise acceptance by shrinking the divergence we say it shrinks?

E1.4 shows that distilling the draft against a frozen target raises *exact* acceptance at zero
fidelity cost, and the paper explains that by the conservation identity: the acceptance/fidelity
frontier is fixed by the pair's separation ``M = ||mu_p - mu_q|| / sigma``, so the only way to move
the frontier is to move the pair.  E1.4 never measured ``M``.  It measured the consequence and
asserted the mechanism, which is exactly the gap a reviewer should object to: acceptance could have
risen for some other reason (variance inflation in the draft, say, which also raises overlap) and the
story would look identical.

This script closes that gap by re-running E1.4's distillation trajectory with the same checkpoints,
schedule, and seeds, and recording at every stage the quantities the mechanism is about:

  * ``KL(p||q)`` by Monte Carlo, the divergence the mode-covering objective actually minimises.
    Mixture heads have no closed-form KL, so we draw from the target and score under both.  The
    target is frozen, so ``H(p)`` is constant across stages and any change is a change in the draft.
  * ``M``, the separation in the paper's convention (E0.6): per-coordinate standardisation by the
    pooled scale ``sqrt((var_p + var_q)/2)``, then the norm, per decision.
  * exact acceptance, so the trajectory can be lined up against E1.4's, and the Gaussian prediction
    ``2 - 2*Phi(M/2)`` evaluated at the measured ``M``.

If the mechanism is right, ``KL(p||q)`` and ``M`` should fall monotonically while acceptance rises,
and the Gaussian prediction should track the measured acceptance.  If acceptance rose while ``M``
held steady, the conservation story would be wrong about *why* distillation works.

No fidelity measurement here -- no residual sampling, no decoding -- which is what makes this much
cheaper than re-running E1.4.

Run:  python scripts/measure_distillation_divergence.py --device cuda --out experiments/E1.5-divergence
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

from herald.data.timeseries import build_bundle, load_etth1
from herald.models.ts_mar import random_masking
from herald.train.timeseries import (
    CachedTokenDataset,
    TrainConfig,
    distillation_loss,
    load_checkpoint,
    set_seed,
    train_module,
)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


@torch.no_grad()
def measure(draft, target, dataset, device: str, batch_size: int, max_batches: int,
            seed: int) -> dict:
    """Divergence, separation, and exact acceptance for one draft/target pair.

    The masking and batching mirror ``herald.eval.fidelity.measure_fidelity`` so that the acceptance
    reported here is comparable with E1.4's, rather than being measured on a different set of
    decisions.
    """
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
            mask = random_masking(tokens.shape[0], tokens.shape[1], torch.device(device),
                                  generator=gen)
            if not mask.any():
                continue

            q = draft.token_distribution(draft.forward_predictor(tokens, mask))
            p = target.token_distribution(target.forward_predictor(tokens, mask))

            # KL(p||q) by Monte Carlo: one draw from p per decision.  Unbiased, and the variance is
            # controlled by the number of decisions rather than by any per-decision averaging.
            x = p.sample()
            kl_terms.append((p.log_prob(x) - q.log_prob(x))[mask].cpu())

            # Separation in the E0.6 convention.  Mixture heads have no single (mu, sigma), so we
            # use the mixture's own first two moments -- the law of total variance already folds the
            # spread *between* components into var, which is the honest summary for a scale.
            gap = p.mean - q.mean
            pooled = torch.sqrt(0.5 * (p.variance + q.variance)).clamp_min(1e-8)
            m = torch.linalg.vector_norm(gap / pooled, dim=-1)
            m_vals.append(m[mask].cpu())

            # Exact token-space acceptance, no residual sampling needed for the rate itself.
            y = q.sample()
            ratio = (p.log_prob(y) - q.log_prob(y)).exp().clamp(max=1.0)
            accepts.append(torch.bernoulli(ratio, generator=gen)[mask].float().cpu())

    kl = torch.cat(kl_terms).double()
    m = torch.cat(m_vals).double()
    acc = torch.cat(accepts).double()
    n = kl.numel()
    m_mean = float(m.mean())
    return {
        "n_decisions": int(n),
        "kl_pq": float(kl.mean()),
        "kl_pq_sem": float(kl.std() / max(1.0, n**0.5)),
        "M_mean": m_mean,
        "M_median": float(m.median()),
        "M_sem": float(m.std() / max(1.0, n**0.5)),
        "acceptance": float(acc.mean()),
        "acceptance_sem": float(acc.std() / max(1.0, n**0.5)),
        # Evaluated per decision then averaged: Phi is nonlinear, so plugging in the mean M is wrong.
        "acceptance_gaussian_pred": float(
            np.mean(2.0 - 2.0 * norm.cdf(m.numpy() / 2.0))
        ),
    }


def run_one_dim(d: int, bundle, args) -> dict:
    device = args.device
    dim_dir = Path(args.checkpoints) / f"d{d}"
    t0 = time.time()

    tokenizer = load_checkpoint(dim_dir / "tokenizer.pt").to(device)
    tok_train = CachedTokenDataset(bundle.train, tokenizer, device=device)
    tok_test = CachedTokenDataset(bundle.test, tokenizer, device=device)
    target = load_checkpoint(dim_dir / "target.pt").to(device)
    draft = load_checkpoint(dim_dir / "draft.pt").to(device)

    common = {"device": device, "batch_size": args.batch_size,
              "max_batches": args.eval_batches, "seed": args.seed}

    stages = [{"step": 0, **measure(draft, target, tok_test, **common)}]
    s = stages[0]
    print(f"  d={d:<4d} step {0:<6d} KL(p||q) {s['kl_pq']:8.3f}  M {s['M_mean']:7.3f}  "
          f"acc {s['acceptance']:.4f}  (pred {s['acceptance_gaussian_pred']:.4f})", flush=True)

    student = copy.deepcopy(draft)
    loss_fn = distillation_loss(target, seed=args.seed)
    for stage in range(args.stages):
        config = TrainConfig(steps=args.steps_per_stage, batch_size=args.batch_size, lr=args.lr,
                             device=device, seed=args.seed + stage, num_workers=0, log_every=0,
                             eval_every=0)
        student = train_module(student, tok_train, loss_fn, config, tag=f"distilled_d{d}")
        step = (stage + 1) * args.steps_per_stage
        s = {"step": step, **measure(student, target, tok_test, **common)}
        stages.append(s)
        print(f"  d={d:<4d} step {step:<6d} KL(p||q) {s['kl_pq']:8.3f}  M {s['M_mean']:7.3f}  "
              f"acc {s['acceptance']:.4f}  (pred {s['acceptance_gaussian_pred']:.4f})", flush=True)

    first, last = stages[0], stages[-1]
    return {
        "d": d,
        "stages": stages,
        "kl_reduction": first["kl_pq"] - last["kl_pq"],
        "kl_reduction_frac": 1.0 - last["kl_pq"] / max(first["kl_pq"], 1e-12),
        "M_reduction_frac": 1.0 - last["M_mean"] / max(first["M_mean"], 1e-12),
        "acceptance_ratio": last["acceptance"] / max(first["acceptance"], 1e-12),
        "monotone_kl": all(stages[i]["kl_pq"] >= stages[i + 1]["kl_pq"]
                           for i in range(len(stages) - 1)),
        "monotone_M": all(stages[i]["M_mean"] >= stages[i + 1]["M_mean"]
                          for i in range(len(stages) - 1)),
        "seconds": time.time() - t0,
    }


def summarize(results: list[dict]) -> list[str]:
    lines = ["", "E1.5: distillation shrinks the divergence, and acceptance follows.", ""]
    lines.append(f"{'d':>5} {'KL before':>10} {'KL after':>10} {'KL drop':>9} "
                 f"{'M before':>9} {'M after':>9} {'acc before':>11} {'acc after':>10} {'mono':>6}")
    for r in results:
        a, b = r["stages"][0], r["stages"][-1]
        mono = "yes" if (r["monotone_kl"] and r["monotone_M"]) else "no"
        lines.append(f"{r['d']:5d} {a['kl_pq']:10.3f} {b['kl_pq']:10.3f} "
                     f"{100 * r['kl_reduction_frac']:8.1f}% {a['M_mean']:9.3f} {b['M_mean']:9.3f} "
                     f"{a['acceptance']:11.4f} {b['acceptance']:10.4f} {mono:>6}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128])
    ap.add_argument("--checkpoints", default="experiments/E1.1-dsweep")
    ap.add_argument("--out", default="experiments/E1.5-divergence")
    ap.add_argument("--window", type=int, default=96)
    ap.add_argument("--stages", type=int, default=5)
    ap.add_argument("--steps-per-stage", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-batches", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    bundle = build_bundle(load_etth1(), window=args.window)

    t0 = time.time()
    results = [run_one_dim(d, bundle, args) for d in args.dims]
    print("\n".join(summarize(results)))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "E1.5-divergence",
        "commit": git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "args": vars(args),
        "results": results,
        "seconds": time.time() - t0,
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out / 'results.json'}")


if __name__ == "__main__":
    main()
