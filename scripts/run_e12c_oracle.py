#!/usr/bin/env python
"""E1.2c -- the oracle-disagreement subspace: can perfect knowledge of delta escape the frontier?

E1.2b tested five ways of choosing which ``k`` of ``d`` directions to verify and found them all on
one acceptance/fidelity curve. The obvious objection is that all five are heuristics:

    "What if the optimal subspace is simply not among your five?"

This script answers it by building the subspace that is optimal *by construction* for the quantity
the conservation identity is about. The identity says acceptance is set by how much of the
draft/target gap ``delta`` the statistic captures and the fidelity cost by how much it misses. So we
measure ``delta`` on the very decisions being verified, form ``E[delta delta^T]``, and verify its
top-``k`` eigenvectors -- the rank-``k`` subspace capturing the most disagreement in the mean-square
sense.

This is an oracle, not a proposal. It needs the target's predictions before the target has run,
which is precisely what a verifier cannot have. That is what makes it the right control: it is an
upper bound on what any linear rule of the same rank could hope to capture, so if it lands on the
same curve as a random subspace, the curve is a property of the problem rather than of our
imagination.

Two outcomes are informative and both are worth having:

* the oracle sits on the random curve -- the frontier is real, and the theorem's empirical support
  goes from "five heuristics agree" to "even the optimal linear rule agrees";
* the oracle sits *below* the curve -- there is room under the frontier that heuristics miss, the
  Gaussian corollary's scope is narrower than the mixture case, and that is a finding to report.

Reuses E1.2b's checkpoints and measurement path, so the numbers are directly comparable; the random
baseline is read from E1.2b's saved results rather than re-measured.

Run:  python scripts/run_e12c_oracle.py --device cuda --out experiments/E1.2c-oracle
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from herald.data.timeseries import build_bundle, load_etth1
from herald.eval import measure_fidelity
from herald.eval.fidelity import TokensWithSignal
from herald.models.ts_mar import random_masking
from herald.train.timeseries import CachedTokenDataset, load_checkpoint, set_seed
from herald.verifiers.structured import disagreement_matrix, top_eigen_projection


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def ks_for(d: int) -> list[int]:
    """The same ladder E1.2b uses, so points can be compared at matched ``k`` as well."""
    return [k for k in (1, 2, 4, 8, 16, 32) if k < d]


@torch.no_grad()
def collect_delta(draft, target, dataset, device: str, batch_size: int, max_batches: int,
                  seed: int) -> torch.Tensor:
    """Per-decision draft/target mean gaps ``(N, d)``.

    Collected on the same masked decisions the fidelity measurement scores, with the same seed, so
    the oracle is built from the disagreement it will actually be tested against. Giving the oracle
    an independent calibration split would be more realistic, but realism is not the point here --
    the aim is an upper bound, and the tightest one comes from letting it see everything.
    """
    draft.eval()
    target.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    gaps = []
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
            gaps.append((p.mean - q.mean)[mask].cpu())
    return torch.cat(gaps)


def baseline_at(rows: list[dict], acceptance: float, key: str) -> float | None:
    """E1.2b's random-projection curve at a given acceptance; no extrapolation."""
    pts = sorted((r["acceptance"], r[key]) for r in rows if r["rule"] == "random")
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    if not xs or not (xs[0] <= acceptance <= xs[-1]):
        return None
    return float(np.interp(acceptance, xs, ys))


def run_one_dim(d: int, bundle, reference: dict | None, args) -> dict:
    device = args.device
    dim_dir = Path(args.checkpoints) / f"d{d}"
    t0 = time.time()

    tokenizer = load_checkpoint(dim_dir / "tokenizer.pt").to(device)
    tok_test = CachedTokenDataset(bundle.test, tokenizer, device=device)
    paired = TokensWithSignal(tok_test, bundle.test)
    draft = load_checkpoint(dim_dir / "draft.pt").to(device)
    target = load_checkpoint(dim_dir / "target.pt").to(device)

    common = {"device": device, "batch_size": args.batch_size, "max_batches": args.eval_batches,
              "seed": args.seed, "tokenizer": tokenizer}

    delta = collect_delta(draft, target, tok_test, device, args.batch_size,
                          args.eval_batches, args.seed)
    m = disagreement_matrix(delta.to(device))
    evals = torch.linalg.eigvalsh(m).flip(0).clamp_min(0)
    captured = (evals.cumsum(0) / evals.sum()).cpu().numpy()
    basis = top_eigen_projection(m, d, largest=True)
    print(f"  d={d}: delta from {delta.shape[0]} decisions; top-1 direction carries "
          f"{100 * captured[0]:.1f}% of E[||delta||^2]", flush=True)

    rows = []
    for k in ks_for(d):
        a = basis[:k].contiguous()
        row = measure_fidelity(draft, target, paired, projection=a, **common)
        row.update(k=k, rule="oracle_disagreement",
                   delta_energy_captured=float(captured[k - 1]))
        if reference is not None:
            for key in ("excess_nll", "signal_energy"):
                base = baseline_at(reference["rows"], row["acceptance"], key)
                row[f"{key}_random_at_matched_acceptance"] = base
                row[f"{key}_residual"] = None if base is None else row[key] - base
        rows.append(row)
        resid = row.get("excess_nll_residual")
        tail = "" if resid is None else f"  resid {resid:+.3f}"
        print(f"  d={d:<4d} k={k:<3d} oracle  acc {row['acceptance']:.4f}  "
              f"nats {row['excess_nll']:+.4f}  energy {row['signal_energy']:+.4f}  "
              f"captures {100 * captured[k - 1]:5.1f}%{tail}", flush=True)

    return {"d": d, "rows": rows,
            "delta_spectrum_cumulative": [float(x) for x in captured],
            "seconds": time.time() - t0}


def summarize(results: list[dict], reference: list[dict] | None) -> list[str]:
    lines = ["", "E1.2c: does the oracle-disagreement subspace escape the frontier?", ""]
    ref_by_d = {r["d"]: r for r in (reference or [])}
    for r in results:
        res = [row["excess_nll_residual"] for row in r["rows"]
               if row.get("excess_nll_residual") is not None]
        if not res:
            lines.append(f"  d={r['d']}: no overlap with the random curve's acceptance range.")
            continue
        ref = ref_by_d.get(r["d"])
        span = None
        if ref is not None:
            costs = [x["excess_nll"] for x in ref["rows"] if x["rule"] != "exact"]
            span = max(costs) - min(costs)
        worst = max(res, key=abs)
        pct = "" if not span else f" ({100 * abs(worst) / span:.1f}% of the E1.2b range)"
        side = "below" if worst < 0 else "above"
        lines.append(f"  d={r['d']}: largest departure {abs(worst):.3f} nats {side} the random "
                     f"curve{pct}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128])
    ap.add_argument("--checkpoints", default="experiments/E1.1-dsweep")
    ap.add_argument("--reference", default="experiments/E1.2b-gpu/results.json",
                    help="E1.2b results, used for the random baseline at matched acceptance")
    ap.add_argument("--out", default="experiments/E1.2c-oracle")
    ap.add_argument("--window", type=int, default=96)
    ap.add_argument("--eval-batches", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    bundle = build_bundle(load_etth1(), window=args.window)

    ref_path = Path(args.reference)
    reference = json.loads(ref_path.read_text())["results"] if ref_path.exists() else None
    ref_by_d = {r["d"]: r for r in (reference or [])}

    t0 = time.time()
    results = [run_one_dim(d, bundle, ref_by_d.get(d), args) for d in args.dims]
    print("\n".join(summarize(results, reference)))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "E1.2c-oracle",
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
