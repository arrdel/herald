#!/usr/bin/env python
"""E0.6 -- the draft/target separation of a real image-model pair.

Everything in this project's dimension sweep is a statement about ``M = ||mu_p - mu_q|| / sigma``,
and until now ``M`` had never been measured on two real image models: the sweep says what happens as
a function of ``d``, not where a published pair sits. This closes that gap for MAR-B against MAR-H at
the KL16 token width ``d = 16``.

It reads an ``E0.3``-style capture, which stores the final-step posterior mean and standard deviation
that each model's diffusion head produces at the *same* context, and reports:

* the per-coordinate gap and its normalisation ``M / sqrt(d)``, comparable across dimensions;
* ``M`` itself, per decision rather than pooled, since acceptance is a nonlinear function of it;
* the acceptance the shared-covariance Gaussian model of Proposition 2 predicts from ``M``;
* the acceptance the exact diagonal-Gaussian rule actually attains on the same tokens.

The last two differ only through the assumptions Proposition 2 makes and the capture does not, so
their gap measures how much the idealisation costs on a real pair. Predicted acceptance is averaged
over decisions rather than evaluated at the mean ``M``: ``2 - 2*Phi(M/2)`` is nonlinear, so
substituting an average separation is not the same as averaging the implied acceptance.

Example:
    python scripts/run_e06.py --capture experiments/E0.6-mar-delta/capture.pt \
        --out experiments/E0.6-mar-delta
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm

from herald.verifiers import (
    TokenGaussian,
    acceptance_from_log_ratio,
    exact_log_ratio,
)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def bootstrap_ci(values: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean, over independent token decisions."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def analyse(capture: Path, out: Path) -> dict:
    blob = torch.load(capture, map_location="cpu", weights_only=True)
    draft = TokenGaussian(blob["draft_mean"], blob["draft_std"], blob["draft_log_sigma_accum"])
    target = TokenGaussian(blob["target_mean"], blob["target_std"], blob["target_log_sigma_accum"])
    x = blob["x"]
    n, dim = x.shape

    gap = (target.mean - draft.mean).numpy().astype(np.float64)          # (N, d)
    sd_p = target.std.numpy().astype(np.float64)
    sd_q = draft.std.numpy().astype(np.float64)

    # Proposition 2 assumes a single shared sigma. The two heads do not share one, so we pool per
    # coordinate; reporting the disagreement between sd_p and sd_q separately says how much that
    # assumption is being asked to absorb.
    sd_bar = np.sqrt(0.5 * (sd_p**2 + sd_q**2))

    z = gap / sd_bar                                                      # standardised gap
    M = np.linalg.norm(z, axis=1)                                         # (N,) per-decision
    per_coord = np.abs(z)                                                 # |delta| / sigma

    predicted = 2.0 - 2.0 * norm.cdf(M / 2.0)                             # per decision, then mean
    measured = acceptance_from_log_ratio(exact_log_ratio(x, draft, target)).numpy().astype(np.float64)

    sd_log_ratio = np.log(sd_p / sd_q)

    res = {
        "commit": git_commit(),
        "capture": str(capture),
        "n_token_decisions": int(n),
        "token_dim": int(dim),
        "per_coordinate_gap_over_sigma": {
            "mean": float(per_coord.mean()),
            "median": float(np.median(per_coord)),
            "p95": float(np.percentile(per_coord, 95)),
        },
        "separation_M": {
            "mean": float(M.mean()),
            "median": float(np.median(M)),
            "p05": float(np.percentile(M, 5)),
            "p95": float(np.percentile(M, 95)),
        },
        "M_over_sqrt_d": {
            "mean": float((M / np.sqrt(dim)).mean()),
            "median": float(np.median(M / np.sqrt(dim))),
        },
        "acceptance_predicted_gaussian": {
            "mean": float(predicted.mean()),
            "ci95": bootstrap_ci(predicted),
        },
        "acceptance_measured_exact": {
            "mean": float(measured.mean()),
            "ci95": bootstrap_ci(measured),
        },
        "acceptance_at_mean_M": float(2.0 - 2.0 * norm.cdf(M.mean() / 2.0)),
        "shared_covariance_check": {
            "mean_abs_log_sd_ratio": float(np.abs(sd_log_ratio).mean()),
            "median_sd_ratio": float(np.exp(np.median(sd_log_ratio))),
            "p95_abs_log_sd_ratio": float(np.percentile(np.abs(sd_log_ratio), 95)),
        },
    }

    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=2))

    pc = res["per_coordinate_gap_over_sigma"]
    sm = res["separation_M"]
    print(f"{n} token decisions from a real MAR-B / MAR-H pair, d = {dim}\n")
    print(f"per-coordinate gap  |delta|/sigma : mean {pc['mean']:.4f}  median {pc['median']:.4f}"
          f"  p95 {pc['p95']:.4f}")
    print(f"separation          M            : mean {sm['mean']:.3f}   median {sm['median']:.3f}"
          f"  [p5 {sm['p05']:.3f}, p95 {sm['p95']:.3f}]")
    print(f"                    M/sqrt(d)     : mean {res['M_over_sqrt_d']['mean']:.4f}")
    ap, am = res["acceptance_predicted_gaussian"], res["acceptance_measured_exact"]
    print(f"\nacceptance predicted (Prop. 2)   : {ap['mean']:.4f}  "
          f"95% CI [{ap['ci95'][0]:.4f}, {ap['ci95'][1]:.4f}]")
    print(f"acceptance measured  (exact rule): {am['mean']:.4f}  "
          f"95% CI [{am['ci95'][0]:.4f}, {am['ci95'][1]:.4f}]")
    print(f"(acceptance evaluated at the mean M instead: {res['acceptance_at_mean_M']:.4f})")
    cc = res["shared_covariance_check"]
    print(f"\nshared-covariance check: mean |log sd_p/sd_q| = {cc['mean_abs_log_sd_ratio']:.4f}, "
          f"median ratio {cc['median_sd_ratio']:.3f}")
    print(f"\nwrote {out / 'results.json'}")
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--capture", type=Path, default=Path("experiments/E0.6-mar-delta/capture.pt"))
    p.add_argument("--out", type=Path, default=Path("experiments/E0.6-mar-delta"))
    args = p.parse_args()
    analyse(args.capture, args.out)


if __name__ == "__main__":
    main()
