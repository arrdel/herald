#!/usr/bin/env python
"""E0.8 -- the analytic reference curves, and the finite-d law for random subspaces.

Three things the paper asserted from measurement alone, established here without training anything.
All of it is closed-form or Monte Carlo over synthetic Gaussians, so it is a reference the learned
testbed can be checked against rather than another system with its own confounds.

1. **The dimensional collapse, free of any tokenizer.** For ``p, q`` shared-covariance Gaussians with
   a fixed per-coordinate gap, exact acceptance is ``2 - 2*Phi(delta*sqrt(d)/2)`` and nothing else
   moves. Sweeping ``d`` to 4096 gives the curve the trained sweep is a noisy sample of, and a direct
   Monte Carlo of the accept/reject rule confirms the closed form rather than assuming it.

2. **Acceptance under a random k-dimensional subspace, at finite d.** The paper previously argued
   this by substituting ``E||A delta||^2 = (k/d)||delta||^2`` into the acceptance formula, which is
   wrong because ``2 - 2*Phi(.)`` is nonlinear: ``E[f(Z)] != f(E[Z])``. The correct statement uses
   ``||A delta||^2 / ||delta||^2 ~ Beta(k/2, (d-k)/2)``. Doing it properly is not a repair but an
   improvement: with ``M = c*sqrt(d)``, the verified separation ``M^2 Z`` converges in distribution to
   ``c^2 * chi^2_k``, which contains no ``d`` at all. Acceptance depending on ``k`` alone stops being
   an empirical coincidence and becomes a limit, with an ``O(1/d)`` finite-size correction this
   quantifies.

3. **A feasibility map.** Acceptance contours over ``(d, delta/sigma)``, which is the engineering
   read of the whole analysis: it says what per-coordinate agreement a pair must reach to be worth
   verifying at a given token width.

Example:
    python scripts/run_e08.py --out experiments/E0.8-analytic
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.stats import beta, chi2, norm


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def acceptance_closed_form(M: np.ndarray | float) -> np.ndarray | float:
    """``E_q[min(1, p/q)] = 2 - 2*Phi(M/2)`` for shared-covariance Gaussians separated by ``M``."""
    return 2.0 - 2.0 * norm.cdf(np.asarray(M) / 2.0)


def acceptance_monte_carlo(M: float, d: int, n: int, rng: np.random.Generator) -> float:
    """The accept/reject rule run directly, to check the closed form rather than trust it.

    Put the whole separation on the first coordinate: the rule depends on the means only through
    ``M``, so this is fully general and keeps the simulation cheap.
    """
    mu = np.zeros(d)
    mu[0] = M
    x = rng.standard_normal((n, d))                       # x ~ q = N(0, I)
    log_ratio = -0.5 * ((x - mu) ** 2).sum(1) + 0.5 * (x**2).sum(1)
    return float(np.minimum(1.0, np.exp(log_ratio)).mean())


def random_subspace_acceptance(M: float, d: int, k: int, n: int,
                               rng: np.random.Generator) -> dict[str, float]:
    """Exact (Beta), naive (plug-in), and limiting (chi-squared) acceptance for a random subspace."""
    z = beta.rvs(k / 2.0, (d - k) / 2.0, size=n, random_state=rng)
    exact = float(np.mean(acceptance_closed_form(M * np.sqrt(z))))
    naive = float(acceptance_closed_form(M * np.sqrt(k / d)))
    # Limit: d*Z -> chi^2_k, so M^2 Z -> (M^2/d) chi^2_k = c^2 chi^2_k with c = M/sqrt(d).
    c2 = M**2 / d
    limit = float(np.mean(acceptance_closed_form(np.sqrt(c2 * chi2.rvs(k, size=n, random_state=rng)))))
    return {"exact_beta": exact, "naive_plugin": naive, "chi2_limit": limit,
            "plugin_rel_error_pct": 100.0 * abs(naive - exact) / exact}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="experiments/E0.8-analytic")
    p.add_argument("--n-mc", type=int, default=200_000)
    p.add_argument("--delta", type=float, default=0.578,
                   help="per-coordinate gap in units of sigma; 0.578 is the testbed's M/sqrt(d)")
    args = p.parse_args()
    rng = np.random.default_rng(0)

    # --- 1. dimensional collapse, closed form checked against the rule itself ------------------
    dims = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    sweep = []
    for d in dims:
        M = args.delta * np.sqrt(d)
        row = {"d": d, "M": float(M), "acceptance": float(acceptance_closed_form(M))}
        if d <= 256:                                  # MC is only informative while acceptance > 0
            row["acceptance_mc"] = acceptance_monte_carlo(M, d, args.n_mc, rng)
        sweep.append(row)
    mc_err = [abs(r["acceptance_mc"] - r["acceptance"]) for r in sweep if "acceptance_mc" in r]

    # --- 2. random-subspace law -----------------------------------------------------------------
    subspace = []
    for d in (16, 64, 128, 512, 4096):
        for k in (1, 2, 4, 8, 16, 32):
            if k >= d:
                continue
            M = args.delta * np.sqrt(d)
            r = random_subspace_acceptance(M, d, k, 200_000, rng)
            subspace.append({"d": d, "k": k, **r})

    # How much does exact acceptance move with d at fixed k? This is the "function of k alone" claim.
    by_k = {}
    for k in (1, 2, 4, 8, 16, 32):
        vals = [r["exact_beta"] for r in subspace if r["k"] == k]
        if len(vals) > 1:
            by_k[k] = {"mean": float(np.mean(vals)),
                       "cv_pct": float(100.0 * np.std(vals) / np.mean(vals)),
                       "n_dims": len(vals)}
    plugin_err = [r["plugin_rel_error_pct"] for r in subspace]

    # --- 3. feasibility map ---------------------------------------------------------------------
    feas = {}
    for target in (0.8, 0.5, 0.2, 0.05):
        M_star = 2.0 * norm.ppf((2.0 - target) / 2.0)
        feas[str(target)] = {"M_star": float(M_star),
                             "delta_over_sigma_at": {str(d): float(M_star / np.sqrt(d))
                                                     for d in (16, 64, 256, 1024, 4096)}}

    res = {
        "commit": git_commit(),
        "delta_per_coordinate_over_sigma": args.delta,
        "n_monte_carlo": args.n_mc,
        "dimension_sweep": sweep,
        "monte_carlo_check": {"max_abs_error": float(max(mc_err)), "n_checked": len(mc_err)},
        "random_subspace": subspace,
        "acceptance_by_k_across_dims": by_k,
        "plugin_approximation_error_pct": {
            "min": float(min(plugin_err)), "median": float(np.median(plugin_err)),
            "max": float(max(plugin_err)),
        },
        "feasibility": feas,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=2))

    print(f"per-coordinate gap delta/sigma = {args.delta}\n")
    print("  d       M    acceptance   (Monte Carlo)")
    for r in sweep:
        mc = f"{r['acceptance_mc']:.5f}" if "acceptance_mc" in r else "     -"
        print(f"{r['d']:5d} {r['M']:7.2f}  {r['acceptance']:10.6f}   {mc}")
    print(f"\nclosed form vs direct simulation: max |error| = {max(mc_err):.5f} "
          f"over {len(mc_err)} dimensions")

    print("\nrandom subspace: is acceptance a function of k alone?")
    print("  k    mean acceptance   CV across d in {16..4096}")
    for k, v in by_k.items():
        print(f"{k:3d}      {v['mean']:.4f}            {v['cv_pct']:.2f}%  ({v['n_dims']} dims)")
    pe = res["plugin_approximation_error_pct"]
    print(f"\nthe old plug-in step E[f(Z)] ~ f(E[Z]) errs by "
          f"{pe['min']:.1f}-{pe['max']:.1f}% (median {pe['median']:.1f}%)")

    print("\nfeasibility: per-coordinate delta/sigma needed to hold a given acceptance")
    print("  target      M*      d=16     d=64    d=256   d=1024   d=4096")
    for t, v in feas.items():
        row = v["delta_over_sigma_at"]
        print(f"   {float(t):.2f}   {v['M_star']:6.3f}  " +
              "  ".join(f"{row[str(d)]:7.4f}" for d in (16, 64, 256, 1024, 4096)))
    print(f"\nwrote {out / 'results.json'}")


if __name__ == "__main__":
    main()
