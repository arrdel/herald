#!/usr/bin/env python
"""Confidence intervals for the frontier-collapse statistic of E1.2b.

``analyze_e12b.py`` reports a point estimate: the largest distance, in units of the cost's dynamic
range, between any structured rule and the random-projection curve evaluated at the *same*
acceptance.  At ``d=128`` that number is 4.0%, and the paper leans on it.  A point estimate alone
does not say whether 4.0% is distinguishable from 0% or from 15%, so this script attaches an
interval to it.

The bootstrap is parametric, and deliberately so.  ``results.json`` stores per-cell means with
standard errors rather than the raw per-decision values, so there is nothing to resample
nonparametrically.  What we can do is propagate the reported uncertainty: each replicate perturbs

  * every cost by its own reported standard error, ``excess_nll ~ N(mu, sem)``, and
  * every acceptance by its binomial standard error, ``sqrt(a(1-a)/n_decisions)``,

then recomputes the whole statistic --- re-interpolating the random curve at the perturbed
acceptances, so that uncertainty in the *baseline* is propagated too, not just in the points scored
against it.  This is the right notion of "would the collapse survive a rerun", conditional on the
reported standard errors being honest.

Two statistics are reported per dimension.  The max-residual is what the paper quotes; being a
maximum it is upward-biased under noise, so its interval should be read as conservative.  The median
absolute residual is the stable complement and is what to cite if a reviewer objects to bootstrapping
an extremum.

Offline, no GPU, no model loading.

Run:  python scripts/bootstrap_frontier.py --results experiments/E1.2b-gpu/results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METRICS = (("excess_nll", "excess NLL under the target (nats/token)"),
           ("signal_energy", "signal-domain energy distance"))


def _acceptance_sem(row: dict) -> float:
    """Binomial standard error of the measured acceptance rate."""
    n = row.get("n_decisions", 0)
    a = row["acceptance"]
    if not n:
        return 0.0
    return float(np.sqrt(max(a * (1.0 - a), 0.0) / n))


def statistic(acc: np.ndarray, cost: np.ndarray, rule: np.ndarray) -> tuple[float, float]:
    """The collapse statistic: max and median |residual| as a percentage of the cost's range.

    The residual of a structured point is its cost minus the random curve's cost at matched
    acceptance.  Points outside the random curve's acceptance range are skipped rather than
    extrapolated, exactly as in ``analyze_e12b.baseline_at``.
    """
    is_rand = rule == "random"
    order = np.argsort(acc[is_rand])
    xs, ys = acc[is_rand][order], cost[is_rand][order]
    if xs.size < 2:
        return np.nan, np.nan

    scored = (~is_rand) & (rule != "exact") & (acc >= xs[0]) & (acc <= xs[-1])
    if not scored.any():
        return np.nan, np.nan
    resid = np.abs(cost[scored] - np.interp(acc[scored], xs, ys))

    span_pool = cost[rule != "exact"]
    span = span_pool.max() - span_pool.min()
    if span <= 0:
        return np.nan, np.nan
    return 100.0 * resid.max() / span, 100.0 * float(np.median(resid)) / span


def bootstrap_dim(rows: list[dict], key: str, n_boot: int, rng: np.random.Generator) -> dict | None:
    """Point estimate and percentile intervals for one dimension and one cost metric."""
    usable = [r for r in rows if key in r and f"{key}_sem" in r]
    if not usable:
        return None
    acc = np.array([r["acceptance"] for r in usable])
    cost = np.array([r[key] for r in usable])
    rule = np.array([r["rule"] for r in usable])
    cost_sem = np.array([r[f"{key}_sem"] for r in usable])
    acc_sem = np.array([_acceptance_sem(r) for r in usable])

    point_max, point_med = statistic(acc, cost, rule)

    draws = np.empty((n_boot, 2))
    for b in range(n_boot):
        draws[b] = statistic(
            np.clip(acc + acc_sem * rng.standard_normal(acc.size), 0.0, 1.0),
            cost + cost_sem * rng.standard_normal(cost.size),
            rule,
        )
    ok = np.isfinite(draws[:, 0])
    lo_max, hi_max = np.percentile(draws[ok, 0], [2.5, 97.5])
    lo_med, hi_med = np.percentile(draws[ok, 1], [2.5, 97.5])
    return {"max": point_max, "max_ci": (lo_max, hi_max),
            "median": point_med, "median_ci": (lo_med, hi_med),
            "n_valid": int(ok.sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="experiments/E1.2b-gpu/results.json")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260910)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    results = json.loads(Path(args.results).read_text())["results"]

    for key, label in METRICS:
        print(f"\n{'=' * 78}\n{label}\n")
        print(f"{'d':>5}  {'max resid (% of range)':>28}  {'median resid (% of range)':>28}")
        for r in results:
            out = bootstrap_dim(r["rows"], key, args.n_boot, rng)
            if out is None:
                continue
            lo, hi = out["max_ci"]
            mlo, mhi = out["median_ci"]
            print(f"{r['d']:5d}  {out['max']:8.1f}  [{lo:5.1f}, {hi:5.1f}]        "
                  f"{out['median']:8.1f}  [{mlo:5.1f}, {mhi:5.1f}]")

    print(f"\n{args.n_boot} parametric replicates; intervals are 2.5/97.5 percentiles.")
    print("Uncertainty propagated from the reported per-cell SEMs and binomial acceptance error,")
    print("including the random baseline the structured rules are scored against.")


if __name__ == "__main__":
    main()
