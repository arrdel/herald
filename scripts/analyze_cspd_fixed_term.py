#!/usr/bin/env python
"""E0.3a -- bound CSpD's ``fixed_term`` from its diffusion schedule alone. No GPU, no model.

``DistributionProcessor.judge_accept`` multiplies the density ratio by
``exp((x2_log_var - x1_log_var).sum(-1))``, where each ``log_var`` is a sum of ``log σ`` over every
intermediate denoising step, giving ``∏_t (σ_p,t / σ_q,t)``.

A factor of this shape **belongs**: CSpD shares one noise draw between the models, so it verifies a
trajectory, and each Gaussian transition contributes ``(|Σ_q,t| / |Σ_p,t|)^{1/2}``. The defect is
that CSpD applies the **reciprocal** of that, so the acceptance ratio is wrong by ``fixed_term²``
and the error exponent is **twice** the magnitudes reported here (see ``docs/RELATED_WORK.md`` and
the "E0.3a (corrected)" log entry).

How much damage it does depends on how far apart the draft's and target's predicted ``log σ`` can
be at each step, and that range is fixed by the schedule, not by the weights: under
``ModelVarType.LEARNED_RANGE`` the model emits an interpolation fraction in ``[0, 1]`` between
``posterior_log_variance_clipped`` and ``log β``. So the worst case is computable exactly, and a
realistic case is computable given an assumption about how much the two models' fractions differ --
which is stated explicitly rather than hidden.

Run:  python scripts/analyze_cspd_fixed_term.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from herald.models.baselines import baseline_imports
from herald.verifiers import fixed_term_magnitude


def load_schedule(num_sampling_steps: str = "100", noise_schedule: str = "cosine"):
    """Pull the respaced diffusion schedule out of CSpD, exactly as ``DiffLoss`` builds it."""
    with baseline_imports("CSpD"):
        from diffusion import create_diffusion

        gd = create_diffusion(timestep_respacing=num_sampling_steps, noise_schedule=noise_schedule)
        return {
            "num_timesteps": int(gd.num_timesteps),
            "betas": np.asarray(gd.betas, dtype=np.float64),
            "posterior_log_variance_clipped": np.asarray(
                gd.posterior_log_variance_clipped, dtype=np.float64
            ),
        }


def analyze(schedule: dict, n_channels: int = 16, frac_gap: float = 0.05) -> dict:
    """Thin wrapper over :func:`herald.verifiers.fixed_term_magnitude`, adding schedule metadata."""
    result = fixed_term_magnitude(
        schedule["posterior_log_variance_clipped"],
        schedule["betas"],
        n_channels=n_channels,
        frac_gap=frac_gap,
    )
    result["num_timesteps"] = schedule["num_timesteps"]
    result["plausible_fixed_term"] = float(np.exp(min(result["plausible_exponent"], 700.0)))
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", default="100")
    p.add_argument("--noise-schedule", default="cosine")
    p.add_argument("--channels", type=int, default=16)
    p.add_argument("--frac-gap", type=float, default=0.05)
    p.add_argument("--out", default="experiments/E0.3a-fixed-term/results.json")
    args = p.parse_args()

    schedule = load_schedule(args.steps, args.noise_schedule)
    result = analyze(schedule, args.channels, args.frac_gap)

    s = result["log_sigma_spread_per_step"]
    print(f"CSpD diffusion schedule: {result['num_timesteps']} respaced steps, "
          f"{result['accumulated_steps']} of them accumulated into log_var")
    print(f"per-step |log sigma| range: min {s['min']:.4f}  median {s['median']:.4f}  "
          f"max {s['max']:.4f}  sum {s['sum']:.2f}")
    print(f"terms in the product: {result['channels_x_steps']} "
          f"({result['n_channels']} channels x {result['accumulated_steps']} steps)")
    print()
    print(f"worst-case exponent          : {result['worst_case_exponent']:.1f}")
    print(f"  -> fixed_term up to         e^{result['worst_case_exponent']:.0f}")
    print(f"plausible exponent (gap={args.frac_gap}) : {result['plausible_exponent']:.1f}")
    print(f"  -> fixed_term ~             {result['plausible_fixed_term']:.3e}")
    print()
    print(f"for reference, doubling acceptance takes only an exponent of {math.log(2):.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"schedule_source": "baselines/CSpD", **result}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
