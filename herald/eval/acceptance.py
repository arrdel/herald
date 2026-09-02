"""Exact speculative acceptance on the E1.x testbed, and the diagnostics that explain it.

Shared by E1.1 (the token-dimension sweep) and E1.1b (its convergence control), and the place
E1.2a/E1.2b/E1.3 will hook into when they change the space the acceptance test lives in. It lives in
the package rather than in a script because every one of those experiments compares against E1.1's
numbers, and a second copy of this function is the most likely way that comparison quietly stops
being valid.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from herald.models.ts_mar import TimeSeriesMAR, random_masking

#: The testbed's draft/target pair. Target is the expensive, accurate model; draft is the cheap one.
#: Held fixed across every sweep so the capacity gap -- the thing that generates disagreement in the
#: first place -- does not drift with whatever else is being varied.
TARGET_ARCH = {"width": 256, "depth": 6, "n_heads": 8, "n_mixture": 8}
DRAFT_ARCH = {"width": 128, "depth": 4, "n_heads": 4, "n_mixture": 4}


@torch.no_grad()
def measure_acceptance(
    draft: TimeSeriesMAR,
    target: TimeSeriesMAR,
    dataset,
    device: str,
    batch_size: int = 128,
    max_batches: int = 40,
    seed: int = 0,
) -> dict:
    """Exact acceptance, plus the per-coordinate disagreement that should explain it.

    Both models see the same tokens and the same mask, which is what the speculative setting
    requires: any difference in their predictive distributions is then model disagreement rather
    than a difference in what each was told.

    Returns ``delta_median`` (the per-coordinate mean gap, in units of the draft's own scale) and
    ``log_sigma_gap_median`` (scale disagreement) alongside the acceptance rate. Acceptance alone
    cannot distinguish a dimensional effect from the draft simply being worse, and the
    equal-covariance closed form is blind to scale disagreement -- so reporting the rate without
    these two is reporting a number that cannot be interpreted.
    """
    draft.eval()
    target.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    probs, gaps, sigma_gaps, log_ratios = [], [], [], []
    # ``MixtureSameFamily.sample`` takes no generator, so seeding only the mask would leave the draw
    # of x -- and therefore the acceptance estimate itself -- at the mercy of whatever global RNG
    # state the caller happened to be in. Forking makes the whole measurement reproducible without
    # perturbing the training stream that surrounds it.
    devices = [torch.device(device)] if torch.device(device).type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        gen = torch.Generator(device=device).manual_seed(seed)

        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            tokens = batch.to(device)
            mask = random_masking(
                tokens.shape[0], tokens.shape[1], torch.device(device), generator=gen
            )
            if not mask.any():
                continue

            q = draft.token_distribution(draft.forward_predictor(tokens, mask))
            p = target.token_distribution(target.forward_predictor(tokens, mask))

            x = q.sample()
            log_ratio = (p.log_prob(x) - q.log_prob(x))[mask]
            probs.append(torch.clamp(log_ratio.exp(), max=1.0).cpu())
            log_ratios.append(log_ratio.cpu())

            d = tokens.shape[-1]
            mean_gap = (p.mean - q.mean)[mask]
            var_q = q.variance[mask].clamp_min(1e-12)
            var_p = p.variance[mask].clamp_min(1e-12)
            sigma = var_q.mean(-1).sqrt().clamp_min(1e-8)
            gaps.append((mean_gap.norm(dim=-1) / (sigma * np.sqrt(d))).cpu())
            sigma_gaps.append((0.5 * (var_p.log() - var_q.log())).abs().mean(-1).cpu())

    if not probs:
        raise RuntimeError("no acceptance decisions collected; dataset may be smaller than one batch")

    probs = torch.cat(probs)
    gaps = torch.cat(gaps)
    sigma_gaps = torch.cat(sigma_gaps)
    log_ratios = torch.cat(log_ratios)
    return {
        "acceptance": float(probs.mean()),
        "acceptance_std": float(probs.std()),
        "n_decisions": int(probs.numel()),
        "delta_median": float(gaps.median()),
        "delta_mean": float(gaps.mean()),
        "log_sigma_gap_median": float(sigma_gaps.median()),
        "log_ratio_median": float(log_ratios.median()),
        "fraction_certain_accept": float((probs >= 0.999).float().mean()),
    }
