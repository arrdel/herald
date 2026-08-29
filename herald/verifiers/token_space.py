"""Token-space acceptance rules -- the CSpD-exact baseline, with temperature made explicit.

CSpD's shipped ``DistributionProcessor`` hardcodes ``temperature=42.`` into the Gaussian
exponent of both densities. Its losslessness proof covers the ratio at temperature 1. This
module reimplements the rule with temperature as a first-class argument so E0.3 can sweep it
and measure what the relaxation is worth, in log space to avoid the underflow that makes the
untempered ratio hard to evaluate directly at d=16.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def diagonal_gaussian_logpdf(
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Log density of a diagonal Gaussian, with CSpD's temperature applied to the exponent.

    ``temperature`` divides the standardized residual, exactly as CSpD does. At 1.0 this is the
    true log density. Above 1.0 it flattens the density toward uniform; the normalizer is scaled
    to match so the comparison stays a well-defined (if no longer exact) ratio.

    Args:
        x, mean, std: ``(..., d)``. ``std`` must be positive.
    Returns:
        ``(...)`` log densities.
    """
    d = x.shape[-1]
    scaled_std = std * temperature
    quad = (((x - mean) / scaled_std) ** 2).sum(-1)
    log_norm = torch.log(scaled_std).sum(-1) + 0.5 * d * torch.log(
        torch.tensor(2 * torch.pi, device=x.device, dtype=x.dtype)
    )
    return -0.5 * quad - log_norm


@dataclass
class TokenSpaceVerifier:
    """Classical speculative acceptance ``min(1, p(x)/q(x))`` over continuous tokens.

    At ``temperature=1.0`` this is the distribution-preserving rule. Higher values reproduce
    CSpD's relaxation. ``last_log_ratio`` retains the most recent log ratios so the sweep can
    report the full distribution rather than only the mean acceptance rate.
    """

    temperature: float = 1.0
    last_log_ratio: torch.Tensor | None = None

    def acceptance_prob(
        self,
        x: torch.Tensor,
        draft_mean: torch.Tensor,
        draft_std: torch.Tensor,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
    ) -> torch.Tensor:
        """Per-token acceptance probability for drafted tokens ``x``."""
        log_q = diagonal_gaussian_logpdf(x, draft_mean, draft_std, self.temperature)
        log_p = diagonal_gaussian_logpdf(x, target_mean, target_std, self.temperature)
        log_ratio = log_p - log_q
        self.last_log_ratio = log_ratio.detach()
        return torch.clamp(log_ratio.exp(), max=1.0)

    def accept_mask(self, *args, generator: torch.Generator | None = None, **kwargs):
        prob = self.acceptance_prob(*args, **kwargs)
        u = torch.rand(prob.shape, device=prob.device, dtype=prob.dtype, generator=generator)
        return u < prob, prob


def effective_dimension_penalty(
    draft_mean: torch.Tensor,
    draft_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> torch.Tensor:
    """KL(q || p) for diagonal Gaussians, summed over dimensions.

    The quantity behind claim C1: for near-factorized token distributions this KL grows roughly
    linearly in ``d``, and expected acceptance falls off accordingly. Reporting it alongside the
    measured acceptance rate separates "the models disagree" from "the space is too big", which
    the acceptance rate alone cannot distinguish.
    """
    var_q, var_p = draft_std**2, target_std**2
    return 0.5 * (
        (var_q / var_p).sum(-1)
        + ((target_mean - draft_mean) ** 2 / var_p).sum(-1)
        - draft_mean.shape[-1]
        + torch.log(var_p / var_q).sum(-1)
    )
