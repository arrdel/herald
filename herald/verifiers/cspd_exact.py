"""CSpD's shipped acceptance rule, reimplemented faithfully and decomposed.

Verified against ``baselines/CSpD/spec_decode/spec_decode_runner.py`` and the diffusion code that
feeds it. CSpD's ``DistributionProcessor.judge_accept`` differs from the exact speculative
acceptance ratio ``p(x)/q(x)`` in **two independent ways**, and separating them is the point of
this module: only one of the two is the documented temperature.

**Deviation 1 -- the temperature.** ``x1_pdf_unormalize`` and ``x2_pdf_unormalize`` both divide the
standardized residual by ``self.temperature``, defaulting to 42. Dividing the exponent by 42²
flattens both densities toward uniform and drives the ratio to 1 regardless of whether the models
agree. Traced call path: ``SpecDecodeRunner.run`` builds ``DistributionProcessor(...,
temperature=self.temperature)`` and calls ``speculative_sample`` → ``judge_accept`` → both
``*_pdf_unormalize``. The temperature is applied *at the acceptance test*, not merely inside a
proposal step.

**Deviation 2 -- the accumulated variance term.** ``judge_accept`` multiplies the ratio by
``fixed_term = exp((x2_log_var - x1_log_var).sum(-1))``. Despite the name, ``x*_log_var`` is not the
log-variance of the final-step Gaussian. ``gaussian_diffusion.py`` accumulates
``manager.log_var += 0.5 * out["log_variance"]`` over *every denoising step except the last*, so it
is a sum of ``log σ`` across ~99 timesteps and all 16 channels. Nothing in the density ratio of the
two final-step Gaussians calls for that factor. It is inert only if draft and target predict
identical variances at every intermediate step; they do not, because ``create_diffusion`` is built
with ``learn_sigma=True`` (``ModelVarType.LEARNED_RANGE``), so the per-token variance is a model
output and differs between the two networks.

Everything here is computed in log space. CSpD evaluates the ratio as a product of raw
``torch.exp`` calls, which underflows to zero at ``d=16`` for even moderately separated Gaussians --
so the shipped rule cannot be evaluated at temperature 1 in its own arithmetic without collapsing
to a hard zero. That is worth knowing before concluding anything from a low measured acceptance.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .token_space import diagonal_gaussian_logpdf


@dataclass
class TokenGaussian:
    """The per-token distribution CSpD hands to its acceptance test.

    Attributes:
        mean: ``(N, d)`` final-step posterior mean (``manager.mean``).
        std: ``(N, d)`` final-step standard deviation (``manager.variance`` -- the name is
            misleading upstream; the value is ``exp(0.5 * log_variance) * sampling_temperature``).
        log_sigma_accum: ``(N, d)`` sum of ``log σ`` over all *intermediate* denoising steps
            (``manager.log_var``). Feeds CSpD's ``fixed_term`` and appears nowhere in the correct
            density ratio.
    """

    mean: torch.Tensor
    std: torch.Tensor
    log_sigma_accum: torch.Tensor | None = None

    def __post_init__(self):
        if (self.std <= 0).any():
            raise ValueError("std must be positive")


def exact_log_ratio(x: torch.Tensor, draft: TokenGaussian, target: TokenGaussian) -> torch.Tensor:
    """``log p(x) - log q(x)`` for the two final-step Gaussians. The genuinely lossless rule."""
    log_p = diagonal_gaussian_logpdf(x, target.mean, target.std, temperature=1.0)
    log_q = diagonal_gaussian_logpdf(x, draft.mean, draft.std, temperature=1.0)
    return log_p - log_q


def cspd_log_ratio(
    x: torch.Tensor,
    draft: TokenGaussian,
    target: TokenGaussian,
    temperature: float = 42.0,
    include_fixed_term: bool = True,
) -> torch.Tensor:
    """CSpD's ``judge_accept``, in log space.

    Setting ``temperature=1.0`` and ``include_fixed_term=False`` recovers :func:`exact_log_ratio`
    exactly, which is the identity the tests pin. The two flags therefore isolate one deviation
    each, and toggling them decomposes the acceptance gap.
    """
    quad_p = (((x - target.mean) / (target.std * temperature)) ** 2).sum(-1)
    quad_q = (((x - draft.mean) / (draft.std * temperature)) ** 2).sum(-1)
    # The normalizers CSpD keeps: prod(s1)/prod(s2), i.e. the untempered scale terms.
    log_scale = torch.log(draft.std).sum(-1) - torch.log(target.std).sum(-1)
    log_ratio = -0.5 * quad_p + 0.5 * quad_q + log_scale

    if include_fixed_term:
        if draft.log_sigma_accum is None or target.log_sigma_accum is None:
            raise ValueError("include_fixed_term=True requires log_sigma_accum on both distributions")
        log_ratio = log_ratio + (target.log_sigma_accum - draft.log_sigma_accum).sum(-1)
    return log_ratio


def acceptance_from_log_ratio(log_ratio: torch.Tensor) -> torch.Tensor:
    """``min(1, exp(log_ratio))`` -- the per-token acceptance probability."""
    return torch.clamp(log_ratio.exp(), max=1.0)


def decompose_acceptance(
    x: torch.Tensor,
    draft: TokenGaussian,
    target: TokenGaussian,
    temperature: float = 42.0,
) -> dict[str, float]:
    """Attribute CSpD's acceptance rate to its two deviations from the exact rule.

    Returns mean acceptance probability under four rules:

    ``exact``
        The lossless ratio. What speculative decoding's guarantee is stated about.
    ``fixed_term_only``
        Temperature 1, accumulated-variance factor left in. Isolates deviation 2.
    ``temperature_only``
        Temperature as shipped, factor removed. Isolates deviation 1.
    ``as_shipped``
        Both. What CSpD actually runs.

    The gap between ``exact`` and ``as_shipped`` is the whole of claim C1; the split between the two
    middle rows says which defect is doing the work, which nobody has reported.
    """
    has_accum = draft.log_sigma_accum is not None and target.log_sigma_accum is not None
    rules = {
        "exact": (1.0, False),
        "temperature_only": (temperature, False),
    }
    if has_accum:
        rules["fixed_term_only"] = (1.0, True)
        rules["as_shipped"] = (temperature, True)

    out = {}
    for name, (temp, fixed) in rules.items():
        lr = cspd_log_ratio(x, draft, target, temperature=temp, include_fixed_term=fixed)
        out[name] = acceptance_from_log_ratio(lr).mean().item()
    return out


def analytic_acceptance_shared_covariance(normalized_gap: float, dim: int) -> float:
    """Exact expected acceptance for two Gaussians sharing a covariance ``σ²I``.

    ``E_q[min(1, p/q)] = 1 - TV(p, q) = 2 - 2Φ(M/2)`` with ``M = ‖m_p - m_q‖ / σ`` the Mahalanobis
    distance. This is Eq. 45 of *Accelerating Speculative Diffusions via Block Verification*
    (arXiv 2606.13426), which derives it for speculative diffusion and analyses the ``σ → 0`` limit.

    Written in terms of dimension because that is the limit claim C1 is about. For a per-coordinate
    gap ``δ = ‖m_p - m_q‖ / (σ√d)``, ``M = δ√d``, so acceptance falls off in ``√d`` at fixed
    per-coordinate disagreement -- the two models need to agree *more precisely* per coordinate as
    the token dimension grows, merely to hold acceptance constant.

    Args:
        normalized_gap: per-coordinate mean difference in units of σ.
        dim: token dimension (16 for KL16).

    This assumes a **shared** covariance. MAR's per-token variances are learned and differ between
    draft and target, so treat this as a reference curve for E0.3, not a prediction.
    """
    import math

    m = normalized_gap * math.sqrt(dim)
    phi = 0.5 * (1.0 + math.erf((m / 2.0) / math.sqrt(2.0)))
    return 2.0 - 2.0 * phi


def fixed_term_magnitude(
    posterior_log_variance_clipped,
    betas,
    n_channels: int = 16,
    frac_gap: float = 0.05,
) -> dict:
    """How large CSpD's accumulated-variance factor can get, from the schedule alone.

    Under ``ModelVarType.LEARNED_RANGE`` the network emits an interpolation fraction in ``[0, 1]``
    and the step's log-variance is ``frac * log β + (1 - frac) * posterior_log_variance_clipped``.
    The per-step ``log σ`` therefore lives in a range fixed by the schedule, so the worst case is
    exactly computable without touching a model.

    Args:
        posterior_log_variance_clipped, betas: the respaced schedule arrays.
        n_channels: token dimension (16 for KL16).
        frac_gap: **assumed** typical gap between the draft's and target's fractions. This is the
            one input that is not derived from the schedule, and E0.3 replaces it with measurement.

    The first entry of each array is dropped: ``p_sample_loop_progressive_with_noise`` accumulates
    ``log_var`` only when ``i != 0``, so the final step never enters the product.
    """
    import numpy as np

    min_log = np.asarray(posterior_log_variance_clipped, dtype=np.float64)
    max_log = np.log(np.asarray(betas, dtype=np.float64))
    spread = np.abs(0.5 * (max_log - min_log)[1:])

    worst = float(n_channels * spread.sum())
    plausible = float(n_channels * frac_gap * spread.sum())
    return {
        "accumulated_steps": int(len(spread)),
        "n_channels": int(n_channels),
        "channels_x_steps": int(n_channels * len(spread)),
        "log_sigma_spread_per_step": {
            "min": float(spread.min()),
            "median": float(np.median(spread)),
            "max": float(spread.max()),
            "sum": float(spread.sum()),
        },
        "worst_case_exponent": worst,
        "plausible_exponent": plausible,
        "frac_gap_assumed": frac_gap,
        "exponent_to_double_acceptance": float(np.log(2)),
    }


def underflows_in_linear_space(log_ratio: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    """Which tokens CSpD's own linear-space arithmetic would evaluate as an exact zero.

    CSpD computes ``exp(-0.5 * quad)`` directly. At ``d=16`` and temperature 1 the quadratic form is
    routinely large enough that both numerator and denominator underflow, giving ``0/0``. Reporting
    a measured acceptance of zero without checking this would confuse a numerical artifact with the
    dimensional effect that claim C1 is about.
    """
    return log_ratio < torch.log(torch.tensor(torch.finfo(dtype).tiny))
