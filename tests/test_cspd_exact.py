"""Checks that our reimplementation of CSpD's acceptance rule is faithful and correctly decomposed.

The load-bearing test is ``test_cspd_rule_reduces_to_the_exact_ratio``: if setting temperature to 1
and dropping the accumulated-variance factor did *not* recover the textbook density ratio, then
either our reading of CSpD is wrong or our "exact" baseline is, and every E0.3 number would be
built on that error.
"""

import math

import pytest
import torch

from herald.verifiers import (
    TokenGaussian,
    acceptance_from_log_ratio,
    analytic_acceptance_shared_covariance,
    cspd_log_ratio,
    decompose_acceptance,
    exact_log_ratio,
    fixed_term_magnitude,
    underflows_in_linear_space,
)


def _pair(d=16, n=512, shift=0.0, std_draft=1.0, std_target=1.0, accum=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    draft = TokenGaussian(
        mean=torch.zeros(n, d),
        std=torch.full((n, d), std_draft),
        log_sigma_accum=None if accum is None else torch.full((n, d), accum[0]),
    )
    target = TokenGaussian(
        mean=torch.full((n, d), shift),
        std=torch.full((n, d), std_target),
        log_sigma_accum=None if accum is None else torch.full((n, d), accum[1]),
    )
    x = draft.mean + draft.std * torch.randn(n, d, generator=g)
    return x, draft, target


def test_cspd_rule_reduces_to_the_exact_ratio():
    """temperature=1 and no fixed_term must reproduce log p(x) - log q(x) exactly."""
    x, draft, target = _pair(shift=0.4, std_draft=0.9, std_target=1.3)
    torch.testing.assert_close(
        cspd_log_ratio(x, draft, target, temperature=1.0, include_fixed_term=False),
        exact_log_ratio(x, draft, target),
    )


def test_identical_distributions_accept_always_under_the_exact_rule():
    x, draft, target = _pair(shift=0.0, std_draft=1.0, std_target=1.0)
    prob = acceptance_from_log_ratio(exact_log_ratio(x, draft, target))
    torch.testing.assert_close(prob, torch.ones_like(prob))


@pytest.mark.parametrize("temperature", [1.0, 2.0, 5.0, 10.0, 42.0])
def test_temperature_monotonically_inflates_acceptance(temperature):
    """The models are fixed; only the knob moves. Any rise is the knob, not agreement."""
    x, draft, target = _pair(shift=0.5)
    rate = acceptance_from_log_ratio(
        cspd_log_ratio(x, draft, target, temperature=temperature, include_fixed_term=False)
    ).mean()
    baseline = acceptance_from_log_ratio(
        cspd_log_ratio(x, draft, target, temperature=1.0, include_fixed_term=False)
    ).mean()
    assert rate >= baseline - 1e-6
    if temperature > 1.0:
        assert rate > baseline


def test_fixed_term_shifts_acceptance_even_when_models_agree():
    """The sharpest statement of deviation 2: draft and target are *identical* distributions.

    The exact ratio is then 1 at every token and acceptance is certain. CSpD's rule still applies
    the accumulated-variance factor, so a difference in the two models' intermediate variance
    schedules alone moves the acceptance decision -- with nothing about the final-step
    distributions differing at all.
    """
    d, n = 16, 256
    x, draft, target = _pair(d=d, n=n, shift=0.0, accum=(0.0, -0.01))

    exact = acceptance_from_log_ratio(exact_log_ratio(x, draft, target)).mean().item()
    with_fixed = acceptance_from_log_ratio(
        cspd_log_ratio(x, draft, target, temperature=1.0, include_fixed_term=True)
    ).mean().item()

    assert exact == pytest.approx(1.0)
    # accum differs by -0.01 per channel over 16 channels => factor exp(-0.16) ~ 0.85
    assert with_fixed == pytest.approx(math.exp(-0.16), rel=1e-3)


def test_shipped_fixed_term_is_the_reciprocal_of_the_trajectory_factor():
    """Deviation 2, stated precisely: the factor belongs, but CSpD applies it upside down.

    Sharing one noise draw between the two models means the object being verified is the whole
    denoising trajectory, whose likelihood ratio picks up ``(|Sigma_q,t|/|Sigma_p,t|)^(1/2)`` per
    Gaussian transition -- i.e. ``prod_t sigma_q,t / sigma_p,t``. CSpD applies
    ``prod_t sigma_p,t / sigma_q,t``. The two log-ratios must therefore straddle the trajectory-free
    ratio symmetrically, which is what this pins.
    """
    x, draft, target = _pair(d=16, n=256, shift=0.3, accum=(0.0, -0.01))

    shipped = cspd_log_ratio(x, draft, target, temperature=1.0, include_fixed_term=True)
    corrected = cspd_log_ratio(x, draft, target, temperature=1.0, include_fixed_term=True,
                               fixed_term_sign=-1.0)
    without = cspd_log_ratio(x, draft, target, temperature=1.0, include_fixed_term=False)

    torch.testing.assert_close(shipped - without, without - corrected)
    # The acceptance ratio is wrong by fixed_term squared.
    torch.testing.assert_close(shipped - corrected, 2.0 * (shipped - without))


def test_the_two_orientations_agree_only_when_the_models_share_a_variance_schedule():
    """Why the sign error is live: it vanishes exactly when LEARNED_RANGE makes it impossible."""
    x, draft, target = _pair(d=16, n=128, shift=0.3, accum=(0.0, 0.0))
    shipped = cspd_log_ratio(x, draft, target, temperature=1.0, include_fixed_term=True)
    corrected = cspd_log_ratio(x, draft, target, temperature=1.0, include_fixed_term=True,
                               fixed_term_sign=-1.0)
    torch.testing.assert_close(shipped, corrected)


def test_fixed_term_requires_the_accumulator():
    x, draft, target = _pair()
    with pytest.raises(ValueError, match="log_sigma_accum"):
        cspd_log_ratio(x, draft, target, temperature=42.0, include_fixed_term=True)


def test_decomposition_reports_every_rule_and_orders_them_sensibly():
    x, draft, target = _pair(shift=0.5, accum=(0.0, 0.0))
    out = decompose_acceptance(x, draft, target, temperature=42.0)

    assert set(out) == {"exact", "temperature_only", "fixed_term_only", "as_shipped",
                        "corrected_trajectory"}
    # With matched accumulators the fixed term is inert under either orientation.
    assert out["fixed_term_only"] == pytest.approx(out["exact"], rel=1e-5)
    assert out["corrected_trajectory"] == pytest.approx(out["exact"], rel=1e-5)
    assert out["as_shipped"] == pytest.approx(out["temperature_only"], rel=1e-5)
    assert out["as_shipped"] > out["exact"]


def test_decomposition_separates_the_sign_error_from_the_trajectory_term():
    """With mismatched schedules the two orientations must straddle the last-step-only rule."""
    x, draft, target = _pair(shift=0.5, accum=(0.0, -0.02))
    out = decompose_acceptance(x, draft, target, temperature=42.0)
    assert out["fixed_term_only"] != pytest.approx(out["corrected_trajectory"], rel=1e-3)
    lo, hi = sorted([out["fixed_term_only"], out["corrected_trajectory"]])
    assert lo < out["exact"] < hi


def test_decomposition_degrades_gracefully_without_accumulators():
    x, draft, target = _pair(shift=0.5)
    out = decompose_acceptance(x, draft, target)
    assert set(out) == {"exact", "temperature_only"}


def test_analytic_acceptance_matches_monte_carlo():
    """Cross-validates two independent things at once.

    The closed form is Eq. 45 of arXiv 2606.13426, derived analytically. Our ``exact_log_ratio`` is
    written from the Gaussian density. If a Monte Carlo estimate of ``E_q[min(1, p/q)]`` agrees with
    the closed form, both are almost certainly right -- and E0.3's measured acceptance can then be
    plotted against a trustworthy reference curve.
    """
    g = torch.Generator().manual_seed(7)
    for d, gap in ((4, 0.5), (16, 0.25), (16, 0.5), (64, 0.1)):
        n = 200_000
        draft = TokenGaussian(torch.zeros(n, d), torch.ones(n, d))
        target = TokenGaussian(torch.full((n, d), gap), torch.ones(n, d))
        x = torch.randn(n, d, generator=g)
        mc = acceptance_from_log_ratio(exact_log_ratio(x, draft, target)).mean().item()
        assert mc == pytest.approx(analytic_acceptance_shared_covariance(gap, d), abs=5e-3)


def test_acceptance_falls_off_with_dimension_at_fixed_per_coordinate_gap():
    """The C1 mechanism, stated precisely: same per-coordinate agreement, more dimensions, worse."""
    rates = [analytic_acceptance_shared_covariance(0.25, d) for d in (1, 4, 16, 64, 256)]
    assert all(a > b for a, b in zip(rates[:-1], rates[1:], strict=True))
    assert rates[0] > 0.85
    assert rates[-1] < 0.05


def test_identical_means_accept_always():
    assert analytic_acceptance_shared_covariance(0.0, 16) == pytest.approx(1.0)


def test_fixed_term_magnitude_math_on_a_known_schedule():
    """Pin the bound arithmetic against a schedule whose answer is hand-computable."""
    import numpy as np

    # 4 steps; drop index 0, leaving 3 accumulated steps with a constant spread.
    # log beta - posterior_log_var = 2.0 everywhere => |0.5 * 2.0| = 1.0 per step.
    betas = np.exp(np.array([1.0, 1.0, 1.0, 1.0]))
    post_log_var = np.array([-1.0, -1.0, -1.0, -1.0])

    out = fixed_term_magnitude(post_log_var, betas, n_channels=8, frac_gap=0.5)
    assert out["accumulated_steps"] == 3
    assert out["channels_x_steps"] == 24
    assert out["log_sigma_spread_per_step"]["sum"] == pytest.approx(3.0)
    assert out["worst_case_exponent"] == pytest.approx(8 * 3.0)
    assert out["plausible_exponent"] == pytest.approx(8 * 0.5 * 3.0)


def test_a_small_exponent_already_saturates_acceptance():
    """Why the bound matters: the accept rule is min(1, ratio), so modest inflation pins it at 1."""
    x, draft, target = _pair(shift=0.5, accum=(0.0, 0.25))  # 16 channels x 0.25 => exponent 4.0
    exact = acceptance_from_log_ratio(exact_log_ratio(x, draft, target)).mean().item()
    inflated = acceptance_from_log_ratio(
        cspd_log_ratio(x, draft, target, temperature=1.0, include_fixed_term=True)
    ).mean().item()
    # Measured: exact ~0.32, inflated ~0.91. The factor nearly triples acceptance on its own,
    # with the temperature still at 1 and the two models' final-step Gaussians untouched.
    assert exact < 0.4
    assert inflated > 0.85
    assert inflated > 2.5 * exact


def test_underflow_detector_flags_what_cspds_own_arithmetic_would_zero():
    """CSpD evaluates exp(-0.5*quad) directly; at d=16 that can underflow to a hard zero."""
    x, draft, target = _pair(d=16, shift=6.0)
    lr = exact_log_ratio(x, draft, target)
    assert underflows_in_linear_space(lr).any(), "expected some tokens below float32 tiny"
    # The log-space rule still returns finite, usable values on exactly those tokens.
    assert torch.isfinite(lr).all()
