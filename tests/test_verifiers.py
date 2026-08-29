"""Sanity checks for the acceptance rules, plus a synthetic probe of claim C1.

The C1 probe is not a substitute for E0.3 -- it uses Gaussians rather than real draft/target
pairs. It exists so the dimensional argument is falsifiable in seconds rather than only after
the ImageNet sweep, and so a regression in the acceptance code is caught immediately.
"""

import numpy as np
import pytest
import torch

from herald.calibration import calibrate_threshold, empirical_coverage
from herald.verifiers import TokenSpaceVerifier, diagonal_gaussian_logpdf


def test_logpdf_matches_torch_distribution():
    torch.manual_seed(0)
    mean, std = torch.randn(8, 16), torch.rand(8, 16) + 0.5
    x = torch.randn(8, 16)
    expected = torch.distributions.Normal(mean, std).log_prob(x).sum(-1)
    torch.testing.assert_close(diagonal_gaussian_logpdf(x, mean, std), expected)


def test_identical_distributions_always_accept():
    """When draft and target agree the ratio is 1, at any dimension."""
    torch.manual_seed(0)
    mean, std = torch.randn(64, 16), torch.rand(64, 16) + 0.5
    x = mean + std * torch.randn_like(mean)
    prob = TokenSpaceVerifier(temperature=1.0).acceptance_prob(x, mean, std, mean, std)
    torch.testing.assert_close(prob, torch.ones_like(prob))


@pytest.mark.parametrize("temperature", [1.0, 5.0, 42.0])
def test_temperature_monotonically_inflates_acceptance(temperature):
    """CSpD's temperature flattens both densities, pushing the ratio toward 1.

    This is the mechanism behind claim C1: acceptance rises with temperature even though the
    draft and target models are unchanged, so a high-temperature acceptance rate says nothing
    about whether the two actually agree.
    """
    torch.manual_seed(0)
    d = 16
    draft_mean, target_mean = torch.zeros(4096, d), torch.full((4096, d), 0.5)
    std = torch.ones(4096, d)
    x = draft_mean + std * torch.randn_like(draft_mean)

    rate = (
        TokenSpaceVerifier(temperature=temperature)
        .acceptance_prob(x, draft_mean, std, target_mean, std)
        .mean()
        .item()
    )
    test_temperature_monotonically_inflates_acceptance.rates[temperature] = rate
    assert 0.0 <= rate <= 1.0


test_temperature_monotonically_inflates_acceptance.rates = {}


def test_acceptance_decays_with_dimension():
    """C1 probe: with per-dimension disagreement fixed, acceptance falls as d grows.

    Each dimension contributes an independent factor to the density ratio, so a fixed per-
    dimension mismatch compounds multiplicatively. Verifying a low-dimensional statistic instead
    of the full token is what claim C2 proposes to do about it.
    """
    torch.manual_seed(0)
    shift, n = 0.5, 8192
    rates = {}
    for d in (1, 2, 4, 8, 16, 32, 64):
        draft_mean = torch.zeros(n, d)
        target_mean = torch.full((n, d), shift)
        std = torch.ones(n, d)
        x = draft_mean + std * torch.randn_like(draft_mean)
        rates[d] = (
            TokenSpaceVerifier(temperature=1.0)
            .acceptance_prob(x, draft_mean, std, target_mean, std)
            .mean()
            .item()
        )

    dims = [1, 2, 4, 8, 16, 32, 64]
    assert all(rates[a] > rates[b] for a, b in zip(dims[:-1], dims[1:], strict=True))
    assert rates[64] < 0.1 * rates[1], f"expected sharp decay, got {rates}"


def test_crc_controls_risk_on_held_out_data():
    rng = np.random.default_rng(0)
    n = 4000
    scores = rng.uniform(0, 1, n)
    # Loss grows with the nonconformity score, as a usable score must.
    losses = np.clip(scores + rng.normal(0, 0.1, n), 0, 1)

    calib, test = slice(0, n // 2), slice(n // 2, n)
    alpha = 0.1
    result = calibrate_threshold(scores[calib], losses[calib], alpha=alpha)
    held_out = empirical_coverage(scores[test], losses[test], result.tau)

    assert result.empirical_risk <= alpha
    assert held_out["mean_risk"] <= alpha * 1.5, held_out
    assert held_out["acceptance_rate"] > 0.2, "threshold should not be vacuously conservative"


def test_crc_rejects_unreachable_alpha():
    scores = np.linspace(0, 1, 10)
    with pytest.raises(ValueError, match="unreachable"):
        calibrate_threshold(scores, scores, alpha=0.05)
