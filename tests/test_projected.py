"""Tests for projected verification (the E1.2/E1.3 machinery).

The anchor is the ``k == d`` invariance. A square orthogonal ``A`` is a bijection with unit Jacobian,
so the projected acceptance ratio must equal the raw one *exactly*, not approximately. That single
identity exercises the pushforward mean, the full ``A diag(σ²) Aᵀ`` covariance, the mixture weights
and the log-normalizer all at once -- and it is the property that makes any acceptance *gain* at
``k < d`` attributable to the projection discarding disagreement, rather than to an error in how the
projected density is computed.
"""

import pytest
import torch
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal

from herald.verifiers import (
    acceptance_from_log_ratio,
    project_mixture,
    projected_log_ratio,
    random_projection,
)


def _mixture(n=64, k=4, d=8, seed=0, shift=0.0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, k, generator=g)
    mean = torch.randn(n, k, d, generator=g) + shift
    std = torch.rand(n, k, d, generator=g) * 0.5 + 0.5
    return MixtureSameFamily(Categorical(logits=logits), Independent(Normal(mean, std), 1))


def test_square_orthogonal_projection_leaves_the_ratio_unchanged():
    d = 8
    q, p = _mixture(d=d, seed=0), _mixture(d=d, seed=1, shift=0.4)
    x = q.sample()
    a = random_projection(d, d, generator=torch.Generator().manual_seed(3))

    raw = p.log_prob(x) - q.log_prob(x)
    projected = projected_log_ratio(x, q, p, a)
    torch.testing.assert_close(projected, raw, atol=1e-4, rtol=1e-4)


def test_random_projection_rows_are_orthonormal():
    a = random_projection(4, 16, generator=torch.Generator().manual_seed(0))
    gram = a @ a.T
    torch.testing.assert_close(gram, torch.eye(4), atol=1e-5, rtol=1e-5)


def test_projection_rejects_impossible_shapes():
    with pytest.raises(ValueError):
        random_projection(17, 16)
    with pytest.raises(ValueError):
        random_projection(0, 16)


def test_projected_mixture_matches_monte_carlo_moments():
    """The pushforward is claimed to be exact, so its moments must match sampling the map."""
    d, k = 8, 3
    dist = _mixture(n=1, d=d, seed=5)
    a = random_projection(k, d, generator=torch.Generator().manual_seed(1))
    proj = project_mixture(dist, a)

    torch.manual_seed(0)
    samples = dist.sample((200_000,))[:, 0] @ a.T
    torch.testing.assert_close(samples.mean(0), proj.mean[0], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(samples.var(0), proj.variance[0], atol=5e-2, rtol=5e-2)


def test_projection_raises_on_dimension_mismatch():
    dist = _mixture(d=8)
    with pytest.raises(ValueError, match="expects tokens of dim"):
        project_mixture(dist, random_projection(2, 6))


@pytest.mark.parametrize("k", [1, 2, 4, 8, 16])
def test_lower_dimensional_projections_accept_more(k):
    """The null-control premise, and the reason a bare acceptance number proves nothing.

    Any projection buys acceptance by discarding the directions in which the models disagree, so a
    structured statistic has to be judged against this curve rather than against raw verification.
    """
    d, n = 16, 4096
    q, p = _mixture(n=n, d=d, seed=0), _mixture(n=n, d=d, seed=1, shift=0.3)
    x = q.sample()
    gen = torch.Generator().manual_seed(7)

    raw = acceptance_from_log_ratio(p.log_prob(x) - q.log_prob(x)).mean()
    proj = acceptance_from_log_ratio(
        projected_log_ratio(x, q, p, random_projection(k, d, generator=gen))
    ).mean()

    assert proj >= raw - 1e-4
    if k < d:
        assert proj > raw
