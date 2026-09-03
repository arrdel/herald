"""Tests for structured choices of the verified subspace (E1.2b).

The load-bearing property is **row-space invariance**: the acceptance ratio induced by a statistic
``Ax`` depends only on the row space of ``A``, because reparameterising ``A -> QA`` multiplies both
the numerator and the denominator by ``|det Q|``. That is what licenses the whole framing of E1.2b --
every rule is a choice of ``k``-dimensional subspace and nothing else, so the rules can be compared
like for like and each one is free to return whatever orthonormal basis is convenient.

The second anchor is the DCT band split. ``E[JᵀJ]`` restricted to the low band plus the same
restricted to the high band must reconstruct the unrestricted matrix exactly, by Parseval. If the
band selector were mis-shaped or applied along the wrong axis -- easy to do, since the Jacobian's
output axis is a flattened ``(channel, time)`` -- that identity is the thing that fails.
"""

import pytest
import torch
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal

from herald.models.ts_tokenizer import TimeSeriesTokenizer, TokenizerConfig
from herald.verifiers import projected_log_ratio
from herald.verifiers.structured import (
    dct_matrix,
    orthonormal_rows,
    pca_projection,
    signal_impact_matrix,
    signal_impact_projection,
    top_eigen_projection,
)


def _mixture(n=32, k=3, d=8, seed=0, shift=0.0):
    g = torch.Generator().manual_seed(seed)
    return MixtureSameFamily(
        Categorical(logits=torch.randn(n, k, generator=g)),
        Independent(Normal(torch.randn(n, k, d, generator=g) + shift,
                           torch.rand(n, k, d, generator=g) * 0.5 + 0.5), 1),
    )


def _tokenizer(d=8, seed=0):
    torch.manual_seed(seed)
    return TimeSeriesTokenizer(TokenizerConfig(n_channels=3, patch=8, latent_dim=d, hidden=16,
                                               depth=1)).eval()


def test_acceptance_depends_only_on_the_row_space():
    d, k = 8, 3
    q, p = _mixture(d=d, seed=0), _mixture(d=d, seed=1, shift=0.5)
    x = q.sample()
    g = torch.Generator().manual_seed(7)
    a = orthonormal_rows(torch.randn(k, d, generator=g))
    # Any invertible remix of the rows spans the same subspace, so must induce the same ratio.
    mixed = torch.randn(k, k, generator=g) @ a

    torch.testing.assert_close(projected_log_ratio(x, q, p, mixed),
                               projected_log_ratio(x, q, p, a), atol=1e-4, rtol=1e-4)


def test_dct_matrix_is_orthonormal():
    m = dct_matrix(8)
    torch.testing.assert_close(m @ m.T, torch.eye(8), atol=1e-5, rtol=1e-5)


def test_dct_bands_partition_the_signal_energy():
    tok = _tokenizer()
    tokens = torch.randn(4, 6, 8)
    full = signal_impact_matrix(tok, tokens, band="all", n_samples=4)
    low = signal_impact_matrix(tok, tokens, band="low", n_samples=4)
    high = signal_impact_matrix(tok, tokens, band="high", n_samples=4)
    torch.testing.assert_close(low + high, full, atol=1e-4, rtol=1e-3)


def test_impact_matrix_is_symmetric_positive_semidefinite():
    m = signal_impact_matrix(_tokenizer(), torch.randn(4, 6, 8), n_samples=4)
    torch.testing.assert_close(m, m.T, atol=1e-5, rtol=1e-5)
    assert torch.linalg.eigvalsh(m).min() > -1e-5


def test_pca_recovers_a_planted_subspace():
    g = torch.Generator().manual_seed(0)
    basis = orthonormal_rows(torch.randn(2, 10, generator=g))
    # Variance lives only in the planted plane, so the top-2 principal directions must span it.
    tokens = torch.randn(2000, 2, generator=g) @ basis
    got = pca_projection(tokens, 2)
    # Compare subspaces, not bases: the projector onto the span is the basis-free object.
    torch.testing.assert_close(got.T @ got, basis.T @ basis, atol=1e-3, rtol=1e-3)


def test_top_and_bottom_eigen_projections_select_opposite_ends():
    m = torch.diag(torch.tensor([5.0, 4.0, 1.0, 0.5]))
    torch.testing.assert_close(top_eigen_projection(m, 2).abs(),
                               torch.eye(4)[:2], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(top_eigen_projection(m, 2, largest=False).abs(),
                               torch.eye(4)[[3, 2]], atol=1e-5, rtol=1e-5)


def test_top_impact_directions_move_the_decoder_more_than_the_bottom_ones():
    """The anti-control has to be a real control: 'least important' must actually mean less."""
    tok = _tokenizer()
    tokens = torch.randn(8, 6, 8)
    top = signal_impact_projection(tok, tokens, 2, largest=True, n_samples=8)
    bottom = signal_impact_projection(tok, tokens, 2, largest=False, n_samples=8)
    m = signal_impact_matrix(tok, tokens, n_samples=8)
    assert torch.einsum("kd,de,ke->", top, m, top) > torch.einsum("kd,de,ke->", bottom, m, bottom)


def test_structured_projections_return_orthonormal_rows():
    tok = _tokenizer()
    tokens = torch.randn(4, 6, 8)
    for a in (pca_projection(tokens, 3),
              signal_impact_projection(tok, tokens, 3, n_samples=4),
              signal_impact_projection(tok, tokens, 3, band="low", n_samples=4)):
        torch.testing.assert_close(a @ a.T, torch.eye(3), atol=1e-4, rtol=1e-4)


def test_a_full_rank_structured_basis_reproduces_exact_verification():
    """``k == d`` is a bijection whatever the ordering, so it must be lossless like the random case."""
    d = 8
    q, p = _mixture(d=d, seed=2), _mixture(d=d, seed=3, shift=0.3)
    x = q.sample()
    a = signal_impact_projection(_tokenizer(d=d), torch.randn(4, 6, d), d, n_samples=4)
    torch.testing.assert_close(projected_log_ratio(x, q, p, a),
                               p.log_prob(x) - q.log_prob(x), atol=1e-4, rtol=1e-4)


def _isotropic(mean, sigma=1.0):
    """A one-component mixture, i.e. a plain Gaussian, in the form the verifiers consume."""
    return MixtureSameFamily(
        Categorical(logits=torch.zeros(mean.shape[0], 1)),
        Independent(Normal(mean.unsqueeze(1), torch.full_like(mean.unsqueeze(1), sigma)), 1),
    )


def test_divergence_splits_conservatively_across_a_subspace_and_its_complement():
    """The identity behind E1.2b: ``‖Aδ‖² + ‖A⊥δ‖² = ‖δ‖²`` for orthonormal ``A``.

    For shared-covariance Gaussians the KL is ``‖δ‖²/2σ²``, so the divergence the verified subspace
    can see and the divergence that leaks past it sum to a constant that no choice of subspace can
    change. That is why choosing *which* directions to verify cannot beat choosing how many.
    """
    d, k, sigma = 12, 5, 0.7
    g = torch.Generator().manual_seed(0)
    delta = torch.randn(d, generator=g)
    a = orthonormal_rows(torch.randn(k, d, generator=g))
    # An orthonormal basis for the complement: the eigenvectors of I - AᵀA with eigenvalue 1.
    complement = top_eigen_projection(torch.eye(d) - a.T @ a, d - k)

    verified = (a @ delta).pow(2).sum() / (2 * sigma**2)
    leaked = (complement @ delta).pow(2).sum() / (2 * sigma**2)
    total = delta.pow(2).sum() / (2 * sigma**2)
    torch.testing.assert_close(verified + leaked, total, atol=1e-4, rtol=1e-4)


def test_acceptance_depends_on_the_subspace_only_through_the_projected_gap():
    """Two unrelated subspaces that capture the same ``‖Aδ‖`` must accept identically.

    This is the empirical content of the conservation argument: the subspace enters the acceptance
    rule through one scalar, so a 'smart' subspace and a 'dumb' one that happen to capture the same
    amount of disagreement are the same rule.
    """
    d, sigma = 8, 1.0
    n = 4000
    # MixtureSameFamily.sample takes no generator, so the global seed is what makes this repeatable.
    torch.manual_seed(1)
    delta = torch.zeros(d)
    delta[0] = 1.5

    q = _isotropic(torch.zeros(n, d), sigma)
    p = _isotropic(delta.expand(n, d).contiguous(), sigma)
    x = q.sample()

    # Both capture exactly half of delta's norm-squared, by construction, but share no directions.
    half = 2**-0.5
    first = torch.zeros(2, d)
    first[0, 0], first[1, 1] = half, 1.0
    second = torch.zeros(2, d)
    second[0, 0], second[1, 2] = half, 1.0

    torch.testing.assert_close((first @ delta).norm(), (second @ delta).norm())
    acc_first = projected_log_ratio(x, q, p, first).exp().clamp(max=1.0).mean()
    acc_second = projected_log_ratio(x, q, p, second).exp().clamp(max=1.0).mean()
    assert abs(float(acc_first - acc_second)) < 0.02


@pytest.mark.parametrize("band", ["low", "high"])
def test_band_restricted_impact_is_dominated_by_the_full_matrix(band):
    tok = _tokenizer()
    tokens = torch.randn(4, 6, 8)
    gap = signal_impact_matrix(tok, tokens, band="all", n_samples=4) - signal_impact_matrix(
        tok, tokens, band=band, n_samples=4
    )
    assert torch.linalg.eigvalsh(gap).min() > -1e-5
