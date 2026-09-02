"""Verify a linear statistic of a continuous token instead of the token itself.

This is the machinery behind the E1.2/E1.3 ladder. The speculative setting is unchanged -- the draft
still proposes a *full* token ``x ~ q`` -- but the acceptance test is applied to ``s = A x`` for some
``k x d`` matrix ``A`` with ``k <= d``. The accepted object is still the full token; what is verified
is only its image under ``A``.

The point of doing this is the dimensional collapse E1.1/E1.1b measured: exact acceptance in the raw
token space falls roughly as ``exp(-c*d)`` at fixed per-coordinate disagreement, so at MAR's ``d=16``
it is already low and at ``d=128`` it is hopeless. Verifying a low-dimensional statistic restores
acceptance because the draft and target only have to agree about ``A x``.

**This is a real relaxation, not a trick.** Everything ``A`` discards is unverified: draft and target
may disagree arbitrarily in the null space of ``A`` and the token is still accepted. So acceptance
gains here are not free, and a bare acceptance number is not a result. That is exactly why E1.2a
exists as the null control -- *any* projection buys acceptance simply by reducing the dimension (the
ABC literature has known this since Fearnhead & Prangle 2012), so the question for a *structured*
statistic is never "does it help" but "does it beat a random projection at matched ``k``".

The mixture head emits diagonal Gaussians, but a projection of a diagonal Gaussian is not diagonal:
``A diag(σ²) Aᵀ`` is a full ``k x k`` covariance. The pushforward is computed exactly rather than
approximated, so the projected acceptance test is itself lossless *with respect to the statistic*.
"""

from __future__ import annotations

import torch
from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal


def random_projection(
    k: int,
    d: int,
    generator: torch.Generator | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """A ``(k, d)`` matrix with orthonormal rows, drawn uniformly from the Stiefel manifold.

    Orthonormal rather than i.i.d. Gaussian so that ``A`` neither stretches nor shrinks the space it
    keeps. A Gaussian matrix has singular values spread around ``√d``, which would confound "this
    projection discards the disagreement" with "this projection rescales it" -- and rescaling alone
    moves acceptance, so the null control has to be free of it.

    When ``k == d`` the result is a full orthogonal matrix, which makes the acceptance test provably
    identical to the unprojected one. That is the invariance the tests use as a correctness anchor.
    """
    if not 1 <= k <= d:
        raise ValueError(f"need 1 <= k <= d, got k={k}, d={d}")
    a = torch.randn(d, k, generator=generator, device=device, dtype=dtype)
    q, r = torch.linalg.qr(a)
    # QR is only unique up to the signs of R's diagonal; fixing them makes the draw uniform on the
    # Stiefel manifold rather than biased toward whatever sign convention LAPACK happened to pick.
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    return q.T.contiguous()


def project_mixture(
    dist: MixtureSameFamily, projection: torch.Tensor, jitter: float = 1e-6
) -> MixtureSameFamily:
    """Push a diagonal-Gaussian mixture through ``A``, exactly.

    Component ``i`` with mean ``μ_i`` and diagonal covariance ``diag(σ_i²)`` maps to a Gaussian with
    mean ``A μ_i`` and covariance ``A diag(σ_i²) Aᵀ``. Mixture weights are unchanged: a linear map
    does not move mass between components.

    Args:
        dist: the head's output -- ``MixtureSameFamily`` over ``Independent(Normal, 1)``.
        projection: ``(k, d)``.
        jitter: added to the diagonal. The exact covariance is positive definite whenever ``A`` has
            full row rank and every ``σ > 0``, but it can be numerically borderline when the head
            saturates ``min_logstd``, and a Cholesky failure mid-sweep is a worse outcome than a
            ``1e-6`` perturbation.
    """
    comp = dist.component_distribution.base_dist
    mean, std = comp.loc, comp.scale
    k, d = projection.shape
    if mean.shape[-1] != d:
        raise ValueError(f"projection expects tokens of dim {d}, got {mean.shape[-1]}")

    proj_mean = mean @ projection.T
    # cov[a,b] = sum_j A[a,j] var[j] A[b,j]
    scaled = projection * (std**2).unsqueeze(-2)
    cov = scaled @ projection.T
    cov = cov + jitter * torch.eye(k, device=cov.device, dtype=cov.dtype)
    return MixtureSameFamily(Categorical(logits=dist.mixture_distribution.logits),
                             MultivariateNormal(proj_mean, covariance_matrix=cov))


def projected_log_ratio(
    x: torch.Tensor,
    draft: MixtureSameFamily,
    target: MixtureSameFamily,
    projection: torch.Tensor,
) -> torch.Tensor:
    """``log p_A(Ax) - log q_A(Ax)`` -- the acceptance ratio for the statistic ``A x``.

    The draft proposes the full token ``x``; only its projection is verified.
    """
    s = x @ projection.T
    return project_mixture(target, projection).log_prob(s) - project_mixture(draft, projection).log_prob(s)
