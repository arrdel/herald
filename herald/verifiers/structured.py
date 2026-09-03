"""Structured choices of *which* ``k`` dimensions to verify.

E1.2a established that acceptance depends only on the verified dimension ``k``, and E1.2-frontier
that the fidelity cost depends on the ``d - k`` dimensions left unverified. So an acceptance rule is
really a choice of **subspace**, and the only way one rule beats another at matched ``k`` is by
discarding directions that matter less.

A useful fact that makes the comparison clean: for an invertible ``k x k`` matrix ``Q``, the
statistics ``A x`` and ``QA x`` induce the same acceptance ratio, because the Jacobian ``|det Q|``
appears in both the numerator and the denominator and cancels. **Only the row space of ``A``
matters.** Every rule here therefore returns an orthonormal basis of its chosen subspace, and the
comparison against ``random_projection`` is like for like.

The rules, in increasing order of how much they know about the signal:

``pca_projection``
    Top-``k`` principal directions of the token distribution. Knows the data, not the decoder.
``signal_impact_projection(band="all")``
    Top-``k`` eigenvectors of ``E[JᵀJ]`` for the decoder Jacobian ``J``. The directions that move the
    reconstructed signal most -- a perceptual criterion in the sense that it weights latent
    directions by how much of the *output* they control, which is what E0.5 measured for KL16.
``signal_impact_projection(band="low"/"high")``
    The same, restricted to a DCT band of the decoded patch: ``E[Jᵀ Cᵀ C J]`` for the band selector
    ``C``. This is the transform-domain statistic of E1.2b, made linear by composing the DCT with
    the decoder Jacobian so the pushforward stays exactly Gaussian and the projected test stays
    exact with respect to the statistic.
``signal_impact_projection(..., largest=False)``
    The anti-control: verify the directions that matter *least*. It should be clearly worse than
    random. If it is not, the fidelity metric is not measuring what it claims to.
"""

from __future__ import annotations

import torch

from herald.models.ts_tokenizer import TimeSeriesTokenizer


def orthonormal_rows(a: torch.Tensor) -> torch.Tensor:
    """Re-express a ``(k, d)`` map as an orthonormal basis of the same row space."""
    q, _ = torch.linalg.qr(a.T)
    return q.T.contiguous()


def pca_projection(tokens: torch.Tensor, k: int) -> torch.Tensor:
    """Top-``k`` principal directions of ``tokens`` ``(N, d)``."""
    x = tokens.reshape(-1, tokens.shape[-1])
    x = x - x.mean(0, keepdim=True)
    # Eigenvectors of the covariance, which for a centred design matrix are the right singular
    # vectors; SVD is used directly because it is better conditioned than forming XᵀX.
    _, _, vh = torch.linalg.svd(x, full_matrices=False)
    return vh[:k].contiguous()


def dct_matrix(n: int, dtype=torch.float32) -> torch.Tensor:
    """Orthonormal DCT-II matrix, rows ordered from lowest to highest frequency."""
    i = torch.arange(n, dtype=dtype).unsqueeze(1)
    f = torch.arange(n, dtype=dtype).unsqueeze(0)
    m = torch.cos(torch.pi * (i + 0.5) * f / n).T
    m[0] *= (1.0 / n) ** 0.5
    m[1:] *= (2.0 / n) ** 0.5
    return m


def decoder_jacobians(
    tokenizer: TimeSeriesTokenizer,
    tokens: torch.Tensor,
    position: int | None = None,
    n_samples: int = 32,
) -> torch.Tensor:
    """Jacobians of the decoded patch with respect to one token, ``(n, C*patch, d)``.

    The decoder has temporal convolutions, so a token in isolation is not representative of a token
    in context: each Jacobian is taken with the *rest of the grid* held at real values, and only the
    output patch owned by ``position`` is differentiated. That keeps the object local and square
    enough to eigendecompose while remaining the derivative the model actually realises.
    """
    tokenizer = tokenizer.eval()
    grids = tokens[:n_samples]
    n_tokens = grids.shape[1]
    pos = n_tokens // 2 if position is None else position
    patch = tokenizer.config.patch
    lo, hi = pos * patch, (pos + 1) * patch

    jacobians = []
    for grid in grids:
        base = grid.unsqueeze(0)

        def patch_of(z_t: torch.Tensor, base=base) -> torch.Tensor:
            z = base.clone()
            z[:, pos] = z_t
            return tokenizer.decode(z)[0, :, lo:hi].reshape(-1)

        jacobians.append(torch.autograd.functional.jacobian(patch_of, base[0, pos], vectorize=True))
    return torch.stack(jacobians)


def signal_impact_matrix(
    tokenizer: TimeSeriesTokenizer,
    tokens: torch.Tensor,
    band: str = "all",
    n_samples: int = 32,
    position: int | None = None,
) -> torch.Tensor:
    """``E[Jᵀ Cᵀ C J]`` -- how much decoded signal energy each latent direction controls.

    Args:
        band: ``"all"`` for the raw patch, ``"low"`` or ``"high"`` for the lower or upper half of the
            DCT spectrum of the patch along time. ``C`` is the corresponding band selector applied
            per channel, so the ``"low"`` matrix ranks latent directions by how much *low-frequency*
            content they produce.
    """
    jac = decoder_jacobians(tokenizer, tokens, position=position, n_samples=n_samples)
    n, out_dim, d = jac.shape
    patch = tokenizer.config.patch
    n_channels = out_dim // patch

    if band != "all":
        dct = dct_matrix(patch, dtype=jac.dtype)
        half = patch // 2
        rows = dct[:half] if band == "low" else dct[half:]
        # Apply the band selector per channel: the Jacobian's output axis is (channel, time).
        j = jac.reshape(n, n_channels, patch, d)
        jac = torch.einsum("fp,ncpd->ncfd", rows, j).reshape(n, -1, d)

    return torch.einsum("nod,noe->de", jac, jac) / n


def top_eigen_projection(matrix: torch.Tensor, k: int, largest: bool = True) -> torch.Tensor:
    """Orthonormal basis of the top (or bottom) ``k`` eigenvectors of a symmetric PSD matrix."""
    evals, evecs = torch.linalg.eigh(matrix)
    idx = torch.argsort(evals, descending=largest)[:k]
    return evecs[:, idx].T.contiguous()


def signal_impact_projection(
    tokenizer: TimeSeriesTokenizer,
    tokens: torch.Tensor,
    k: int,
    band: str = "all",
    largest: bool = True,
    n_samples: int = 32,
) -> torch.Tensor:
    """Verify the ``k`` latent directions carrying the most (or least) decoded signal energy."""
    m = signal_impact_matrix(tokenizer, tokens, band=band, n_samples=n_samples)
    return top_eigen_projection(m, k, largest=largest)
