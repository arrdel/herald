"""E0.5 -- how a latent perturbation maps to pixel deviation, as a function of direction.

Every acceptance rule in this project thresholds an isotropic norm in latent space: accept the
block if ``||E_ema(x̂) - z|| <= τ``. That is only the right shape if the decoder's gain is roughly
equal in every latent direction. If it is not, then two drafts at identical latent distance from
the target produce visibly different images, one of them fine and one of them not, and a single
scalar τ cannot separate them. The remedy in that case is a Jacobian-weighted nonconformity score,
which is a change to ``herald/calibration/`` -- so this module exists to decide a design question,
not to fill a section of the report.

The decoder Jacobian is far too large to form (roughly 196k x 4k for a 256x256 image over KL16
latents), so the spectrum comes from randomized SVD using matrix-free products: JVPs give ``J @ v``
and VJPs give ``Jᵀ @ u``, and neither ever materializes ``J``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

DecodeFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class JacobianSpectrum:
    """Local spectrum of the decoder Jacobian at one latent point.

    Attributes:
        singular_values: ``(k,)`` descending. Pixel-space gain per unit latent perturbation along
            each corresponding direction.
        right_vectors: ``(k, *latent_shape)`` latent-space directions, aligned with the singular
            values. The first is the most amplified direction, the last the most attenuated.
        latent_shape: shape of the latent this was computed at.
    """

    singular_values: torch.Tensor
    right_vectors: torch.Tensor
    latent_shape: torch.Size

    @property
    def anisotropy(self) -> float:
        """Ratio of largest to smallest recovered singular value.

        The number that decides E0.5. Near 1 means an isotropic threshold is defensible and the
        conformal score stays as it is. Large means equal-norm latent perturbations produce very
        unequal pixel changes, and the score should be reweighted. Note this is a *local* ratio at
        one latent point over the top ``k`` directions, so it is a lower bound on the full
        condition number -- do not report it as one.
        """
        s = self.singular_values
        return (s[0] / s[-1].clamp_min(torch.finfo(s.dtype).tiny)).item()

    def effective_rank(self) -> float:
        """Entropy-based effective rank of the recovered spectrum.

        Complements ``anisotropy``, which is a two-point summary and therefore fragile: a single
        tiny singular value makes it enormous. Effective rank says how many directions genuinely
        carry pixel response, which is the more stable statement about whether the latent geometry
        is close to isotropic.
        """
        p = self.singular_values / self.singular_values.sum().clamp_min(
            torch.finfo(self.singular_values.dtype).tiny
        )
        p = p[p > 0]
        return torch.exp(-(p * p.log()).sum()).item()


def _flat_jvp(decode: DecodeFn, z: torch.Tensor, tangent_flat: torch.Tensor) -> torch.Tensor:
    tangent = tangent_flat.view_as(z)
    _, out = torch.func.jvp(decode, (z,), (tangent,))
    return out.flatten()


def _flat_vjp(vjp_fn: Callable, cotangent_flat: torch.Tensor, out_shape: torch.Size) -> torch.Tensor:
    cotangent = cotangent_flat.view(out_shape)
    return vjp_fn(cotangent)[0].flatten()


def jacobian_spectrum(
    decode: DecodeFn,
    z: torch.Tensor,
    k: int = 32,
    oversample: int = 8,
    power_iters: int = 2,
    generator: torch.Generator | None = None,
) -> JacobianSpectrum:
    """Randomized SVD of the decoder Jacobian at ``z``.

    Args:
        decode: latent ``->`` image. Must be differentiable and accept exactly ``z``'s shape.
        z: a single latent, no batch dimension beyond whatever ``decode`` expects.
        k: number of singular values to recover.
        oversample: extra random probes, discarded before returning. Randomized SVD is only
            accurate for the top ``k`` when the sketch has room to spare.
        power_iters: subspace iterations. Decoder spectra decay slowly enough that zero iterations
            noticeably biases the small singular values upward; two is the usual compromise.

    Returns:
        A :class:`JacobianSpectrum`. Costs ``(k + oversample) * (1 + 2 * power_iters)`` JVPs and
        about as many VJPs, so keep ``k`` modest on full-resolution decoders.
    """
    z = z.detach()
    n_latent = z.numel()
    sketch = min(k + oversample, n_latent)

    probe_out, vjp_fn = torch.func.vjp(decode, z)
    out_shape = probe_out.shape

    omega = torch.randn(n_latent, sketch, device=z.device, dtype=z.dtype, generator=generator)

    def apply_j(mat: torch.Tensor) -> torch.Tensor:
        return torch.stack([_flat_jvp(decode, z, mat[:, i]) for i in range(mat.shape[1])], dim=1)

    def apply_jt(mat: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [_flat_vjp(vjp_fn, mat[:, i], out_shape) for i in range(mat.shape[1])], dim=1
        )

    y = apply_j(omega)
    for _ in range(power_iters):
        q, _ = torch.linalg.qr(y)
        y = apply_j(apply_jt(q))

    q, _ = torch.linalg.qr(y)
    b = apply_jt(q).T  # (sketch, n_latent)

    _, s, vh = torch.linalg.svd(b, full_matrices=False)
    keep = min(k, s.shape[0])
    return JacobianSpectrum(
        singular_values=s[:keep],
        right_vectors=vh[:keep].reshape(keep, *z.shape),
        latent_shape=z.shape,
    )


def directional_response(
    decode: DecodeFn,
    z: torch.Tensor,
    direction: torch.Tensor,
    eps: float = 1e-2,
    central: bool = True,
) -> torch.Tensor:
    """Decoded pixel response to a unit-norm latent perturbation along ``direction``.

    Uses a central difference by default. The decoder is not linear, and the whole question here
    is whether the local linearization is a good guide, so a finite difference at the ε the method
    will actually see is a more honest probe than the Jacobian alone.
    """
    d = direction / direction.norm().clamp_min(torch.finfo(direction.dtype).tiny)
    with torch.no_grad():
        if central:
            return (decode(z + eps * d) - decode(z - eps * d)) / 2.0
        return decode(z + eps * d) - decode(z)


def direction_gain_profile(
    decode: DecodeFn,
    z: torch.Tensor,
    spectrum: JacobianSpectrum,
    eps: float = 1e-2,
    n_bins: int = 64,
) -> dict:
    """Pixel response along the most- and least-amplified latent directions.

    Returns the measured gains alongside the linearized prediction from the Jacobian. When the two
    disagree the linearization is not valid at this ε, and a Jacobian-weighted score built on it
    would be reweighting by the wrong numbers -- which is worth knowing before building one.
    """
    from .pixel import radially_averaged_psd

    out: dict = {}
    for label, idx in (("top", 0), ("bottom", -1)):
        response = directional_response(decode, z, spectrum.right_vectors[idx], eps)
        freqs, psd = radially_averaged_psd(response.unsqueeze(0) if response.dim() == 3 else response, n_bins)
        out[label] = {
            "response": response,
            "measured_gain": (response.norm() / eps).item(),
            "predicted_gain": spectrum.singular_values[idx].item(),
            "freqs": freqs,
            "psd": psd,
        }
    return out
