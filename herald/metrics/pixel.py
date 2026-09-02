"""Paired pixel-domain metrics for E0.4 and E2.5.

These exist because FID and CMMD cannot answer the question speculative decoding actually makes.
Both compare a bag of generated samples against a bag of real ones, so they are insensitive by
construction to a per-image deviation that leaves the marginal distribution intact. CSpD's defense
of ``inner_temperature=42.`` is precisely that FID barely moves. The metrics here compare an
accepted-draft decode against the target-sampler decode *of the same seed*, which is the only way
to see what the acceptance rule actually did to the image.

Everything in this module takes images as float tensors ``(N, C, H, W)``. Pass ``data_range`` to
match your convention -- VAE decoders in this project emit roughly ``[-1, 1]``, so ``data_range=2.0``
is usual, while ``[0, 1]`` images want ``1.0``. Getting it wrong silently rescales PSNR and SSIM.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

_SOBEL_X = torch.tensor(
    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
)


def paired_psnr(a: torch.Tensor, b: torch.Tensor, data_range: float = 2.0) -> torch.Tensor:
    """Per-image PSNR between two decodes. Returns ``(N,)``.

    Identical inputs give ``inf``; the caller should expect that rather than a large finite
    number, since an accepted block that was never modified is a real and common case.
    """
    mse = ((a - b) ** 2).flatten(1).mean(-1)
    return 10.0 * torch.log10(data_range**2 / mse)


def radially_averaged_psd(
    x: torch.Tensor, n_bins: int = 64
) -> tuple[torch.Tensor, torch.Tensor]:
    """Radially averaged power spectral density, averaged over channels.

    Args:
        x: ``(N, C, H, W)``.
        n_bins: number of radial frequency bins spanning DC to Nyquist.

    Returns:
        ``(freqs, psd)`` where ``freqs`` is ``(n_bins,)`` in cycles per pixel on ``[0, 0.5]`` and
        ``psd`` is ``(N, n_bins)``.

    Frequencies beyond Nyquist (the spectrum's corners) are dropped rather than folded into the
    last bin, which would otherwise put a spurious shelf at the high end of every curve.
    """
    n, c, h, w = x.shape
    fy = torch.fft.fftfreq(h, device=x.device).view(-1, 1)
    fx = torch.fft.fftfreq(w, device=x.device).view(1, -1)
    radius = torch.sqrt(fy**2 + fx**2).flatten()

    nyquist = 0.5
    keep = radius <= nyquist
    bin_idx = torch.clamp((radius / nyquist * n_bins).long(), max=n_bins - 1)

    power = (torch.fft.fft2(x).abs() ** 2 / (h * w)).reshape(n * c, h * w)
    power, bin_idx_kept = power[:, keep], bin_idx[keep]

    sums = torch.zeros(n * c, n_bins, device=x.device, dtype=power.dtype)
    sums.scatter_add_(1, bin_idx_kept.expand(n * c, -1), power)
    counts = torch.bincount(bin_idx_kept, minlength=n_bins).clamp(min=1)

    psd = (sums / counts).reshape(n, c, n_bins).mean(1)
    freqs = (torch.arange(n_bins, device=x.device) + 0.5) / n_bins * nyquist
    return freqs, psd


def residual_spectrum(
    accepted: torch.Tensor, target: torch.Tensor, n_bins: int = 64
) -> dict[str, torch.Tensor]:
    """Spectral profile of the accepted-vs-target residual.

    ``relative`` is the quantity to read: residual power divided by target power in the same
    band, so a value of 0.1 in the top band means the acceptance rule perturbed a tenth of the
    image's high-frequency energy. Raw residual power alone is uninformative because natural
    images have far less power at high frequency to begin with, so an absolute residual spectrum
    slopes downward no matter what the acceptance rule did.
    """
    freqs, psd_residual = radially_averaged_psd(accepted - target, n_bins)
    _, psd_target = radially_averaged_psd(target, n_bins)
    return {
        "freqs": freqs,
        "residual": psd_residual,
        "target": psd_target,
        "relative": psd_residual / psd_target.clamp_min(torch.finfo(psd_target.dtype).tiny),
    }


def gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Sobel gradient magnitude, averaged over channels. Returns ``(N, 1, H, W)``."""
    c = x.shape[1]
    kx = _SOBEL_X.to(x.device, x.dtype).view(1, 1, 3, 3).expand(c, 1, 3, 3)
    ky = kx.transpose(-1, -2).contiguous()
    padded = F.pad(x, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(padded, kx, groups=c)
    gy = F.conv2d(padded, ky, groups=c)
    return torch.sqrt(gx**2 + gy**2 + 1e-12).mean(1, keepdim=True)


def local_std(x: torch.Tensor, window: int = 7) -> torch.Tensor:
    """Local standard deviation in a box window, averaged over channels. ``(N, 1, H, W)``."""
    pad = window // 2
    padded = F.pad(x, (pad,) * 4, mode="replicate")
    mean = F.avg_pool2d(padded, window, stride=1)
    mean_sq = F.avg_pool2d(padded**2, window, stride=1)
    return torch.sqrt((mean_sq - mean**2).clamp_min(0.0) + 1e-12).mean(1, keepdim=True)


def region_masks(
    x: torch.Tensor,
    edge_quantile: float = 0.9,
    texture_quantile: float = 0.5,
    window: int = 7,
) -> dict[str, torch.Tensor]:
    """Partition pixels into edge, texture, and flat regions. Each mask is ``(N, 1, H, W)`` bool.

    Edges are the top ``1 - edge_quantile`` of Sobel gradient magnitude. Texture is what remains
    above the ``texture_quantile`` of local standard deviation -- high local variance without a
    coherent gradient direction. Flat is everything else.

    Thresholds are per-image quantiles, not absolute values, so the three regions occupy roughly
    the same pixel fractions across images of very different contrast. That makes the regional
    residual energies comparable across a batch, which absolute thresholds would not.

    Comparison against the quantile is strict. On an image with large exactly-constant areas the
    quantile itself can be zero -- a synthetic step edge is the extreme case, where over 90% of
    pixels have zero gradient -- and a non-strict test would then mark every pixel an edge and
    leave the flat region empty.
    """
    grad, sigma = gradient_magnitude(x), local_std(x, window)

    def _threshold(v: torch.Tensor, q: float) -> torch.Tensor:
        return torch.quantile(v.flatten(1), q, dim=1).view(-1, 1, 1, 1)

    edge = grad > _threshold(grad, edge_quantile)
    texture = (~edge) & (sigma > _threshold(sigma, texture_quantile))
    return {"edge": edge, "texture": texture, "flat": ~(edge | texture)}


def regional_residual_energy(
    accepted: torch.Tensor, target: torch.Tensor, masks: dict[str, torch.Tensor]
) -> dict[str, dict[str, torch.Tensor]]:
    """Where the accepted-vs-target residual lives, by region.

    Reports both ``mean_sq`` (per-pixel residual energy inside the region) and ``share`` (that
    region's fraction of total residual energy) because they answer different questions and
    disagree whenever the regions differ in size. ``share`` alone would make flat regions look
    catastrophic purely because there are more flat pixels; ``mean_sq`` alone hides the fact that
    a severe artifact confined to 2% of pixels may not matter. ``pixel_fraction`` is returned so
    neither number is read without its denominator.
    """
    sq = ((accepted - target) ** 2).mean(1, keepdim=True)
    total = sq.flatten(1).sum(-1).clamp_min(torch.finfo(sq.dtype).tiny)

    out = {}
    for name, mask in masks.items():
        count = mask.flatten(1).sum(-1)
        energy = (sq * mask).flatten(1).sum(-1)
        out[name] = {
            "mean_sq": energy / count.clamp_min(1),
            "share": energy / total,
            "pixel_fraction": count / mask[0].numel(),
        }
    return out


@dataclass
class PairedImageAudit:
    """The E0.4 measurement bundle: one call, every paired number the plan asks for.

    ``lpips_net`` is loaded lazily on first use because it pulls pretrained weights over the
    network, which should not happen on import in a batch job. Set ``use_lpips=False`` to skip it
    entirely -- the spectral and regional numbers do not depend on it.
    """

    data_range: float = 2.0
    n_bins: int = 64
    use_lpips: bool = True
    lpips_net: str = "alex"
    _lpips: object | None = field(default=None, init=False, repr=False)
    _ssim: object | None = field(default=None, init=False, repr=False)

    def _to_unit(self, x: torch.Tensor) -> torch.Tensor:
        """Map to [0, 1] for the torchmetrics implementations, which assume it."""
        lo = -self.data_range / 2 if self.data_range == 2.0 else 0.0
        return ((x - lo) / self.data_range).clamp(0.0, 1.0)

    def __call__(self, accepted: torch.Tensor, target: torch.Tensor) -> dict:
        """Compare accepted-draft decodes against target-sampler decodes, pairwise.

        Args:
            accepted, target: ``(N, C, H, W)``, matched element-wise by seed. Pairing is the whole
                point -- passing unpaired batches produces numbers that look plausible and mean
                nothing.
        """
        if accepted.shape != target.shape:
            raise ValueError(f"paired audit needs matched shapes, got {accepted.shape} vs {target.shape}")

        result: dict = {
            "psnr": paired_psnr(accepted, target, self.data_range),
            "spectrum": residual_spectrum(accepted, target, self.n_bins),
            "regions": regional_residual_energy(accepted, target, region_masks(target)),
        }
        result.update(self._perceptual(accepted, target))
        return result

    def _perceptual(self, accepted: torch.Tensor, target: torch.Tensor) -> dict:
        from torchmetrics.functional.image import (
            multiscale_structural_similarity_index_measure as ms_ssim,
        )
        from torchmetrics.functional.image import structural_similarity_index_measure as ssim

        a, b = self._to_unit(accepted), self._to_unit(target)
        out = {
            "ssim": ssim(a, b, data_range=1.0, reduction="none"),
        }
        # MS-SSIM needs enough resolution for its five-scale pyramid; below ~160px it raises.
        if min(a.shape[-2:]) >= 160:
            out["ms_ssim"] = ms_ssim(a, b, data_range=1.0, reduction="none")

        if self.use_lpips:
            out["lpips"] = self._lpips_fn(a, b)
        return out

    def _lpips_fn(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self._lpips is None:
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

            self._lpips = LearnedPerceptualImagePatchSimilarity(
                net_type=self.lpips_net, normalize=True, reduction="mean"
            ).to(a.device)
        # Per-image rather than batch-mean: the distribution over images is the interesting part,
        # and a batch mean would hide a bimodal split between untouched and heavily edited images.
        return torch.stack([self._lpips(a[i : i + 1], b[i : i + 1]).detach() for i in range(a.shape[0])])


def summarize_audit(result: dict) -> dict[str, float]:
    """Flatten an audit into scalars for the experiment log.

    PSNR is reported as a median because identical pairs contribute ``inf``, which destroys a mean.
    """
    psnr = result["psnr"]
    out: dict[str, float] = {
        "psnr_median": psnr.median().item(),
        "psnr_finite_frac": torch.isfinite(psnr).float().mean().item(),
        "ssim_mean": result["ssim"].mean().item(),
    }
    for key in ("ms_ssim", "lpips"):
        if key in result:
            out[f"{key}_mean"] = result[key].mean().item()

    rel = result["spectrum"]["relative"].mean(0)
    third = rel.shape[0] // 3
    out["relative_residual_low_band"] = rel[:third].mean().item()
    out["relative_residual_high_band"] = rel[-third:].mean().item()

    for name, stats in result["regions"].items():
        out[f"residual_share_{name}"] = stats["share"].mean().item()
        out[f"residual_mean_sq_{name}"] = stats["mean_sq"].mean().item()
    return out
