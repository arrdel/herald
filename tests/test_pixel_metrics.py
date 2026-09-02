"""Checks for the E0.4/E0.5 pixel-domain machinery, on synthetic inputs with known answers.

These are correctness tests, not experiments. The point is that when E0.4 reports "the residual
is concentrated at high frequency on edges", that statement is the measurement and not a bug in
the measuring instrument -- so each function is checked against a case where the right answer is
known in advance.
"""

import math

import pytest
import torch

from herald.metrics import (
    PairedImageAudit,
    jacobian_spectrum,
    paired_psnr,
    radially_averaged_psd,
    region_masks,
    regional_residual_energy,
    residual_spectrum,
    summarize_audit,
)


def _sinusoid(freq: float, size: int = 64) -> torch.Tensor:
    """Horizontal sinusoid at ``freq`` cycles per pixel."""
    x = torch.arange(size, dtype=torch.float32)
    return torch.sin(2 * math.pi * freq * x).view(1, 1, 1, size).expand(1, 1, size, size).contiguous()


def test_psnr_matches_closed_form():
    a = torch.zeros(3, 1, 8, 8)
    b = torch.full((3, 1, 8, 8), 0.1)
    expected = 10 * math.log10(2.0**2 / 0.01)
    torch.testing.assert_close(
        paired_psnr(a, b, data_range=2.0), torch.full((3,), expected), rtol=1e-5, atol=1e-4
    )


def test_psnr_is_infinite_for_identical_pairs():
    a = torch.randn(2, 3, 8, 8)
    assert torch.isinf(paired_psnr(a, a.clone())).all()


def test_white_noise_spectrum_is_flat():
    torch.manual_seed(0)
    _, psd = radially_averaged_psd(torch.randn(8, 1, 64, 64), n_bins=16)
    mean_psd = psd.mean(0)
    # White noise has equal expected power at every frequency; allow generous slack for the small
    # number of samples in the low-frequency bins.
    assert (mean_psd.max() / mean_psd.min()).item() < 2.5, mean_psd


def test_sinusoid_peaks_in_the_correct_radial_bin():
    n_bins = 32
    freq = 0.25
    _, psd = radially_averaged_psd(_sinusoid(freq), n_bins=n_bins)
    peak = psd[0].argmax().item()
    expected = int(freq / 0.5 * n_bins)
    assert abs(peak - expected) <= 1, f"peak at bin {peak}, expected near {expected}"


def test_blurred_image_loses_high_frequency_power():
    torch.manual_seed(0)
    sharp = torch.randn(4, 1, 64, 64)
    kernel = torch.ones(1, 1, 5, 5) / 25
    blurred = torch.nn.functional.conv2d(torch.nn.functional.pad(sharp, (2,) * 4, mode="reflect"), kernel)

    _, psd_sharp = radially_averaged_psd(sharp, n_bins=16)
    _, psd_blur = radially_averaged_psd(blurred, n_bins=16)
    ratio = psd_blur.mean(0) / psd_sharp.mean(0)
    assert ratio[-4:].mean() < 0.1 * ratio[:4].mean(), ratio


def test_relative_residual_spectrum_localizes_injected_high_frequency_error():
    """The measurement E0.4 hinges on: does the residual sit in a specific band?"""
    torch.manual_seed(0)
    target = torch.randn(4, 1, 64, 64)
    kernel = torch.ones(1, 1, 7, 7) / 49
    target = torch.nn.functional.conv2d(torch.nn.functional.pad(target, (3,) * 4, mode="reflect"), kernel)

    accepted = target + 0.01 * _sinusoid(0.4, 64)

    rel = residual_spectrum(accepted, target, n_bins=32)["relative"].mean(0)
    third = rel.shape[0] // 3
    assert rel[-third:].mean() > 100 * rel[:third].mean(), rel


def test_region_masks_partition_every_pixel_exactly_once():
    torch.manual_seed(0)
    x = torch.randn(3, 3, 32, 32)
    masks = region_masks(x)
    stacked = torch.stack([masks[k] for k in ("edge", "texture", "flat")]).long().sum(0)
    assert (stacked == 1).all()


def test_edge_mask_finds_a_synthetic_step_edge():
    x = torch.zeros(1, 1, 32, 32)
    x[..., 16:] = 1.0
    edge = region_masks(x, edge_quantile=0.9)["edge"]

    # Sobel responds in the columns straddling the step; nothing far from it should fire.
    columns = edge[0, 0].any(0).nonzero().flatten()
    assert columns.numel() > 0
    assert (columns - 16).abs().max().item() <= 2, columns


def test_regional_energy_attributes_a_flat_region_perturbation_correctly():
    x = torch.zeros(1, 1, 32, 32)
    x[..., 16:] = 1.0
    masks = region_masks(x)

    residual = torch.zeros_like(x)
    residual[masks["flat"].expand_as(residual)] = 0.5
    stats = regional_residual_energy(x + residual, x, masks)

    assert stats["flat"]["share"].item() > 0.99
    assert stats["edge"]["share"].item() < 0.01
    total_fraction = sum(stats[k]["pixel_fraction"].item() for k in stats)
    assert abs(total_fraction - 1.0) < 1e-5


def test_audit_runs_end_to_end_and_summarizes():
    torch.manual_seed(0)
    target = torch.rand(2, 3, 64, 64) * 2 - 1
    accepted = target + 0.02 * torch.randn_like(target)

    result = PairedImageAudit(data_range=2.0, use_lpips=False)(accepted, target)
    summary = summarize_audit(result)

    assert summary["psnr_median"] > 20
    assert 0.0 <= summary["ssim_mean"] <= 1.0
    assert "ms_ssim_mean" not in summary, "64px is below the MS-SSIM pyramid minimum"
    assert abs(sum(summary[f"residual_share_{k}"] for k in ("edge", "texture", "flat")) - 1.0) < 1e-4


def test_audit_rejects_unpaired_batches():
    with pytest.raises(ValueError, match="matched shapes"):
        PairedImageAudit(use_lpips=False)(torch.rand(2, 3, 8, 8), torch.rand(3, 3, 8, 8))


def test_jacobian_spectrum_recovers_a_designed_linear_spectrum():
    """A linear decoder has a Jacobian equal to its matrix, so the answer is known exactly."""
    torch.manual_seed(0)
    n_latent, n_pixels, rank = 64, 256, 16
    u, _ = torch.linalg.qr(torch.randn(n_pixels, rank))
    v, _ = torch.linalg.qr(torch.randn(n_latent, rank))
    singular = torch.linspace(10.0, 1.0, rank)
    a = u @ torch.diag(singular) @ v.T

    def decode(z):
        return (a @ z.flatten()).view(1, 1, 16, 16)

    spectrum = jacobian_spectrum(decode, torch.randn(n_latent), k=8, oversample=8, power_iters=3)
    torch.testing.assert_close(spectrum.singular_values, singular[:8], rtol=2e-2, atol=2e-2)


def test_anisotropy_is_one_for_an_isotropic_decoder():
    torch.manual_seed(0)
    n_latent, rank = 32, 16
    u, _ = torch.linalg.qr(torch.randn(64, rank))
    v, _ = torch.linalg.qr(torch.randn(n_latent, rank))
    a = 3.0 * u @ v.T  # every direction in the row space has identical gain

    def decode(z):
        return (a @ z.flatten()).view(1, 1, 8, 8)

    spectrum = jacobian_spectrum(decode, torch.randn(n_latent), k=8, oversample=6, power_iters=2)
    assert abs(spectrum.anisotropy - 1.0) < 0.05, spectrum.singular_values
    assert spectrum.effective_rank() > 7.5
