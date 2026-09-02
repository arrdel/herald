"""A patch VAE for time series -- the 1-D analogue of the KL16 image tokenizer.

The image pipeline is fixed at ``d=16`` because that is what KL16 emits, which is exactly why
ImageNet cannot produce the acceptance-vs-``d`` curve C1 needs. Here the latent dimension is a
constructor argument, so E1.1 can retrain the tokenizer at each ``d`` in the sweep and hold
everything else fixed.

The interface deliberately mirrors ``diffusers.AutoencoderKL`` (``encode`` returning a diagonal
Gaussian, ``decode`` taking latents) so that code written against the image VAE ports over with
no changes. The E0.5 Jacobian machinery in ``herald.metrics.decoder_sensitivity`` works on this
decoder too, which makes it a cheap way to validate that code path before pointing it at KL16.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TokenizerConfig:
    n_channels: int
    patch: int = 8
    latent_dim: int = 16
    hidden: int = 128
    depth: int = 3
    kl_weight: float = 1e-6

    def n_tokens(self, window: int) -> int:
        if window % self.patch:
            raise ValueError(f"window {window} is not divisible by patch {self.patch}")
        return window // self.patch


class DiagonalGaussian:
    """Factorized Gaussian posterior over tokens, shaped ``(B, T, d)``.

    ``logvar`` is clamped because early in training the encoder happily drives it to -60 on
    easy patches, which makes the KL term explode and the reconstruction collapse onto a few
    dimensions -- the standard latent-collapse failure, and a silent one.
    """

    def __init__(self, mean: torch.Tensor, logvar: torch.Tensor):
        self.mean = mean
        self.logvar = logvar.clamp(-30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)

    @property
    def mode(self) -> torch.Tensor:
        return self.mean

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        noise = torch.randn(self.mean.shape, device=self.mean.device, dtype=self.mean.dtype, generator=generator)
        return self.mean + self.std * noise

    def kl(self) -> torch.Tensor:
        """KL to a standard normal, summed over latent dims and averaged over tokens."""
        return 0.5 * (self.mean**2 + self.logvar.exp() - 1.0 - self.logvar).sum(-1).mean()


class _ResBlock1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TimeSeriesTokenizer(nn.Module):
    """Patch autoencoder mapping ``(B, C, L)`` windows to ``(B, L/patch, d)`` tokens."""

    def __init__(self, config: TokenizerConfig):
        super().__init__()
        self.config = config
        c, h, d = config.n_channels, config.hidden, config.latent_dim

        self.encoder = nn.Sequential(
            nn.Conv1d(c, h, config.patch, stride=config.patch),
            *[_ResBlock1d(h) for _ in range(config.depth)],
            nn.GroupNorm(min(8, h), h),
            nn.SiLU(),
            nn.Conv1d(h, 2 * d, 1),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(d, h, 1),
            *[_ResBlock1d(h) for _ in range(config.depth)],
            nn.GroupNorm(min(8, h), h),
            nn.SiLU(),
            nn.ConvTranspose1d(h, c, config.patch, stride=config.patch),
        )

    def encode(self, x: torch.Tensor) -> DiagonalGaussian:
        mean, logvar = self.encoder(x).chunk(2, dim=1)
        return DiagonalGaussian(mean.transpose(1, 2), logvar.transpose(1, 2))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """``(B, T, d)`` tokens back to a ``(B, C, L)`` window."""
        return self.decoder(z.transpose(1, 2))

    def forward(self, x: torch.Tensor, sample: bool = True) -> dict[str, torch.Tensor]:
        posterior = self.encode(x)
        z = posterior.sample() if sample else posterior.mode
        recon = self.decode(z)
        recon_loss = F.mse_loss(recon, x)
        kl = posterior.kl()
        return {
            "recon": recon,
            "z": z,
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": recon_loss + self.config.kl_weight * kl,
        }
