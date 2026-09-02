"""MAR-style masked autoregressive model over continuous time-series tokens (E1.1).

Structure follows MAR: a bidirectional transformer sees the visible tokens, mask tokens are
inserted at the unknown positions, and a second stack produces a per-position conditioning vector
``z``. A small per-token head then defines the conditional density of that token given ``z``.
Generation unmasks tokens in groups over ``num_iter`` steps on a cosine schedule.

**Why the head is a Gaussian mixture and not a diffusion head.** MAR and D-JEPA use a per-token
diffusion loss, which has no tractable density. That is fine for generation and fatal for E1.1,
whose entire output is a measurement of *exact* speculative acceptance as a function of ``d``. With
a diffusion head there is no exact acceptance rate to measure, and any surrogate would confound the
dimensional effect we are trying to isolate with the error of the density approximation -- the same
confound that makes CSpD's temperature so hard to reason about. A diagonal-Gaussian-mixture head
keeps ``log_prob`` exact while staying expressive enough that the model is not trivially
misspecified, so the acceptance-vs-``d`` curve measures dimensionality and nothing else.

This means E1.1 is deliberately *not* a faithful reproduction of MAR. It is a controlled instrument
for one measurement. The faithful comparison lives on ImageNet in Phase 2, where the real diffusion
head is in play; if the two disagree, the plan's dimensional story is what breaks, and that is worth
knowing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal


@dataclass
class MARConfig:
    latent_dim: int
    n_tokens: int
    width: int = 256
    depth: int = 6
    n_heads: int = 8
    mlp_ratio: float = 4.0
    n_mixture: int = 8
    head_hidden: int = 256
    dropout: float = 0.0
    min_logstd: float = -7.0
    max_logstd: float = 3.0


def cosine_mask_schedule(n_tokens: int, num_iter: int) -> list[int]:
    """Number of tokens still masked after each generation step.

    MAR's schedule: unmask few tokens early, when context is thin and mistakes are expensive, and
    many late. Always ends at zero so generation terminates.
    """
    counts = [
        max(1, int(n_tokens * math.cos(math.pi / 2 * (i + 1) / num_iter))) for i in range(num_iter)
    ]
    counts[-1] = 0
    return counts


def random_masking(
    batch: int, n_tokens: int, device: torch.device, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Bool mask ``(B, T)``, True where a token is hidden and must be predicted.

    The mask ratio is drawn per sample from a truncated cosine distribution over ``[0.7, 1.0]`` of
    tokens, matching MAR: training mostly on heavily masked inputs is what makes the model usable
    at the early generation steps, where almost everything is unknown.
    """
    ratios = torch.rand(batch, device=device, generator=generator) * 0.3 + 0.7
    n_masked = (ratios * n_tokens).long().clamp(1, n_tokens)
    noise = torch.rand(batch, n_tokens, device=device, generator=generator)
    rank = noise.argsort(dim=1).argsort(dim=1)
    return rank < n_masked.unsqueeze(1)


class MixtureHead(nn.Module):
    """Per-token conditional density: a diagonal Gaussian mixture conditioned on ``z``.

    Exact ``log_prob``, which is what makes E1.1's acceptance measurement well defined.
    """

    def __init__(self, config: MARConfig):
        super().__init__()
        self.config = config
        d, k = config.latent_dim, config.n_mixture
        self.net = nn.Sequential(
            nn.Linear(config.width, config.head_hidden),
            nn.SiLU(),
            nn.Linear(config.head_hidden, config.head_hidden),
            nn.SiLU(),
            nn.Linear(config.head_hidden, k * (2 * d + 1)),
        )

    def forward(self, z: torch.Tensor) -> MixtureSameFamily:
        d, k = self.config.latent_dim, self.config.n_mixture
        params = self.net(z)
        logits, mean, logstd = params.split([k, k * d, k * d], dim=-1)
        mean = mean.unflatten(-1, (k, d))
        logstd = logstd.unflatten(-1, (k, d)).clamp(self.config.min_logstd, self.config.max_logstd)
        return MixtureSameFamily(
            Categorical(logits=logits),
            Independent(Normal(mean, logstd.exp()), 1),
        )


class TimeSeriesMAR(nn.Module):
    """Masked autoregressive transformer over continuous tokens.

    ``forward_predictor`` returns the conditioning vector ``z`` for every position. That tensor is
    the object HERALD's free-drafter claim is about: it is computed once per step for all masked
    positions in a single pass, and the expensive per-token head is what consumes it.
    """

    def __init__(self, config: MARConfig):
        super().__init__()
        self.config = config
        w = config.width

        self.token_proj = nn.Linear(config.latent_dim, w)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, w))
        self.pos_embed = nn.Parameter(torch.zeros(1, config.n_tokens, w))

        layer = nn.TransformerEncoderLayer(
            d_model=w,
            nhead=config.n_heads,
            dim_feedforward=int(w * config.mlp_ratio),
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Nested tensors are unavailable with norm_first and would only help padded batches, which
        # this model never has: every sequence is exactly n_tokens long.
        self.encoder = nn.TransformerEncoder(layer, config.depth // 2, enable_nested_tensor=False)
        self.decoder = nn.TransformerEncoder(
            layer, config.depth - config.depth // 2, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(w)
        self.head = MixtureHead(config)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward_predictor(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``(B, T, d)`` tokens and ``(B, T)`` mask to ``(B, T, width)`` conditioning vectors.

        Masked positions contribute the learned mask token rather than their true value, so no
        information about the targets leaks through the bidirectional attention.
        """
        x = self.token_proj(tokens)
        x = torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(x), x)
        x = x + self.pos_embed[:, : x.shape[1]]
        return self.norm(self.decoder(self.encoder(x)))

    def token_distribution(self, z: torch.Tensor) -> MixtureSameFamily:
        return self.head(z)

    def loss(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Negative log-likelihood of the masked tokens under the model.

        Only masked positions contribute. Scoring visible positions too would let the model earn
        loss for copying its own input, which is the standard way a masked model looks like it is
        learning while learning nothing.
        """
        if mask is None:
            mask = random_masking(tokens.shape[0], tokens.shape[1], tokens.device)
        z = self.forward_predictor(tokens, mask)
        log_prob = self.token_distribution(z).log_prob(tokens)
        n_masked = mask.sum().clamp(min=1)
        nll = -(log_prob * mask).sum() / n_masked
        return {"loss": nll, "nll": nll.detach(), "mask_fraction": mask.float().mean().detach()}

    @torch.no_grad()
    def sample_tokens(
        self,
        n_samples: int,
        num_iter: int = 16,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
        ledger=None,
    ) -> torch.Tensor:
        """Generate ``(n_samples, T, d)`` tokens by iterative unmasking.

        Pass an :class:`~herald.metrics.NFELedger` to record the transformer-versus-head split.
        That split is the empirical basis for the free-drafter claim, so it is worth recording on
        every run rather than reconstructing later.
        """
        device = device or self.pos_embed.device
        t, d = self.config.n_tokens, self.config.latent_dim
        tokens = torch.zeros(n_samples, t, d, device=device)
        mask = torch.ones(n_samples, t, dtype=torch.bool, device=device)

        for remaining in cosine_mask_schedule(t, num_iter):
            if ledger is not None:
                ledger.record("transformer", n_tokens=int(mask.sum()))
            z = self.forward_predictor(tokens, mask)
            dist = self.token_distribution(z)
            sampled = dist.sample()

            # Reveal the positions the model is most confident about, which is MAR's ordering and
            # matters more here than it looks: confidence ordering is what makes later steps easy.
            confidence = dist.log_prob(sampled).masked_fill(~mask, -math.inf)
            n_reveal = int(mask[0].sum().item()) - remaining
            if n_reveal <= 0:
                continue
            reveal_idx = confidence.topk(n_reveal, dim=1).indices
            reveal = torch.zeros_like(mask).scatter_(1, reveal_idx, True) & mask

            if ledger is not None:
                ledger.record("head", n_tokens=int(reveal.sum()))
            tokens = torch.where(reveal.unsqueeze(-1), sampled, tokens)
            mask = mask & ~reveal

        return tokens
