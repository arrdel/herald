"""What a relaxed acceptance rule *costs*, as opposed to what it buys.

E1.2a established that acceptance is a function of the verified dimension ``k`` essentially alone: a
random projection reaches the same rate as any structured one at matched ``k``. So acceptance can no
longer distinguish the rungs of the E1.2/E1.3 ladder, and the only axis left is fidelity -- at
matched ``k``, what does each rule let through that the target would not have emitted?

The measurement here runs one full speculative step and compares the *emitted* token distribution
``p̃`` against the target ``p``:

- accept the draft's proposal ``x ~ q`` with the rule's probability ``α(x)``;
- on rejection, draw from the residual ``r ∝ (p - q)₊`` exactly as standard speculative sampling
  does, by rejection sampling with ``p`` as the proposal.

That second step is what makes the measurement trustworthy. Under the **exact** rule
``α = min(1, p/q)`` this scheme emits ``p`` exactly -- ``q·min(1, p/q) + (1-Ā)·(p-q)₊/Z = p`` -- so
the excess NLL below must come out at zero, and it is used as the anchor. Had rejection simply
resampled from ``p`` (which is what a naive implementation does, and what is tempting because it is
one line) even exact verification would score as lossy, and every number here would be measuring the
harness rather than the rule.

Two costs are reported, and they are not interchangeable.

**Excess NLL under the target** asks how many nats less likely the emitted tokens are under ``p``
than genuine ``p`` samples. It is the right metric for the question "did we still sample from the
model?", and it is the one the exact rule anchors at zero.

**Signal-domain energy distance** decodes the emitted tokens and asks whether the *distribution* of
reconstructed waveforms still matches the target's. For samples ``X, X' ~ p̃`` and ``Y, Y' ~ p`` the
statistic is ``2E‖X-Y‖ - E‖X-X'‖ - E‖Y-Y'‖``, which is non-negative and zero exactly when the two
distributions agree, so the exact rule anchors it at zero alongside the excess NLL.

It would be simpler to decode the emitted tokens and report squared error against the real window,
and that is what a first attempt did. It is a trap. The draft is the smaller model and its samples
sit closer to the conditional mean, so emitting *its* token beats a genuine target draw on squared
error: measured that way the rules come out with **negative** excess distortion, and the rule that
verifies the least important directions looks the best of all. Squared error against one ground
truth rewards variance shrinkage, and a rule that always emitted the conditional mean would win it
outright while being a useless sampler. ``excess_mse`` is still reported, but as the diagnostic that
demonstrates the failure, not as a score. The energy distance is the metric, because shrinking the
predictive variance increases it.
"""

from __future__ import annotations

import torch
from torch.distributions import MixtureSameFamily
from torch.utils.data import DataLoader, Dataset

from herald.models.ts_mar import TimeSeriesMAR, random_masking
from herald.models.ts_tokenizer import TimeSeriesTokenizer
from herald.verifiers.projected import project_mixture


class TokensWithSignal(Dataset):
    """Pairs each cached token grid with the window it was encoded from.

    :class:`~herald.train.timeseries.CachedTokenDataset` encodes ``base`` in order with no shuffling,
    so index ``i`` refers to the same window in both -- which is what lets the signal-domain metric
    score a generated token against the ground truth it was supposed to reproduce, rather than
    against another sample from the model.
    """

    def __init__(self, tokens: Dataset, base: Dataset):
        if len(tokens) != len(base):
            raise ValueError(f"length mismatch: {len(tokens)} tokens vs {len(base)} windows")
        self.tokens = tokens
        self.base = base

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int):
        return self.tokens[idx], self.base[idx]


def residual_sample(
    draft: MixtureSameFamily,
    target: MixtureSameFamily,
    max_tries: int = 100,
    needed: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample from ``r ∝ (p - q)₊`` by rejection, proposing from ``p``.

    Proposing ``y ~ p`` and accepting with probability ``(1 - q(y)/p(y))₊`` targets ``(p - q)₊``
    normalized, which is the residual standard speculative sampling draws from after a rejection.

    Args:
        needed: optional mask of positions that actually require a residual draw. Positions whose
            draft proposal was accepted never use this value, and at the high acceptance rates the
            projected rules reach they are the large majority -- marking them done up front is what
            keeps the rejection loop from dominating the run.

    Returns the samples and a mask of which ones converged. Tokens that never accepted keep their
    last proposal, a draw from ``p`` -- a small bias, but reported rather than hidden, and the same
    fallback CSpD's ``sample_distribution`` uses.
    """
    sample = target.sample()
    done = torch.zeros(sample.shape[:-1], dtype=torch.bool, device=sample.device)
    if needed is not None:
        done = done | ~needed

    for _ in range(max_tries):
        if done.all():
            break
        proposal = target.sample()
        log_q = draft.log_prob(proposal)
        log_p = target.log_prob(proposal)
        alpha = (1.0 - (log_q - log_p).exp()).clamp_min(0.0)
        take = (torch.rand_like(alpha) < alpha) & ~done
        sample = torch.where(take.unsqueeze(-1), proposal, sample)
        done = done | take

    return sample, done


@torch.no_grad()
def measure_fidelity(
    draft: TimeSeriesMAR,
    target: TimeSeriesMAR,
    dataset,
    device: str,
    projection: torch.Tensor | None = None,
    batch_size: int = 64,
    max_batches: int = 40,
    seed: int = 0,
    max_tries: int = 100,
    tokenizer: TimeSeriesTokenizer | None = None,
) -> dict:
    """Acceptance *and* the fidelity cost of getting it, for one acceptance rule.

    Args:
        projection: ``(k, d)`` statistic to verify, or ``None`` for exact token-space verification.
            ``None`` is the losslessness anchor: it must return ``excess_nll ≈ 0``.
        tokenizer: if given, the signal-domain cost is measured too and ``dataset`` must yield
            ``(tokens, window)`` pairs -- see :class:`TokensWithSignal`. The decoder is only ever
            read, never trained, so this adds a forward pass and no other coupling.

    Returns ``acceptance`` and ``excess_nll`` (nats/token, the quantity to minimise at matched
    acceptance), plus ``excess_nll_sem`` so the two can be compared honestly, and
    ``residual_convergence`` as a health check on the rejection sampler. With a ``tokenizer`` it also
    returns ``excess_mse`` and ``excess_mse_sem``: the extra squared error the emitted tokens incur
    against the *real* window, over and above what a genuine target draw incurs.
    """
    draft.eval()
    target.eval()
    if tokenizer is not None:
        tokenizer.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    out_nll, ref_nll, accepts, converged = [], [], [], []
    out_mse, ref_mse, energy = [], [], []
    devices = [torch.device(device)] if torch.device(device).type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        gen = torch.Generator(device=device).manual_seed(seed)

        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            if tokenizer is None:
                tokens, window = batch.to(device), None
            else:
                tokens, window = batch[0].to(device), batch[1].to(device)
            mask = random_masking(
                tokens.shape[0], tokens.shape[1], torch.device(device), generator=gen
            )
            if not mask.any():
                continue

            q = draft.token_distribution(draft.forward_predictor(tokens, mask))
            p = target.token_distribution(target.forward_predictor(tokens, mask))

            emitted, accepted, ok = _emit(q, p, projection, max_tries)
            reference = p.sample()

            # Reference: genuine target draws, scored by the same model on the same conditioning.
            # Comparing against a fixed constant would fold the target's own entropy into the metric.
            out_nll.append((-p.log_prob(emitted))[mask].cpu())
            ref_nll.append((-p.log_prob(reference))[mask].cpu())
            accepts.append(accepted[mask].float().cpu())
            # Only rejected positions ever consult the residual, so only they can fail to converge.
            converged.append(ok[mask].float().cpu())

            if tokenizer is not None:
                # The energy distance needs a second independent draw from each side; they must come
                # from the same conditioning, so both are generated here rather than across batches.
                emitted2, _, _ = _emit(q, p, projection, max_tries)
                # Splice each candidate into the real grid at the masked positions only, so the
                # comparison isolates what the acceptance rule emitted from what it was handed.
                fill = mask.unsqueeze(-1)
                x1, x2, y1, y2 = (
                    _decode_spliced(tokenizer, tokens, fill, z)
                    for z in (emitted, emitted2, reference, p.sample())
                )
                energy.append(energy_distance(x1, x2, y1, y2).cpu())
                out_mse.append((x1 - window.flatten(1)).pow(2).mean(-1).cpu())
                ref_mse.append((y1 - window.flatten(1)).pow(2).mean(-1).cpu())

    if not out_nll:
        raise RuntimeError("no decisions collected; dataset may be smaller than one batch")

    out_nll = torch.cat(out_nll)
    ref_nll = torch.cat(ref_nll)
    accepts = torch.cat(accepts)
    diff = out_nll - ref_nll
    result = {
        "verified_dim": int(projection.shape[0]) if projection is not None else int(tokens.shape[-1]),
        "acceptance": float(accepts.mean()),
        "excess_nll": float(diff.mean()),
        "excess_nll_sem": float(diff.std() / max(1.0, diff.numel() ** 0.5)),
        "emitted_nll": float(out_nll.mean()),
        "target_nll": float(ref_nll.mean()),
        "residual_convergence": float(torch.cat(converged).mean()),
        "n_decisions": int(accepts.numel()),
    }
    if energy:
        ed = torch.cat(energy)
        result["signal_energy"] = float(ed.mean())
        result["signal_energy_sem"] = float(ed.std() / max(1.0, ed.numel() ** 0.5))
        # Paired per window: both candidates saw the same conditioning and the same mask, so the
        # difference cancels the window's intrinsic reconstruction error. Reported as the diagnostic
        # that squared error is gameable, not as a score -- see the module docstring.
        gap = torch.cat(out_mse) - torch.cat(ref_mse)
        result["excess_mse"] = float(gap.mean())
        result["excess_mse_sem"] = float(gap.std() / max(1.0, gap.numel() ** 0.5))
    return result


def energy_distance(
    x: torch.Tensor, x2: torch.Tensor, y: torch.Tensor, y2: torch.Tensor
) -> torch.Tensor:
    """``2E‖X-Y‖ - E‖X-X'‖ - E‖Y-Y'‖`` per row, for paired independent draws.

    Zero in expectation exactly when the two distributions agree, and positive otherwise. Unlike
    squared error against a single reference it cannot be reduced by shrinking the predictive
    variance: the ``-E‖X-X'‖`` term rewards spread, so a rule that collapses towards the conditional
    mean is penalised rather than flattered.

    Args:
        x, x2: two independent draws from the emitted distribution.
        y, y2: two independent draws from the target, under the same conditioning.
    """
    return 2 * (x - y).norm(dim=-1) - (x - x2).norm(dim=-1) - (y - y2).norm(dim=-1)


def _decode_spliced(
    tokenizer: TimeSeriesTokenizer,
    tokens: torch.Tensor,
    fill: torch.Tensor,
    candidate: torch.Tensor,
) -> torch.Tensor:
    """Decode the real grid with ``candidate`` substituted at the masked positions, flattened."""
    return tokenizer.decode(torch.where(fill, candidate, tokens)).flatten(1)


def _emit(
    draft: MixtureSameFamily,
    target: MixtureSameFamily,
    projection: torch.Tensor | None,
    max_tries: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One full speculative step: propose from ``q``, verify, and fall back to the residual."""
    x = draft.sample()
    if projection is None:
        log_ratio = target.log_prob(x) - draft.log_prob(x)
    else:
        s = x @ projection.T
        log_ratio = (project_mixture(target, projection).log_prob(s)
                     - project_mixture(draft, projection).log_prob(s))
    accepted = torch.rand_like(log_ratio) < torch.clamp(log_ratio.exp(), max=1.0)
    residual, ok = residual_sample(draft, target, max_tries=max_tries, needed=~accepted)
    return torch.where(accepted.unsqueeze(-1), x, residual), accepted, ok
