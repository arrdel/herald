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

The headline metric is **excess NLL under the target**: how many nats less likely the emitted tokens
are under ``p`` than genuine ``p`` samples. Zero means the relaxation cost nothing; larger means the
rule is waving through tokens the target considers unlikely. Paired with the acceptance rate it
gives the acceptance-fidelity frontier that E1.2b and E1.3 have to beat.
"""

from __future__ import annotations

import torch
from torch.distributions import MixtureSameFamily
from torch.utils.data import DataLoader

from herald.models.ts_mar import TimeSeriesMAR, random_masking
from herald.verifiers.projected import project_mixture


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
) -> dict:
    """Acceptance *and* the fidelity cost of getting it, for one acceptance rule.

    Args:
        projection: ``(k, d)`` statistic to verify, or ``None`` for exact token-space verification.
            ``None`` is the losslessness anchor: it must return ``excess_nll ≈ 0``.

    Returns ``acceptance`` and ``excess_nll`` (nats/token, the quantity to minimise at matched
    acceptance), plus ``excess_nll_sem`` so the two can be compared honestly, and
    ``residual_convergence`` as a health check on the rejection sampler.
    """
    draft.eval()
    target.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    out_nll, ref_nll, accepts, converged = [], [], [], []
    devices = [torch.device(device)] if torch.device(device).type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        gen = torch.Generator(device=device).manual_seed(seed)

        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            tokens = batch.to(device)
            mask = random_masking(
                tokens.shape[0], tokens.shape[1], torch.device(device), generator=gen
            )
            if not mask.any():
                continue

            q = draft.token_distribution(draft.forward_predictor(tokens, mask))
            p = target.token_distribution(target.forward_predictor(tokens, mask))

            x = q.sample()
            if projection is None:
                log_ratio = p.log_prob(x) - q.log_prob(x)
            else:
                s = x @ projection.T
                log_ratio = (project_mixture(p, projection).log_prob(s)
                             - project_mixture(q, projection).log_prob(s))
            alpha = torch.clamp(log_ratio.exp(), max=1.0)
            accepted = torch.rand_like(alpha) < alpha

            residual, ok = residual_sample(q, p, max_tries=max_tries, needed=~accepted)
            emitted = torch.where(accepted.unsqueeze(-1), x, residual)

            # Reference: genuine target draws, scored by the same model on the same conditioning.
            # Comparing against a fixed constant would fold the target's own entropy into the metric.
            out_nll.append((-p.log_prob(emitted))[mask].cpu())
            ref_nll.append((-p.log_prob(p.sample()))[mask].cpu())
            accepts.append(accepted[mask].float().cpu())
            # Only rejected positions ever consult the residual, so only they can fail to converge.
            converged.append(ok[mask].float().cpu())

    if not out_nll:
        raise RuntimeError("no decisions collected; dataset may be smaller than one batch")

    out_nll = torch.cat(out_nll)
    ref_nll = torch.cat(ref_nll)
    accepts = torch.cat(accepts)
    diff = out_nll - ref_nll
    return {
        "verified_dim": int(projection.shape[0]) if projection is not None else int(tokens.shape[-1]),
        "acceptance": float(accepts.mean()),
        "excess_nll": float(diff.mean()),
        "excess_nll_sem": float(diff.std() / max(1.0, diff.numel() ** 0.5)),
        "emitted_nll": float(out_nll.mean()),
        "target_nll": float(ref_nll.mean()),
        "residual_convergence": float(torch.cat(converged).mean()),
        "n_decisions": int(accepts.numel()),
    }
