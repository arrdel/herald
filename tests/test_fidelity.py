"""Tests for the fidelity side of the acceptance/fidelity trade.

The anchor is losslessness: with exact token-space verification and a correct residual sampler, the
emitted distribution *is* the target, so ``excess_nll`` must be zero to within Monte Carlo error.
Everything the E1.2/E1.3 ladder will claim is a comparison against that zero, so if the harness
scores exact verification as lossy then every later number is measuring the harness.
"""

import pytest
import torch
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal
from torch.utils.data import Dataset

from herald.eval import measure_fidelity, residual_sample
from herald.eval.fidelity import TokensWithSignal, energy_distance
from herald.models.ts_mar import MARConfig, TimeSeriesMAR
from herald.models.ts_tokenizer import TimeSeriesTokenizer, TokenizerConfig
from herald.verifiers import random_projection


class _Tokens(Dataset):
    def __init__(self, n=512, t=8, d=8, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, t, d, generator=g)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i]


def _mar(d=8, t=8, seed=0):
    torch.manual_seed(seed)
    return TimeSeriesMAR(MARConfig(latent_dim=d, n_tokens=t, width=32, depth=2, n_heads=4,
                                   n_mixture=2))


def _mixture(n, d, seed, shift=0.0):
    g = torch.Generator().manual_seed(seed)
    return MixtureSameFamily(
        Categorical(logits=torch.randn(n, 2, generator=g)),
        Independent(Normal(torch.randn(n, 2, d, generator=g) + shift,
                           torch.rand(n, 2, d, generator=g) * 0.4 + 0.6), 1),
    )


def test_exact_verification_is_lossless():
    """The anchor. Exact rule + residual sampler emits p, so the excess NLL is zero."""
    draft, target = _mar(seed=0), _mar(seed=1)
    out = measure_fidelity(draft, target, _Tokens(), device="cpu", batch_size=64, max_batches=8)
    assert abs(out["excess_nll"]) < 4.0 * max(out["excess_nll_sem"], 1e-6)
    assert out["verified_dim"] == 8


def test_projected_verification_is_lossy_and_worsens_as_k_falls():
    """The trade the ladder is about: more acceptance, less fidelity."""
    draft, target = _mar(seed=0), _mar(seed=1)
    kw = {"device": "cpu", "batch_size": 64, "max_batches": 8}
    exact = measure_fidelity(draft, target, _Tokens(), **kw)

    gen = torch.Generator().manual_seed(0)
    coarse = measure_fidelity(draft, target, _Tokens(),
                              projection=random_projection(1, 8, generator=gen), **kw)
    fine = measure_fidelity(draft, target, _Tokens(),
                            projection=random_projection(4, 8, generator=gen), **kw)

    assert coarse["acceptance"] > fine["acceptance"] > exact["acceptance"]
    assert coarse["excess_nll"] > exact["excess_nll"]
    assert coarse["verified_dim"] == 1


def test_residual_sampler_targets_the_positive_part():
    """(p - q)_+ puts no mass where q dominates p, which is the property the anchor relies on."""
    n, d = 20_000, 2
    q = _mixture(n, d, seed=0)
    p = _mixture(n, d, seed=1, shift=1.5)
    torch.manual_seed(0)
    sample, done = residual_sample(q, p, max_tries=200)

    assert done.float().mean() > 0.9
    ok = done
    # Residual samples must be likelier under p than under q; that is what (p - q)_+ means.
    assert (p.log_prob(sample)[ok] > q.log_prob(sample)[ok]).float().mean() > 0.95


def test_residual_convergence_is_reported():
    draft, target = _mar(seed=0), _mar(seed=1)
    out = measure_fidelity(draft, target, _Tokens(), device="cpu", batch_size=64, max_batches=4)
    assert 0.0 <= out["residual_convergence"] <= 1.0


def test_fidelity_is_deterministic_for_a_fixed_seed():
    draft, target = _mar(seed=0), _mar(seed=1)
    kw = {"device": "cpu", "batch_size": 64, "max_batches": 4, "seed": 3}
    a = measure_fidelity(draft, target, _Tokens(), **kw)
    b = measure_fidelity(draft, target, _Tokens(), **kw)
    assert a["excess_nll"] == pytest.approx(b["excess_nll"])
    assert a["acceptance"] == pytest.approx(b["acceptance"])


def test_empty_measurement_fails_loudly():
    draft, target = _mar(seed=0), _mar(seed=1)
    with pytest.raises(RuntimeError, match="no decisions collected"):
        measure_fidelity(draft, target, _Tokens(n=4), device="cpu", batch_size=64, max_batches=4)


class _Signals(Dataset):
    """Token grids paired with the windows a tokenizer would have encoded them from."""

    def __init__(self, tokens: _Tokens, tokenizer):
        self.tokens = tokens
        with torch.no_grad():
            self.windows = tokenizer.decode(tokens.x)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        return self.tokens[i], self.windows[i]


def _tokenizer(d=8, seed=0):
    torch.manual_seed(seed)
    return TimeSeriesTokenizer(
        TokenizerConfig(n_channels=3, patch=4, latent_dim=d, hidden=16, depth=1)
    ).eval()


def _signal_kwargs(tokenizer):
    return {"device": "cpu", "batch_size": 64, "max_batches": 8, "tokenizer": tokenizer}


def test_exact_verification_is_lossless_in_signal_space_too():
    """The signal anchor. If the exact rule scored as distorting, the metric would be broken."""
    tok = _tokenizer()
    draft, target = _mar(seed=0), _mar(seed=1)
    out = measure_fidelity(draft, target, _Signals(_Tokens(), tok), **_signal_kwargs(tok))
    assert abs(out["signal_energy"]) < 4.0 * max(out["signal_energy_sem"], 1e-6)


def _draws(n, d, scale=1.0, shift=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(n, d, generator=g) * scale + shift for _ in range(2)]


def _energy_in_sems(scale=1.0, shift=0.0, n=20_000, d=4):
    """The energy distance in units of its own standard error.

    The per-row estimator has a standard deviation of order the data scale, so an absolute
    threshold is really a disguised statement about ``n``. Reporting the effect in standard errors
    keeps the assertions meaningful if the sample size ever changes.
    """
    x, x2 = _draws(n, d, scale=scale, shift=shift, seed=0)
    y, y2 = _draws(n, d, seed=1)
    ed = energy_distance(x, x2, y, y2)
    return float(ed.mean()) / float(ed.std() / n**0.5)


def test_energy_distance_vanishes_for_matching_distributions():
    assert abs(_energy_in_sems()) < 3.0


def test_energy_distance_detects_a_shifted_distribution():
    assert _energy_in_sems(shift=0.5) > 10.0


def test_energy_distance_punishes_variance_shrinkage():
    """The property squared error lacks, and the reason this metric replaced it.

    A rule that collapses towards the conditional mean wins on MSE against a single reference and
    must lose here, or the anti-control could pass itself off as the best rule in the study.
    """
    assert _energy_in_sems(scale=0.3) > 10.0


def test_signal_metrics_are_absent_without_a_tokenizer():
    """The signal pass costs four extra decodes, so it must not run when it was not asked for."""
    draft, target = _mar(seed=0), _mar(seed=1)
    out = measure_fidelity(draft, target, _Tokens(), device="cpu", batch_size=64, max_batches=4)
    assert "signal_energy" not in out


def test_pairing_a_mismatched_dataset_fails_loudly():
    tok = _tokenizer()
    with pytest.raises(ValueError, match="length mismatch"):
        TokensWithSignal(_Tokens(n=16), _Signals(_Tokens(n=8), tok))
