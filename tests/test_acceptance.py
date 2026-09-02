"""Tests for the shared E1.x acceptance measurement.

This function produces the numbers E1.1, E1.1b, and the E1.2/E1.3 ladder are all compared against,
so its failure modes are expensive: a subtly wrong acceptance rate would not look wrong, it would
look like a finding. The anchor test is that a model measured against *itself* accepts with
probability exactly 1 -- the one case where the right answer is known without computing anything.
"""

import pytest
import torch

from herald.eval import DRAFT_ARCH, TARGET_ARCH, measure_acceptance
from herald.models.ts_mar import MARConfig, TimeSeriesMAR


class _Tokens(torch.utils.data.Dataset):
    """Plain (N, T, d) token tensors; the real datasets yield the same shape."""

    def __init__(self, n=256, t=8, d=4, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, t, d, generator=g)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i]


def _mar(d=4, t=8, seed=0, **arch):
    torch.manual_seed(seed)
    return TimeSeriesMAR(MARConfig(latent_dim=d, n_tokens=t, **(arch or {"width": 32, "depth": 2, "n_heads": 4, "n_mixture": 2})))


def test_a_model_against_itself_accepts_everything():
    """p and q are the same distribution, so the exact ratio is 1 at every token."""
    model = _mar()
    out = measure_acceptance(model, model, _Tokens(), device="cpu", batch_size=32, max_batches=4)
    assert out["acceptance"] == pytest.approx(1.0, abs=1e-6)
    assert out["fraction_certain_accept"] == pytest.approx(1.0)
    assert out["delta_median"] == pytest.approx(0.0, abs=1e-6)
    assert out["log_sigma_gap_median"] == pytest.approx(0.0, abs=1e-6)
    assert out["log_ratio_median"] == pytest.approx(0.0, abs=1e-6)


def test_different_models_accept_less_than_everything():
    draft = _mar(seed=0)
    target = _mar(seed=1)
    out = measure_acceptance(draft, target, _Tokens(), device="cpu", batch_size=32, max_batches=4)
    assert 0.0 <= out["acceptance"] < 1.0
    assert out["delta_median"] > 0.0
    assert out["n_decisions"] > 0


def test_measurement_is_deterministic_for_a_fixed_seed():
    """The d-sweep is assembled from separate runs; drifting randomness would masquerade as signal."""
    draft, target = _mar(seed=0), _mar(seed=1)
    kw = {"device": "cpu", "batch_size": 32, "max_batches": 4, "seed": 7}
    a = measure_acceptance(draft, target, _Tokens(), **kw)
    b = measure_acceptance(draft, target, _Tokens(), **kw)
    assert a["acceptance"] == pytest.approx(b["acceptance"])
    assert a["delta_median"] == pytest.approx(b["delta_median"])


def test_acceptance_probabilities_stay_in_range():
    draft, target = _mar(seed=0), _mar(seed=2)
    out = measure_acceptance(draft, target, _Tokens(), device="cpu", batch_size=32, max_batches=4)
    assert 0.0 <= out["acceptance"] <= 1.0
    assert 0.0 <= out["fraction_certain_accept"] <= 1.0


def test_empty_measurement_fails_loudly():
    """drop_last=True means a dataset below one batch yields nothing; silence would be worse."""
    draft, target = _mar(seed=0), _mar(seed=1)
    tiny = _Tokens(n=4)
    with pytest.raises(RuntimeError, match="no acceptance decisions"):
        measure_acceptance(draft, target, tiny, device="cpu", batch_size=64, max_batches=4)


def test_shared_arch_constants_keep_target_larger_than_draft():
    """The whole testbed presumes a capacity gap; a silent edit that removed it would void E1.1."""
    assert TARGET_ARCH["width"] > DRAFT_ARCH["width"]
    assert TARGET_ARCH["depth"] > DRAFT_ARCH["depth"]

    t = TimeSeriesMAR(MARConfig(latent_dim=8, n_tokens=8, **TARGET_ARCH))
    d = TimeSeriesMAR(MARConfig(latent_dim=8, n_tokens=8, **DRAFT_ARCH))
    assert sum(p.numel() for p in t.parameters()) > sum(p.numel() for p in d.parameters())
