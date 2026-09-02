"""Tests for the E0.3 capture mechanism.

These do not need CSpD or a GPU: the capture works by rebinding a module attribute, so a stand-in
module with the same shape exercises the same code path. What matters is that the swap records the
right tensors, delegates faithfully, and restores the original -- a capture that silently recorded
nothing would produce an empty sweep, and one that failed to restore would leak into later runs.
"""

import types

import pytest
import torch

from herald.models.cspd_adapter import AcceptanceCapture, capturing_processor


class FakeProcessor:
    """Same constructor signature as CSpD's DistributionProcessor."""

    def __init__(self, x1_mean, x1_std, x1_log_var, x2_mean, x2_std, x2_log_var, dim, temperature=42.0):
        self.args = (x1_mean, x2_mean, dim, temperature)

    def speculative_sample(self, x):
        return x * 2


@pytest.fixture
def fake_module():
    mod = types.ModuleType("fake_runner")
    mod.DistributionProcessor = FakeProcessor
    return mod


def _run(mod, capture, n=8, d=16, steps=3, log_var=None):
    for _ in range(steps):
        proc = mod.DistributionProcessor(
            torch.zeros(n, d), torch.ones(n, d), torch.zeros(n, d) if log_var is None else log_var,
            torch.ones(n, d), torch.ones(n, d), torch.zeros(n, d) if log_var is None else log_var,
            d, 42.0,
        )
        proc.speculative_sample(torch.randn(n, d))


def test_capture_records_every_verification_step(fake_module):
    capture = AcceptanceCapture()
    with capturing_processor(fake_module, capture):
        _run(fake_module, capture, n=8, steps=3)

    assert len(capture) == 3
    assert capture.n_tokens == 24
    assert capture.dim == 16
    stacked = capture.concat()
    assert stacked["x"].shape == (24, 16)
    assert stacked["draft_mean"].shape == (24, 16)


def test_capture_restores_the_original_class(fake_module):
    capture = AcceptanceCapture()
    with capturing_processor(fake_module, capture):
        assert fake_module.DistributionProcessor is not FakeProcessor
    assert fake_module.DistributionProcessor is FakeProcessor


def test_capture_restores_even_when_the_run_raises(fake_module):
    capture = AcceptanceCapture()
    with pytest.raises(RuntimeError), capturing_processor(fake_module, capture):
        raise RuntimeError("generation blew up")
    assert fake_module.DistributionProcessor is FakeProcessor


def test_recording_subclass_still_delegates(fake_module):
    """The capture must observe, not alter: the sampled output has to match upstream's."""
    capture = AcceptanceCapture()
    x = torch.randn(4, 16)
    with capturing_processor(fake_module, capture):
        proc = fake_module.DistributionProcessor(
            torch.zeros(4, 16), torch.ones(4, 16), torch.zeros(4, 16),
            torch.zeros(4, 16), torch.ones(4, 16), torch.zeros(4, 16), 16, 42.0,
        )
        out = proc.speculative_sample(x)
    torch.testing.assert_close(out, x * 2)


def test_scalar_log_var_is_broadcast_not_dropped(fake_module):
    """manager.log_var starts life as the float 0.0; recording it as a scalar would corrupt shapes."""
    capture = AcceptanceCapture()
    with capturing_processor(fake_module, capture):
        _run(fake_module, capture, n=4, steps=1, log_var=0.0)
    stacked = capture.concat()
    assert stacked["draft_log_sigma_accum"].shape == (4, 16)
    assert torch.all(stacked["draft_log_sigma_accum"] == 0)


def test_empty_capture_fails_loudly_rather_than_returning_nothing():
    capture = AcceptanceCapture()
    with pytest.raises(RuntimeError, match="never reached a verification step"):
        capture.concat()
