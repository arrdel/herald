"""Checks for NFE accounting.

The per-component split is the empirical basis for the free-drafter claim -- "the transformer is
cheap, the denoising head is everything" -- so an instrumentation bug here does not produce an
obviously wrong number, it produces a plausible wrong number. Both known failure modes are pinned:
the two forks' differing method names, and the denoising steps hidden inside one ``sample`` call.
"""

import pytest
import torch

from herald.metrics import NFELedger, instrument_mar_sampler
from herald.metrics.nfe import diffusion_steps_per_call


class _FakeDiffusion:
    def __init__(self, steps):
        self.num_timesteps = steps


class _FakeDiffLoss:
    def __init__(self, steps=100):
        self.gen_diffusion = _FakeDiffusion(steps)

    def sample(self, z, *args, **kwargs):
        return torch.zeros_like(z)


class _FakeMAR:
    """MAR naming: forward_mae_encoder / forward_mae_decoder."""

    def __init__(self, steps=100):
        self.diffloss = _FakeDiffLoss(steps)

    def forward_mae_encoder(self, x, *args):
        return x

    def forward_mae_decoder(self, x, *args):
        return x


class _FakeDJEPA:
    """D-JEPA naming: forward_encoder / forward_decoder."""

    def __init__(self, steps=50):
        self.diffloss = _FakeDiffLoss(steps)

    def forward_encoder(self, x, *args):
        return x

    def forward_decoder(self, x, *args):
        return x


@pytest.mark.parametrize("model_cls,steps", [(_FakeMAR, 100), (_FakeDJEPA, 50)])
def test_both_fork_naming_conventions_are_instrumented(model_cls, steps):
    model = model_cls(steps)
    ledger = NFELedger()
    restore = instrument_mar_sampler(model, ledger)

    x = torch.zeros(4, 8)
    model.forward_mae_encoder(x) if hasattr(model, "forward_mae_encoder") else model.forward_encoder(x)
    model.forward_mae_decoder(x) if hasattr(model, "forward_mae_decoder") else model.forward_decoder(x)
    model.diffloss.sample(torch.zeros(7, 16))

    counts = ledger.as_dict()["token_evals"]
    assert counts["encoder"] == 4
    assert counts["decoder"] == 4
    assert counts["diffusion_head"] == 7 * steps
    restore()


def test_sequence_inputs_are_counted_as_tokens_not_rows():
    """Both components must be in the same units or their ratio means nothing."""
    model = _FakeMAR(steps=100)
    ledger = NFELedger()
    restore = instrument_mar_sampler(model, ledger)

    model.forward_mae_encoder(torch.zeros(4, 320, 768))  # batch 4, 320 positions
    restore()
    assert ledger.as_dict()["token_evals"]["encoder"] == 4 * 320


def test_realistic_mar_step_accounting():
    """One MAR sampling run, counted the way E0.1 reports it.

    Both stages come out within an order of magnitude in token evaluations, which is the point:
    the head's dominance of *wall time* is not explained by raw evaluation count. It comes from
    those evaluations being 100 sequential small-MLP calls rather than one batched matmul.
    """
    model = _FakeMAR(steps=100)
    ledger = NFELedger()
    restore = instrument_mar_sampler(model, ledger)

    seq_len, buffer, num_iter, bsz = 256, 64, 64, 2
    for _ in range(num_iter):
        model.forward_mae_encoder(torch.zeros(bsz, seq_len + buffer, 768))
        model.forward_mae_decoder(torch.zeros(bsz, seq_len + buffer, 768))
        model.diffloss.sample(torch.zeros(bsz * seq_len // num_iter, 16))
    restore()

    counts = ledger.as_dict()["token_evals"]
    transformer = counts["encoder"] + counts["decoder"]
    head = counts["diffusion_head"]
    assert head == bsz * seq_len * 100
    assert transformer == 2 * num_iter * bsz * (seq_len + buffer)
    assert 0.1 < head / transformer < 10, (head, transformer)


def test_restore_unpatches_everything():
    """Bound methods are recreated on each access, so compare the underlying function."""
    model = _FakeMAR()
    original = model.forward_mae_encoder.__func__

    restore = instrument_mar_sampler(model, NFELedger())
    # The wrapper is stored as a plain function on the instance, so it has no __func__ at all.
    assert getattr(model.forward_mae_encoder, "__func__", None) is not original
    restore()
    assert model.forward_mae_encoder.__func__ is original
    assert "forward_mae_encoder" not in vars(model), "class method should not be shadowed after restore"


def test_repeated_instrument_restore_cycles_do_not_double_count():
    model = _FakeMAR(steps=10)
    for _ in range(3):
        ledger = NFELedger()
        restore = instrument_mar_sampler(model, ledger)
        model.forward_mae_encoder(torch.zeros(2, 4))
        model.diffloss.sample(torch.zeros(2, 4))
        restore()
        counts = ledger.as_dict()
        assert counts["calls"]["encoder"] == 1
        assert counts["token_evals"]["diffusion_head"] == 2 * 10


def test_refuses_to_guess_the_denoising_step_count():
    """Silently defaulting to 1 would understate the head by ~100x; fail loudly instead."""

    class _NoDiffusion:
        gen_diffusion = None

    with pytest.raises(ValueError, match="100x too small"):
        diffusion_steps_per_call(_NoDiffusion())
