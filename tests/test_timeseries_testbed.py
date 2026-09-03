"""Checks for the E1.x testbed: data handling, tokenizer, MAR model, training harness.

The two tests that matter most are ``test_mixture_head_is_a_normalized_density`` and the
leakage checks. The first because E1.1's whole output is an *exact* acceptance rate, which is
meaningless if the head's ``log_prob`` is not a real density; the second because a chronological
leak would inflate every number in the sweep while leaving all the plots looking reasonable.
"""

import numpy as np
import torch

from herald.data import (
    SlidingWindowDataset,
    Standardizer,
    build_bundle,
    chronological_split,
    synthetic_ar,
)
from herald.metrics import NFELedger
from herald.models import (
    MARConfig,
    TimeSeriesMAR,
    TimeSeriesTokenizer,
    TokenizerConfig,
    cosine_mask_schedule,
    random_masking,
)
from herald.train import (
    TrainConfig,
    load_checkpoint,
    mar_loss,
    save_checkpoint,
    set_seed,
    tokenizer_loss,
    train_module,
)


def _tiny_mar(d=4, n_tokens=4):
    return TimeSeriesMAR(MARConfig(latent_dim=d, n_tokens=n_tokens, width=64, depth=2, n_heads=4, n_mixture=4))


# --------------------------------------------------------------------------- data


def test_synthetic_ar_is_stable():
    series = synthetic_ar(n_steps=2048, n_channels=3, seed=0)
    assert series.shape == (2048, 3)
    assert np.isfinite(series).all()
    assert np.abs(series).max() < 1e3, "AR process diverged; companion-matrix shrinkage failed"


def test_chronological_split_preserves_order_and_covers_everything():
    series = np.arange(100, dtype=np.float32).reshape(100, 1)
    train, val, test = chronological_split(series, (0.7, 0.1, 0.2))
    assert len(train) == 70 and len(val) == 10 and len(test) == 20
    np.testing.assert_array_equal(np.concatenate([train, val, test]), series)
    assert train.max() < val.min() < test.min()


def test_standardizer_uses_training_statistics_only():
    """Fitting on the full series before splitting is the classic leak; guard against it."""
    series = np.concatenate([np.zeros((70, 1)), np.full((30, 1), 100.0)]).astype(np.float32)
    bundle = build_bundle(series, window=10, fractions=(0.7, 0.1, 0.2))

    train_windows = torch.stack([bundle.train[i] for i in range(len(bundle.train))])
    assert abs(train_windows.mean().item()) < 1e-4

    # The test split is far outside the training distribution and must stay that way after
    # standardization -- if it were centred too, statistics had leaked across the boundary.
    test_windows = torch.stack([bundle.test[i] for i in range(len(bundle.test))])
    assert test_windows.mean().item() > 10


def test_standardizer_inverse_round_trips():
    x = np.random.default_rng(0).normal(3.0, 7.0, (100, 4)).astype(np.float32)
    scaler = Standardizer.fit(x)
    np.testing.assert_allclose(scaler.inverse(scaler.transform(x)), x, rtol=1e-4, atol=1e-4)


def test_sliding_windows_are_channel_first_and_contiguous_in_time():
    series = np.arange(40, dtype=np.float32).reshape(20, 2)
    ds = SlidingWindowDataset(series, window=5, stride=5)
    assert len(ds) == 4
    assert ds[0].shape == (2, 5)
    torch.testing.assert_close(ds[0][0], torch.tensor([0.0, 2.0, 4.0, 6.0, 8.0]))


# ----------------------------------------------------------------------- tokenizer


def test_tokenizer_shapes_across_the_d_sweep():
    for d in (4, 8, 16, 32):
        tok = TimeSeriesTokenizer(TokenizerConfig(n_channels=3, patch=8, latent_dim=d, hidden=32, depth=1))
        x = torch.randn(2, 3, 64)
        posterior = tok.encode(x)
        assert posterior.mean.shape == (2, 8, d)
        assert tok.decode(posterior.sample()).shape == x.shape


def test_cached_token_dataset_matches_shapes_and_stays_stochastic():
    """The cache must store posterior *parameters*, not a drawn sample.

    If it cached samples, the E1.1 models would train against a fixed deterministic dataset and
    quietly lose the posterior noise that the encode-on-access version provides.
    """
    from herald.data.timeseries import build_bundle, synthetic_ar
    from herald.train import CachedTokenDataset

    bundle = build_bundle(synthetic_ar(n_steps=400, n_channels=2), window=32, stride=8)
    tok = TimeSeriesTokenizer(TokenizerConfig(n_channels=2, patch=8, latent_dim=4, hidden=32, depth=1))
    ds = CachedTokenDataset(bundle.train, tok)

    assert len(ds) == len(bundle.train)
    assert ds[0].shape == (4, 4)
    torch.manual_seed(0)
    a = ds[0]
    torch.manual_seed(1)
    b = ds[0]
    assert not torch.allclose(a, b), "repeated access must resample the posterior"
    # ...but around the same mean, so it is the same window each time.
    draws = torch.stack([ds[0] for _ in range(64)])
    assert torch.allclose(draws.mean(0), ds.mean[0], atol=0.5)


def test_tokenizer_kl_is_nonnegative_and_zero_at_the_prior():
    from herald.models import DiagonalGaussian

    mean = torch.zeros(2, 4, 8)
    at_prior = DiagonalGaussian(mean, torch.zeros_like(mean))
    assert abs(at_prior.kl().item()) < 1e-6
    assert DiagonalGaussian(torch.randn(2, 4, 8), torch.randn(2, 4, 8)).kl().item() >= 0


def test_tokenizer_overfits_a_single_batch():
    set_seed(0)
    tok = TimeSeriesTokenizer(TokenizerConfig(n_channels=2, patch=8, latent_dim=8, hidden=32, depth=1))
    batch = torch.randn(4, 2, 32)
    opt = torch.optim.Adam(tok.parameters(), lr=3e-3)

    first = tok(batch)["recon_loss"].item()
    for _ in range(200):
        opt.zero_grad()
        out = tok(batch)
        out["loss"].backward()
        opt.step()
    assert out["recon_loss"].item() < 0.3 * first, (first, out["recon_loss"].item())


# ----------------------------------------------------------------------------- MAR


def test_cosine_schedule_terminates_and_is_non_increasing():
    counts = cosine_mask_schedule(n_tokens=16, num_iter=8)
    assert counts[-1] == 0
    assert all(a >= b for a, b in zip(counts[:-1], counts[1:], strict=True))
    assert counts[0] < 16


def test_random_masking_hides_between_seventy_and_a_hundred_percent():
    mask = random_masking(256, 32, torch.device("cpu"))
    fractions = mask.float().mean(1)
    assert (mask.sum(1) >= 1).all()
    assert fractions.min() >= 0.68 and fractions.max() <= 1.0


def test_mixture_head_is_a_normalized_density():
    """E1.1 measures exact acceptance, so the head's log_prob must integrate to one."""
    torch.manual_seed(0)
    model = TimeSeriesMAR(MARConfig(latent_dim=1, n_tokens=4, width=32, depth=2, n_heads=4, n_mixture=4))
    dist = model.token_distribution(torch.randn(1, 32))

    grid = torch.linspace(-40, 40, 20_001).view(-1, 1, 1)
    density = dist.log_prob(grid).exp().squeeze(-1)
    integral = torch.trapezoid(density, grid.squeeze()).item()
    assert abs(integral - 1.0) < 1e-3, integral


def test_masked_positions_do_not_leak_their_values():
    """z at a masked position must not depend on the hidden token, or the NLL is meaningless."""
    torch.manual_seed(0)
    model = _tiny_mar().eval()
    tokens = torch.randn(1, 4, 4)
    mask = torch.tensor([[True, False, False, False]])

    with torch.no_grad():
        z_a = model.forward_predictor(tokens, mask)
        altered = tokens.clone()
        altered[0, 0] = 99.0
        z_b = model.forward_predictor(altered, mask)
    torch.testing.assert_close(z_a, z_b)


def test_mar_learns_on_synthetic_data():
    set_seed(0)
    model = _tiny_mar(d=4, n_tokens=8)
    tokens = torch.from_numpy(synthetic_ar(n_steps=8 * 256, n_channels=4, seed=1)).view(256, 8, 4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    for _ in range(300):
        batch = tokens[torch.randint(0, 256, (32,))]
        opt.zero_grad()
        out = model.loss(batch)
        out["loss"].backward()
        opt.step()
        losses.append(out["loss"].item())

    early, late = np.mean(losses[:25]), np.mean(losses[-25:])
    assert late < early - 0.5, f"NLL did not fall: {early:.3f} -> {late:.3f}"


def test_sampling_shapes_and_nfe_split():
    torch.manual_seed(0)
    model = _tiny_mar(d=4, n_tokens=8).eval()
    ledger = NFELedger()
    tokens = model.sample_tokens(n_samples=3, num_iter=4, ledger=ledger)

    assert tokens.shape == (3, 8, 4)
    assert torch.isfinite(tokens).all()

    counts = ledger.as_dict()["token_evals"]
    assert counts["transformer"] > 0 and counts["head"] > 0
    # Every token must be emitted exactly once per sample, across all iterations.
    assert counts["head"] == 3 * 8


# ------------------------------------------------------------------------ training


def test_checkpoint_round_trip_preserves_configuration_and_outputs():
    import tempfile
    from pathlib import Path

    torch.manual_seed(0)
    model = _tiny_mar(d=8, n_tokens=4).eval()
    z = torch.randn(2, 4, 8)
    with torch.no_grad():
        before = model.forward_predictor(z, torch.zeros(2, 4, dtype=torch.bool))

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mar.pt"
        save_checkpoint(model, TrainConfig(steps=1, out_dir=Path(tmp)), path)
        restored = load_checkpoint(path).eval()

    assert restored.config.latent_dim == 8
    with torch.no_grad():
        torch.testing.assert_close(restored.forward_predictor(z, torch.zeros(2, 4, dtype=torch.bool)), before)


def test_train_module_runs_both_losses_end_to_end():
    series = synthetic_ar(n_steps=1200, n_channels=2, seed=0)
    bundle = build_bundle(series, window=32, stride=4)
    config = TrainConfig(steps=12, batch_size=8, num_workers=0, device="cpu", amp=False, log_every=4)

    tok = TimeSeriesTokenizer(TokenizerConfig(n_channels=2, patch=8, latent_dim=4, hidden=32, depth=1))
    train_module(tok, bundle.train, tokenizer_loss, config, tag="tokenizer")
    assert any(entry["split"] == "train" for entry in config.history)

    from herald.train import TokenizedDataset

    token_ds = TokenizedDataset(bundle.train, tok)
    assert token_ds[0].shape == (4, 4)
    train_module(_tiny_mar(d=4, n_tokens=4), token_ds, mar_loss, config, tag="mar")


def test_distillation_loss_follows_the_batch_device():
    """Regression: the masking generator was built on CPU and crashed on CUDA batches.

    torch requires a generator's device to match the tensors it draws, so the generator cannot be
    created until a batch arrives. This asserts the device is taken from the batch rather than fixed
    at construction; it is the CPU half of a failure that only appeared on GPU.
    """
    from herald.train.timeseries import distillation_loss

    target, student = _tiny_mar(d=4, n_tokens=4), _tiny_mar(d=4, n_tokens=4)
    loss_fn = distillation_loss(target, seed=0)
    batch = torch.randn(8, 4, 4)

    out = loss_fn(student, batch)
    assert out["loss"].requires_grad and torch.isfinite(out["loss"])
    # The frozen target must not accumulate gradients through the distillation objective.
    out["loss"].backward()
    assert all(p.grad is None for p in target.parameters())
