"""Training harness for the E1.x time-series testbed.

Two stages, in order, mirroring the image pipeline: fit the tokenizer, freeze it, then fit the
masked autoregressive model on its latents. The whole sweep is run once per token dimension ``d``,
so everything here takes ``d`` from the model config rather than hardcoding it.

Determinism is not decoration in this project. The acceptance-vs-``d`` curve is the headline plot
of claim C1, and a curve assembled from runs with drifting seeds cannot distinguish a dimensional
effect from run-to-run variance. Every entry point takes a seed and records it in the checkpoint.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from herald.models.ts_mar import MARConfig, TimeSeriesMAR, random_masking
from herald.models.ts_tokenizer import TimeSeriesTokenizer, TokenizerConfig


@dataclass
class TrainConfig:
    steps: int = 20_000
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_steps: int = 500
    grad_clip: float = 1.0
    eval_every: int = 1_000
    log_every: int = 200
    seed: int = 0
    #: Save an intermediate checkpoint every N steps, as ``{tag}_step{N}.pt``. Needed whenever a
    #: quantity depends on *two* models at matched training budgets -- acceptance, for one, which
    #: cannot be evaluated from inside either model's own loop.
    checkpoint_every: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True
    num_workers: int = 4
    out_dir: Path | None = None
    history: list[dict] = field(default_factory=list, repr=False)


def set_seed(seed: int) -> None:
    """Seed every generator this project touches.

    ``cudnn.benchmark`` is left off deliberately: its autotuner picks different kernels run to run
    depending on machine load, which reintroduces nondeterminism through the back door after the
    seeds have been set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _lr_scale(step: int, config: TrainConfig) -> float:
    if step < config.warmup_steps:
        return (step + 1) / max(1, config.warmup_steps)
    progress = (step - config.warmup_steps) / max(1, config.steps - config.warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def _infinite(loader: DataLoader) -> Iterator:
    while True:
        yield from loader


def train_module(
    model: torch.nn.Module,
    dataset: Dataset,
    loss_fn: Callable[[torch.nn.Module, torch.Tensor], dict[str, torch.Tensor]],
    config: TrainConfig,
    val_dataset: Dataset | None = None,
    tag: str = "model",
) -> torch.nn.Module:
    """Generic training loop shared by the tokenizer and the autoregressive model."""
    set_seed(config.seed)
    device = torch.device(config.device)
    model = model.to(device)

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: _lr_scale(s, config))
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model.train()
    stream = _infinite(loader)
    for step in range(config.steps):
        batch = next(stream).to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            out = loss_fn(model, batch)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(out["loss"]).backward()
        if config.grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if config.log_every and step % config.log_every == 0:
            config.history.append(
                {"step": step, "split": "train", **{k: float(v) for k, v in out.items()}}
            )
        if val_dataset is not None and config.eval_every and step and step % config.eval_every == 0:
            config.history.append(
                {"step": step, "split": "val", **evaluate(model, val_dataset, loss_fn, config)}
            )
            model.train()
        if config.checkpoint_every and step and step % config.checkpoint_every == 0:
            if config.out_dir is None:
                raise ValueError("checkpoint_every requires out_dir")
            save_checkpoint(model, config, Path(config.out_dir) / f"{tag}_step{step}.pt")
            model.train()

    if config.out_dir is not None:
        save_checkpoint(model, config, Path(config.out_dir) / f"{tag}.pt")
    return model


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: Dataset,
    loss_fn: Callable[[torch.nn.Module, torch.Tensor], dict[str, torch.Tensor]],
    config: TrainConfig,
    max_batches: int = 50,
) -> dict[str, float]:
    """Mean loss over up to ``max_batches`` validation batches, at fixed batch order."""
    model.eval()
    device = torch.device(config.device)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)

    totals: dict[str, float] = {}
    n = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        out = loss_fn(model, batch.to(device))
        for k, v in out.items():
            totals[k] = totals.get(k, 0.0) + float(v)
        n += 1
    return {k: v / max(1, n) for k, v in totals.items()}


class TokenizedDataset(Dataset):
    """Windows encoded to latent tokens by a frozen tokenizer.

    Encodes on access rather than caching the whole split: the posterior is stochastic, so
    resampling each epoch is a free regularizer, and caching would silently turn it into a fixed
    deterministic dataset.
    """

    def __init__(self, base: Dataset, tokenizer: TimeSeriesTokenizer, device: str = "cpu"):
        self.base = base
        self.tokenizer = tokenizer.to(device).eval()
        self.device = device
        for p in self.tokenizer.parameters():
            p.requires_grad_(False)

    def __len__(self) -> int:
        return len(self.base)

    @torch.no_grad()
    def __getitem__(self, idx: int) -> torch.Tensor:
        window = self.base[idx].unsqueeze(0).to(self.device)
        return self.tokenizer.encode(window).sample().squeeze(0).cpu()


class CachedTokenDataset(Dataset):
    """Posterior parameters encoded once, resampled on every access.

    :class:`TokenizedDataset` runs the tokenizer per item, which costs one forward pass per window
    per epoch and dominates the E1.1 sweep -- twelve models across six token dimensions, each
    re-encoding the same windows thousands of times. Caching the *parameters* rather than a drawn
    sample keeps the posterior stochastic (the point of encoding on access) while paying the
    encoder cost exactly once.
    """

    def __init__(
        self,
        base: Dataset,
        tokenizer: TimeSeriesTokenizer,
        device: str = "cpu",
        batch_size: int = 256,
    ):
        tokenizer = tokenizer.to(device).eval()
        loader = DataLoader(base, batch_size=batch_size, shuffle=False, drop_last=False)
        means, stds = [], []
        with torch.no_grad():
            for batch in loader:
                posterior = tokenizer.encode(batch.to(device))
                means.append(posterior.mean.cpu())
                stds.append(posterior.std.cpu())
        self.mean = torch.cat(means)
        self.std = torch.cat(stds)

    def __len__(self) -> int:
        return self.mean.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.mean[idx] + self.std[idx] * torch.randn_like(self.std[idx])


def tokenizer_loss(model: TimeSeriesTokenizer, batch: torch.Tensor) -> dict[str, torch.Tensor]:
    out = model(batch)
    return {"loss": out["loss"], "recon_loss": out["recon_loss"].detach(), "kl": out["kl"].detach()}


def mar_loss(model: TimeSeriesMAR, batch: torch.Tensor) -> dict[str, torch.Tensor]:
    return model.loss(batch)


def distillation_loss(
    target: TimeSeriesMAR, seed: int = 0
) -> Callable[[TimeSeriesMAR, torch.Tensor], dict[str, torch.Tensor]]:
    """Train a draft to imitate a frozen target, rather than to fit the data.

    E1.2b's conservation result says the acceptance/fidelity frontier is fixed by how far apart the
    draft and target are: any acceptance rule only chooses a point on it. The single lever that moves
    the frontier itself is therefore to move ``q`` towards ``p``, and this is that lever.

    The direction matters. Minimising ``KL(p‖q)`` -- estimated by drawing from the target and scoring
    those draws under the draft -- is mode-*covering*, so the draft is pushed to put mass wherever the
    target does. That is exactly the failure mode speculative sampling is sensitive to, because a
    draft that misses a region of the target's support is rejected there every time. The reverse KL
    would let the draft collapse onto one mode and score well while being useless as a proposal.

    The target is frozen and used only under ``no_grad``, so the draft can never improve the loss by
    dragging the target towards itself.
    """
    target = target.eval()
    for param in target.parameters():
        param.requires_grad_(False)
    # Built lazily on first call: torch requires the generator's device to match the tensors it
    # draws, so the device cannot be known until a batch arrives.
    generators: dict[torch.device, torch.Generator] = {}

    def loss_fn(model: TimeSeriesMAR, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        if batch.device not in generators:
            generators[batch.device] = torch.Generator(device=batch.device).manual_seed(seed)
        generator = generators[batch.device]
        mask = random_masking(batch.shape[0], batch.shape[1], batch.device, generator=generator)
        with torch.no_grad():
            sample = target.token_distribution(target.forward_predictor(batch, mask)).sample()
        q = model.token_distribution(model.forward_predictor(batch, mask))
        nll = -q.log_prob(sample)[mask].mean()
        return {"loss": nll, "distill_nll": nll.detach()}

    return loss_fn


def save_checkpoint(model: torch.nn.Module, config: TrainConfig, path: Path) -> None:
    """Write weights, the model config, and the training config together.

    Bundled deliberately: a checkpoint whose token dimension has to be inferred from tensor shapes
    is a checkpoint that will eventually be loaded at the wrong ``d`` during the sweep.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    train_config = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(config).items()}
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": asdict(model.config),
            "model_class": type(model).__name__,
            "train_config": train_config,
        },
        path,
    )
    (path.parent / f"{path.stem}_history.json").write_text(json.dumps(config.history, indent=2))


def load_checkpoint(path: Path, map_location: str = "cpu") -> torch.nn.Module:
    """Rebuild a model from a checkpoint written by :func:`save_checkpoint`."""
    blob = torch.load(path, map_location=map_location, weights_only=False)
    classes = {
        "TimeSeriesTokenizer": (TimeSeriesTokenizer, TokenizerConfig),
        "TimeSeriesMAR": (TimeSeriesMAR, MARConfig),
    }
    if blob["model_class"] not in classes:
        raise ValueError(f"unknown model class {blob['model_class']!r} in {path}")
    model_cls, config_cls = classes[blob["model_class"]]
    model = model_cls(config_cls(**blob["model_config"]))
    model.load_state_dict(blob["state_dict"])
    return model
