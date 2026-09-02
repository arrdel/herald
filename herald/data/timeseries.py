"""Time-series data for the E1.x testbed.

This is the secondary testbed and the only place in the project that trains from scratch. It earns
its keep for one reason: token dimension ``d`` is a free parameter here, so the acceptance-vs-``d``
curve that claim C1 rests on can be measured across a real sweep. ImageNet gives two or three
points on that curve at most, because ``d`` is fixed by whichever VAE the checkpoint was trained
with.

Datasets follow the standard Informer/Autoformer protocol so the numbers are comparable to the
time-series literature: chronological (never random) splits at 12/4/4 months for ETTh1, and
per-channel standardization fit on the training split alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ETTH1_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"


@dataclass
class Standardizer:
    """Per-channel z-scoring with statistics frozen from the training split.

    Kept as an explicit object rather than a preprocessing step because the inverse transform is
    needed at evaluation time: reconstruction error and acceptance thresholds have to be reported
    in the data's own units, not in standardized ones, or they are not comparable across the ``d``
    sweep.
    """

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> Standardizer:
        return cls(mean=x.mean(0), std=x.std(0).clip(min=1e-8))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse(self, x: np.ndarray | torch.Tensor):
        if isinstance(x, torch.Tensor):
            mean = torch.as_tensor(self.mean, device=x.device, dtype=x.dtype)
            std = torch.as_tensor(self.std, device=x.device, dtype=x.dtype)
            return x * std + mean
        return x * self.std + self.mean


class SlidingWindowDataset(Dataset):
    """Contiguous windows of a multivariate series, as ``(C, L)`` float32 tensors.

    Channels come first to match the 1-D convolutional tokenizer, which treats the series the way
    an image tokenizer treats a picture: patch along the length axis, keep channels as features.
    """

    def __init__(self, series: np.ndarray, window: int, stride: int = 1):
        if series.ndim != 2:
            raise ValueError(f"expected (T, C) series, got shape {series.shape}")
        if len(series) < window:
            raise ValueError(f"series of length {len(series)} is shorter than window {window}")
        self.series = np.ascontiguousarray(series, dtype=np.float32)
        self.window = window
        self.stride = stride
        self.starts = np.arange(0, len(series) - window + 1, stride)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = int(self.starts[idx])
        return torch.from_numpy(self.series[start : start + self.window].T.copy())


@dataclass
class TimeSeriesBundle:
    """Train/val/test windows plus the scaler that produced them."""

    train: SlidingWindowDataset
    val: SlidingWindowDataset
    test: SlidingWindowDataset
    standardizer: Standardizer
    n_channels: int

    def describe(self) -> str:
        return (
            f"{self.n_channels}ch  train={len(self.train)}  val={len(self.val)}  "
            f"test={len(self.test)}  window={self.train.window}"
        )


def chronological_split(
    series: np.ndarray, fractions: tuple[float, float, float] = (0.7, 0.1, 0.2)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a series in time order.

    Never shuffle before splitting. Random splits leak future values into the training set through
    overlapping windows, which inflates every downstream number and is the most common way a
    time-series result turns out to be nothing.
    """
    if not abs(sum(fractions) - 1.0) < 1e-6:
        raise ValueError(f"fractions must sum to 1, got {fractions}")
    n = len(series)
    n_train = int(n * fractions[0])
    n_val = int(n * fractions[1])
    return series[:n_train], series[n_train : n_train + n_val], series[n_train + n_val :]


def build_bundle(
    series: np.ndarray,
    window: int = 96,
    stride: int = 1,
    fractions: tuple[float, float, float] = (0.7, 0.1, 0.2),
) -> TimeSeriesBundle:
    """Split chronologically, standardize on train only, and window each split.

    Windows are cut *after* splitting, so no window ever straddles a split boundary and no test
    window shares a timestamp with a training one.
    """
    train_raw, val_raw, test_raw = chronological_split(series, fractions)
    scaler = Standardizer.fit(train_raw)
    return TimeSeriesBundle(
        train=SlidingWindowDataset(scaler.transform(train_raw), window, stride),
        val=SlidingWindowDataset(scaler.transform(val_raw), window, stride),
        test=SlidingWindowDataset(scaler.transform(test_raw), window, stride),
        standardizer=scaler,
        n_channels=series.shape[1],
    )


def load_etth1(path: str | Path = "data/ETTh1.csv") -> np.ndarray:
    """Load ETTh1 as a ``(T, 7)`` array. Hourly electricity transformer temperature."""
    import pandas as pd

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Fetch it with:\n"
            f"  mkdir -p {path.parent} && curl -L {ETTH1_URL} -o {path}"
        )
    df = pd.read_csv(path)
    return df.drop(columns=[c for c in ("date",) if c in df.columns]).to_numpy(np.float32)


def load_electricity(path: str | Path = "data/LD2011_2014.txt", n_clients: int | None = 32) -> np.ndarray:
    """Load UCI Electricity as ``(T, n_clients)``, resampled hourly.

    The raw file is 15-minute resolution over 370 clients. Hourly resampling and a client subset
    keep the sweep affordable; ``n_clients=None`` returns all of them.
    """
    import pandas as pd

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download LD2011_2014.txt.zip from the UCI "
            f"'ElectricityLoadDiagrams20112014' dataset and unzip it to {path}."
        )
    df = pd.read_csv(path, sep=";", index_col=0, parse_dates=True, decimal=",")
    df = df.resample("1h").mean()
    if n_clients is not None:
        # Highest-variance clients: the flat ones carry almost no signal to model.
        df = df[df.std().sort_values(ascending=False).index[:n_clients]]
    return df.to_numpy(np.float32)


def synthetic_ar(
    n_steps: int = 4096,
    n_channels: int = 4,
    order: int = 3,
    noise: float = 0.1,
    seed: int = 0,
) -> np.ndarray:
    """A stable AR(``order``) process, for tests and for smoke-running the training loop.

    Exists so that every piece of the E1.x pipeline can be exercised without a download, and so
    that a training bug shows up as a failure to fit a process whose structure is known rather
    than as a mediocre number on real data. Coefficients are shrunk until the companion matrix is
    stable, which keeps the series from diverging.
    """
    rng = np.random.default_rng(seed)
    coeffs = rng.normal(0, 0.5, size=(order, n_channels))
    for _ in range(100):
        companion = np.zeros((order, order))
        companion[0] = coeffs.mean(1)
        companion[1:, :-1] = np.eye(order - 1)
        if np.abs(np.linalg.eigvals(companion)).max() < 0.95:
            break
        coeffs *= 0.8

    series = np.zeros((n_steps + order, n_channels), dtype=np.float32)
    series[:order] = rng.normal(0, 1, (order, n_channels))
    for t in range(order, n_steps + order):
        series[t] = (coeffs * series[t - order : t][::-1]).sum(0) + rng.normal(
            0, noise, n_channels
        )
    return series[order:]
