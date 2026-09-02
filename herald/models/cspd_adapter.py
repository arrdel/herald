"""Driving CSpD's speculative decoder and capturing what its acceptance test actually sees.

Two jobs.

**Loading.** CSpD's `MAR` is a fork of upstream MAR with the same constructor signature and the same
parameter set, plus extra *methods* for stepwise sampling. So MAR's public checkpoints load into it,
and :func:`load_cspd_mar` insists on that with ``strict=True`` -- if the fork ever diverged in
parameters we want a loud failure, not a silently half-initialised network.

**Capture, and why it is shaped this way.** The obvious way to run E0.3 is one generation per
temperature. That is six generations for six temperatures, and with all eight GPUs on this host
allocated to another job, six of anything is not affordable.

It is also unnecessary. The temperature does not influence which tokens get drafted or what the two
models predict -- it enters only at ``judge_accept``, after both distributions are already fixed
(verified by call path; see ``docs/RELATED_WORK.md``). So a single generation can record every
``DistributionProcessor`` input, and the whole sweep becomes arithmetic on cached tensors. Six GPU
runs collapse to one, and the offline part costs nothing.

The one thing this design cannot see is the *downstream* effect of temperature: accepting different
tokens changes what gets drafted next, so a cached sweep gives the acceptance rate the shipped
sampler would apply at each decision point along **one** trajectory, not the end-to-end speedup or
sample quality. That is the right trade here -- acceptance rate is what C1 is about -- but it is a
real limitation and E0.3's writeup has to say so. Measuring end-to-end speedup and FID per
temperature still needs the expensive version.

Nothing here modifies ``baselines/``. The capture works by swapping a recording subclass into the
runner module's namespace at runtime and restoring it afterwards.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .baselines import REPO_ROOT, baseline_imports
from .mar_adapter import MAR_SPECS, _load_safetensors

CSPD_REPO = REPO_ROOT / "baselines" / "CSpD"

#: The demo notebook's configuration -- the only place upstream wires the speculative path up.
DEMO_CONFIG = {
    "draft": "base",
    "target": "huge",
    "ratio": 0.05,
    "num_draft": 4,
    "cfg": 4.0,
    "cfg_schedule": "constant",
}


def load_cspd_mar(
    size: str,
    num_sampling_steps: str = "100",
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
):
    """Build CSpD's MAR at ``size`` and load MAR's public weights into it strictly."""
    if size not in MAR_SPECS:
        raise KeyError(f"unknown MAR size {size!r}; expected one of {sorted(MAR_SPECS)}")
    spec = MAR_SPECS[size]
    ckpt = REPO_ROOT / "checkpoints" / "mar" / spec.weights
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} not found; run scripts/download_checkpoints.sh")

    with baseline_imports(CSPD_REPO):
        import models.mar as cspd_mar

        builder = getattr(cspd_mar, spec.builder)
        model = builder(buffer_size=64, num_sampling_steps=num_sampling_steps)

    state = _load_safetensors(ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"CSpD's MAR fork does not match MAR's public {spec.weights}. "
            f"missing={sorted(missing)[:8]} unexpected={sorted(unexpected)[:8]}. "
            "The fork has diverged in parameters; the capture would be measuring a different model."
        )

    n_params = sum(p.numel() for p in model.parameters())
    return model.to(device=device, dtype=dtype).eval(), n_params


@dataclass
class AcceptanceCapture:
    """Everything CSpD's acceptance test was handed, one entry per verification step."""

    draft_mean: list[torch.Tensor] = field(default_factory=list)
    draft_std: list[torch.Tensor] = field(default_factory=list)
    draft_log_sigma_accum: list[torch.Tensor] = field(default_factory=list)
    target_mean: list[torch.Tensor] = field(default_factory=list)
    target_std: list[torch.Tensor] = field(default_factory=list)
    target_log_sigma_accum: list[torch.Tensor] = field(default_factory=list)
    x: list[torch.Tensor] = field(default_factory=list)
    dim: int | None = None

    def __len__(self) -> int:
        return len(self.x)

    @property
    def n_tokens(self) -> int:
        return sum(int(t.shape[0]) for t in self.x)

    def concat(self) -> dict[str, torch.Tensor]:
        if not self.x:
            raise RuntimeError(
                "capture is empty: the runner never reached a verification step. "
                "Check that num_iter * ratio < num_iter and that num_iter > num_draft."
            )
        return {
            name: torch.cat(getattr(self, name), dim=0).float()
            for name in (
                "draft_mean", "draft_std", "draft_log_sigma_accum",
                "target_mean", "target_std", "target_log_sigma_accum", "x",
            )
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"dim": self.dim, **self.concat()}, path)
        return path


@contextlib.contextmanager
def capturing_processor(runner_module, capture: AcceptanceCapture):
    """Swap in a ``DistributionProcessor`` that records its inputs, then restore the original.

    ``SpecDecodeRunner.run`` looks ``DistributionProcessor`` up as a module global, so rebinding the
    module attribute is enough -- no upstream file is touched.
    """
    original = runner_module.DistributionProcessor

    class Recording(original):
        def __init__(self, x1_mean, x1_std, x1_log_var, x2_mean, x2_std, x2_log_var, dim, temperature=42.0):
            capture.dim = dim
            capture.draft_mean.append(x1_mean.detach().cpu())
            capture.draft_std.append(x1_std.detach().cpu())
            capture.draft_log_sigma_accum.append(_as_tensor(x1_log_var, x1_mean).detach().cpu())
            capture.target_mean.append(x2_mean.detach().cpu())
            capture.target_std.append(x2_std.detach().cpu())
            capture.target_log_sigma_accum.append(_as_tensor(x2_log_var, x2_mean).detach().cpu())
            super().__init__(x1_mean, x1_std, x1_log_var, x2_mean, x2_std, x2_log_var, dim, temperature)

        def speculative_sample(self, x):
            capture.x.append(x.detach().cpu())
            return super().speculative_sample(x)

    runner_module.DistributionProcessor = Recording
    try:
        yield
    finally:
        runner_module.DistributionProcessor = original


def _as_tensor(value, like: torch.Tensor) -> torch.Tensor:
    """``manager.log_var`` is initialised to the float ``0.`` and only then accumulated into.

    If a model somehow ran a single denoising step it would still be a scalar, and silently
    broadcasting that into the capture would make the accumulated term look uniformly zero.
    """
    if torch.is_tensor(value):
        return value
    return torch.full_like(like, float(value))
