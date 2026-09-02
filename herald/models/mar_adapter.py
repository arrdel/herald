"""Adapter onto the public MAR checkpoints, for E0.1 and everything downstream.

``baselines/`` stays pristine, so nothing here patches upstream. The repo is put on ``sys.path``
just long enough to import its modules (see :func:`herald.models.baselines.baseline_imports`, which
also keeps the three baselines' identically-named ``models`` packages from shadowing each other),
and the model is then driven through its own public API.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .baselines import REPO_ROOT, baseline_imports

#: MAR scales VAE latents by this factor on encode and divides it out on decode
#: (``baselines/mar/engine_mar.py`` lines 65 and 159). Omitting it does not crash -- it produces
#: washed-out images and a quietly terrible FID.
LATENT_SCALE = 0.2325

@dataclass(frozen=True)
class MARSpec:
    """Per-size configuration, cross-checked against the checkpoint and the upstream README.

    ``published_fid`` is FID-50K with classifier-free guidance at ``cfg``, from the table in
    ``baselines/mar/README.md``. ``cfg`` is the value that table's numbers were produced at;
    reproducing at any other guidance scale is not a reproduction.
    """

    builder: str
    weights: str
    diffloss_d: int
    diffloss_w: int
    cfg: float
    published_fid: float
    published_inception: float
    params: str


MAR_SPECS: dict[str, MARSpec] = {
    "base": MARSpec("mar_base", "mar-base.safetensors", 6, 1024, 2.9, 2.31, 281.7, "208M"),
    "large": MARSpec("mar_large", "mar-large.safetensors", 8, 1280, 3.0, 1.78, 296.0, "479M"),
    "huge": MARSpec("mar_huge", "mar-huge.safetensors", 12, 1536, 3.2, 1.55, 303.7, "943M"),
}

def _load_safetensors(path: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/download_checkpoints.sh")
    return load_file(str(path))


def load_mar(
    size: str = "base",
    ckpt_dir: str | Path = "checkpoints",
    num_sampling_steps: str = "100",
    device: str = "cuda",
    mar_repo: Path | None = None,
) -> torch.nn.Module:
    """Build a MAR model and load public weights into it.

    ``num_sampling_steps`` is the denoising-head step count and is passed through as a string
    because upstream treats it as a ``timestep_respacing`` spec, not an integer.

    Loading is strict. A silently partial load is the single most likely way to produce a
    plausible-looking FID that reproduces nothing, and this is the function every later experiment
    depends on.
    """
    if size not in MAR_SPECS:
        raise ValueError(f"unknown MAR size {size!r}; expected one of {sorted(MAR_SPECS)}")
    spec = MAR_SPECS[size]
    repo = Path(mar_repo) if mar_repo else REPO_ROOT / "baselines" / "mar"

    with baseline_imports(repo):
        import models.mar as mar_module

        model = getattr(mar_module, spec.builder)(
            img_size=256,
            vae_stride=16,
            patch_size=1,
            vae_embed_dim=16,
            diffloss_d=spec.diffloss_d,
            diffloss_w=spec.diffloss_w,
            num_sampling_steps=num_sampling_steps,
        )

    state = _load_safetensors(Path(ckpt_dir) / "mar" / spec.weights)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"MAR-{size} checkpoint does not match the model definition.\n"
            f"  missing ({len(missing)}): {sorted(missing)[:8]}\n"
            f"  unexpected ({len(unexpected)}): {sorted(unexpected)[:8]}"
        )
    return model.to(device).eval()


def load_kl16(
    ckpt_dir: str | Path = "checkpoints",
    device: str = "cuda",
    mar_repo: Path | None = None,
) -> torch.nn.Module:
    """Load the KL-16 VAE that MAR, D-JEPA, and CSpD all share."""
    repo = Path(mar_repo) if mar_repo else REPO_ROOT / "baselines" / "mar"
    with baseline_imports(repo):
        from models.vae import AutoencoderKL

        vae = AutoencoderKL(embed_dim=16, ch_mult=(1, 1, 2, 2, 4), ckpt_path=None)

    path = Path(ckpt_dir) / "vae" / "kl16.safetensors"
    if not path.exists():
        path = Path(ckpt_dir) / "mar" / "kl16.safetensors"
    missing, unexpected = vae.load_state_dict(_load_safetensors(path), strict=False)
    # The checkpoint carries no loss-module weights, which the class defines but decoding never
    # touches; anything else missing means the wrong file.
    real_missing = [k for k in missing if not k.startswith("loss.")]
    if real_missing or unexpected:
        raise RuntimeError(f"KL16 mismatch: missing={real_missing[:8]} unexpected={unexpected[:8]}")
    return vae.to(device).eval()


@dataclass
class MARSampler:
    """MAR + KL16 wired together, with NFE accounting on by default.

    ``generate`` returns decoded images in ``[-1, 1]``, matching the convention
    :class:`~herald.metrics.PairedImageAudit` expects at ``data_range=2.0``.
    """

    model: torch.nn.Module
    vae: torch.nn.Module
    size: str = "base"
    device: str = "cuda"

    @classmethod
    def load(cls, size: str = "base", ckpt_dir: str | Path = "checkpoints", device: str = "cuda", **kwargs):
        return cls(
            model=load_mar(size, ckpt_dir, device=device, **kwargs),
            vae=load_kl16(ckpt_dir, device=device),
            size=size,
            device=device,
        )

    @property
    def spec(self) -> MARSpec:
        return MAR_SPECS[self.size]

    @torch.no_grad()
    def generate(
        self,
        labels: torch.Tensor,
        num_iter: int = 256,
        cfg: float | None = None,
        cfg_schedule: str = "linear",
        temperature: float = 1.0,
        ledger=None,
        seed: int | None = None,
        autocast: bool = True,
    ) -> torch.Tensor:
        """Generate one image per entry of ``labels``. Returns ``(N, 3, 256, 256)`` in ``[-1, 1]``.

        Defaults reproduce the upstream evaluation command: 256 autoregressive iterations, linear
        CFG schedule, and the per-size guidance scale the published FID was measured at.
        """
        from herald.metrics import instrument_mar_sampler

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        cfg = self.spec.cfg if cfg is None else cfg
        labels = labels.to(self.device).long()

        restore = instrument_mar_sampler(self.model, ledger) if ledger is not None else None
        try:
            with torch.autocast("cuda", enabled=autocast and self.device.startswith("cuda")):
                tokens = self.model.sample_tokens(
                    bsz=labels.shape[0],
                    num_iter=num_iter,
                    cfg=cfg,
                    cfg_schedule=cfg_schedule,
                    labels=labels,
                    temperature=temperature,
                )
                images = self.vae.decode(tokens.float() / LATENT_SCALE)
        finally:
            if restore is not None:
                restore()
        return images.float()


def to_uint8(images: torch.Tensor) -> torch.Tensor:
    """``[-1, 1]`` float images to ``(N, H, W, 3)`` uint8, matching upstream's saved PNGs."""
    x = ((images + 1) / 2).clamp(0, 1).mul(255).round().to(torch.uint8)
    return x.permute(0, 2, 3, 1).contiguous()
