#!/usr/bin/env python
"""E0.5 -- is the KL16 decoder's gain isotropic in latent space?

Gate-exempt: this is a frozen public VAE, and the answer does not presuppose that HERALD works. It
decides a design question. Every acceptance rule in the plan thresholds an isotropic latent norm
(``||E_ema(x̂) - z|| <= τ``), which is only the right shape if equal-size latent perturbations
produce equal-size pixel changes in every direction. If they do not, a single scalar τ cannot
separate a harmless draft from a visibly wrong one, and the nonconformity score needs Jacobian
weighting.

Latents come from re-encoding E0.1's MAR samples, so they sit where the method will actually
operate rather than at a random point of latent space. The caveat is that they are model samples
rather than photographs; a real-ImageNet pass should confirm the number once data is available.

Run:  python scripts/run_e05.py --k 16 --power-iters 1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from herald.metrics.decoder_sensitivity import direction_gain_profile, jacobian_spectrum
from herald.models.mar_adapter import LATENT_SCALE, load_kl16


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def load_grid_images(path: Path, tile: int = 256, pad: int = 2, nrow: int = 3, limit: int = 4):
    """Slice a torchvision ``make_grid`` montage back into individual images in [-1, 1]."""
    grid = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    images = []
    step = tile + pad
    rows = (grid.shape[0] - pad) // step
    cols = (grid.shape[1] - pad) // step
    for r in range(rows):
        for c in range(cols):
            if len(images) >= limit:
                break
            y, x = pad + r * step, pad + c * step
            crop = grid[y : y + tile, x : x + tile]
            if crop.shape[:2] != (tile, tile):
                continue
            # A fully black tile is grid padding where the batch ran out, not an image.
            if crop.max() < 1e-6:
                continue
            images.append(torch.from_numpy(crop).permute(2, 0, 1) * 2.0 - 1.0)
    if not images:
        raise RuntimeError(f"no tiles recovered from {path}")
    return torch.stack(images[:limit])


def build_decode_fn(vae, crop: int):
    """MAR's token space -> pixels. The latent scale matters for absolute gain, not for anisotropy."""

    def decode(z: torch.Tensor) -> torch.Tensor:
        return vae.decode(z / LATENT_SCALE)

    return decode


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples", default="experiments/E0.1-mar-base/samples.png")
    p.add_argument("--images", default=None,
                   help="a .pt tensor of images in [-1,1] from generate_mar_samples.py, used "
                        "instead of slicing --samples. The montage path caps the count at the "
                        "tiles it happens to contain and quantises to 8 bits; this does neither.")
    p.add_argument("--n-latents", type=int, default=4)
    p.add_argument("--crop", type=int, default=256, help="image size; 128 quarters the memory")
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--oversample", type=int, default=8)
    p.add_argument("--power-iters", type=int, default=1)
    p.add_argument("--eps", type=float, default=1e-2)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="experiments/E0.5-kl16-jacobian")
    args = p.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.images is not None:
        images = torch.load(args.images, map_location="cpu")[: args.n_latents].float()
    else:
        images = load_grid_images(Path(args.samples), limit=args.n_latents)
    if args.crop != 256:
        off = (256 - args.crop) // 2
        images = images[:, :, off : off + args.crop, off : off + args.crop]
    print(f"{images.shape[0]} images at {tuple(images.shape[-2:])}")

    def run_on(dev: str):
        vae = load_kl16(device=dev)
        for prm in vae.parameters():
            prm.requires_grad_(False)
        decode = build_decode_fn(vae, args.crop)
        gen = torch.Generator(device=dev).manual_seed(args.seed)

        per_latent = []
        for i, img in enumerate(images):
            with torch.no_grad():
                z = vae.encode(img.unsqueeze(0).to(dev)).sample() * LATENT_SCALE
            t0 = time.time()
            spec = jacobian_spectrum(
                decode, z, k=args.k, oversample=args.oversample,
                power_iters=args.power_iters, generator=gen,
            )
            gains = direction_gain_profile(decode, z, spec, eps=args.eps)
            s = spec.singular_values.detach().cpu()
            per_latent.append({
                "index": i,
                "latent_shape": list(z.shape),
                "singular_values": s.tolist(),
                "anisotropy_top_k": spec.anisotropy,
                "effective_rank": spec.effective_rank(),
                "s_max": float(s[0]),
                "s_min": float(s[-1]),
                "top_direction": {
                    "measured_gain": gains["top"]["measured_gain"],
                    "predicted_gain": gains["top"]["predicted_gain"],
                },
                "bottom_direction": {
                    "measured_gain": gains["bottom"]["measured_gain"],
                    "predicted_gain": gains["bottom"]["predicted_gain"],
                },
                "seconds": time.time() - t0,
            })
            print(f"  latent {i}: anisotropy(top-{args.k}) {spec.anisotropy:8.2f}  "
                  f"eff.rank {spec.effective_rank():6.2f}  "
                  f"s [{s[0]:.3f} .. {s[-1]:.4f}]  ({per_latent[-1]['seconds']:.1f}s)")
        return per_latent

    try:
        print(f"device: {device}")
        per_latent = run_on(device)
    except torch.cuda.OutOfMemoryError:
        print(f"CUDA OOM at {args.crop}px (only ~1 GB free on this host); falling back to CPU")
        torch.cuda.empty_cache()
        device = "cpu"
        per_latent = run_on(device)

    aniso = [r["anisotropy_top_k"] for r in per_latent]
    erank = [r["effective_rank"] for r in per_latent]
    lin_err = [
        abs(r[d]["measured_gain"] - r[d]["predicted_gain"]) / max(r[d]["predicted_gain"], 1e-12)
        for r in per_latent for d in ("top_direction", "bottom_direction")
    ]

    summary = {
        "experiment": "E0.5",
        "commit": git_commit(),
        "device": device,
        "config": vars(args),
        "n_latents": len(per_latent),
        # Quartiles rather than min/max alone: with more than a handful of latents the extremes are
        # driven by the sample size, whereas the IQR says how much the spectrum varies image to image.
        "anisotropy_top_k": {"median": float(np.median(aniso)), "min": float(min(aniso)),
                             "max": float(max(aniso)),
                             "q25": float(np.percentile(aniso, 25)),
                             "q75": float(np.percentile(aniso, 75))},
        "effective_rank": {"median": float(np.median(erank)), "of_k": args.k,
                           "min": float(min(erank)), "max": float(max(erank)),
                           "q25": float(np.percentile(erank, 25)),
                           "q75": float(np.percentile(erank, 75))},
        # The between-image spread of the top singular value is the finding E0.5 surfaced by
        # accident: a latent-space tolerance buys different pixel error depending on the image.
        "s_max_between_images": {
            "median": float(np.median([r["s_max"] for r in per_latent])),
            "min": float(min(r["s_max"] for r in per_latent)),
            "max": float(max(r["s_max"] for r in per_latent)),
            "q25": float(np.percentile([r["s_max"] for r in per_latent], 25)),
            "q75": float(np.percentile([r["s_max"] for r in per_latent], 75)),
        },
        "linearization_relative_error": {"median": float(np.median(lin_err)),
                                         "max": float(max(lin_err))},
        "per_latent": per_latent,
    }

    print(f"\n{len(per_latent)} latents")
    print(f"anisotropy over top-{args.k} directions : median {np.median(aniso):.2f} "
          f"(IQR {np.percentile(aniso, 25):.2f}-{np.percentile(aniso, 75):.2f}, "
          f"range {min(aniso):.2f}-{max(aniso):.2f})")
    print(f"effective rank of those {args.k}          : median {np.median(erank):.2f} "
          f"(IQR {np.percentile(erank, 25):.2f}-{np.percentile(erank, 75):.2f})")
    smax = [r["s_max"] for r in per_latent]
    print(f"top singular value, between images      : median {np.median(smax):.2f} "
          f"(range {min(smax):.2f}-{max(smax):.2f}, {max(smax)/min(smax):.1f}x spread)")
    print(f"linearization relative error            : median {np.median(lin_err):.3f}, "
          f"max {max(lin_err):.3f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out / 'results.json'}")


if __name__ == "__main__":
    main()
