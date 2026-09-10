#!/usr/bin/env python
"""Generate a bank of MAR samples and save them as a tensor, for downstream latent analysis.

E0.5 characterises the KL16 decoder's Jacobian spectrum at latents obtained by re-encoding MAR's own
samples, so that the measurement sits where the sampler actually operates rather than at an arbitrary
point of latent space. The original pass used the four tiles recoverable from E0.1's ``samples.png``
montage, which a reviewer reasonably called pilot-scale. This script produces as many samples as
asked for and stores them directly as a float tensor in ``[-1, 1]``, skipping the montage round trip
that limited the sample count and quantised the images to 8 bits on the way.

Sampling settings default to the ones E0.1 reproduced MAR-B with, so the latents are drawn from the
same distribution as the original four and the expanded measurement is comparable to it rather than
a different experiment.

Run:  python scripts/generate_mar_samples.py --num-images 50 --out experiments/E0.5b-samples
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from herald.models.mar_adapter import MAR_SPECS, MARSampler


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def balanced_labels(n: int, n_classes: int = 1000, seed: int = 0) -> torch.Tensor:
    """Class labels spread evenly over ImageNet, shuffled deterministically (as in E0.1)."""
    labels = np.tile(np.arange(n_classes), int(np.ceil(n / n_classes)))[:n]
    np.random.default_rng(seed).shuffle(labels)
    return torch.from_numpy(labels).long()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", default="base")
    p.add_argument("--num-images", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-iter", type=int, default=256)
    p.add_argument("--cfg", type=float, default=None, help="defaults to the size's published value")
    p.add_argument("--cfg-schedule", default="linear")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--out", default="experiments/E0.5b-samples")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("MAR sampling needs a GPU")

    sampler = MARSampler.load(size=args.size, ckpt_dir=args.ckpt_dir, device="cuda")
    cfg = MAR_SPECS[args.size].cfg if args.cfg is None else args.cfg
    labels = balanced_labels(args.num_images, seed=args.seed)
    print(f"MAR-{args.size}, {args.num_images} images, cfg={cfg}, {args.num_iter} iters", flush=True)

    chunks = []
    t0 = time.time()
    for start in range(0, args.num_images, args.batch_size):
        batch = labels[start : start + args.batch_size]
        images = sampler.generate(
            batch, num_iter=args.num_iter, cfg=cfg, cfg_schedule=args.cfg_schedule,
            temperature=args.temperature, seed=args.seed + start,
        )
        # Kept on CPU in float32: the Jacobian pass wants full precision, and 50 images at 256px is
        # only ~40 MB, so there is no reason to quantise to uint8 and lose it.
        chunks.append(images.cpu())
        done = start + batch.shape[0]
        print(f"  {done}/{args.num_images} images, {time.time() - t0:.0f}s", flush=True)

    samples = torch.cat(chunks)[: args.num_images]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(samples, out / "images.pt")
    (out / "meta.json").write_text(json.dumps({
        "commit": git_commit(), "host": platform.node(),
        "gpu": torch.cuda.get_device_name(0),
        "size": args.size, "num_images": int(samples.shape[0]),
        "num_iter": args.num_iter, "cfg": cfg, "cfg_schedule": args.cfg_schedule,
        "temperature": args.temperature, "seed": args.seed,
        "shape": list(samples.shape), "seconds": time.time() - t0,
    }, indent=2))
    print(f"\nwrote {out / 'images.pt'}  {tuple(samples.shape)}  in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
