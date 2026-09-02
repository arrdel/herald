#!/usr/bin/env python
"""E0.1 -- reproduce MAR unmodified and instrument where the compute actually goes.

Two deliverables, with very different sample-count requirements:

1. **The NFE / wall-time breakdown.** Transformer passes versus denoising-head evaluations. This
   is the empirical basis for the free-drafter claim in section 1 of the research plan, and it
   converges after a handful of batches because it is a deterministic property of the sampling
   loop, not a statistical estimate.
2. **The FID reproduction gate** (within 0.2 of published). This needs the full 50k samples.
   FID is badly biased upward at small sample counts, so this script refuses to report it below
   ``--min-fid-images`` rather than emit a number that looks like a failed reproduction.

Usage:
    python scripts/run_e01.py --size base --num-images 64 --batch-size 8      # breakdown only
    python scripts/run_e01.py --size base --num-images 50000 --compute-fid    # the gate
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from herald.metrics import NFELedger
from herald.models.mar_adapter import MAR_SPECS, MARSampler, to_uint8

#: Below this, FID is dominated by its own small-sample bias and says nothing about reproduction.
DEFAULT_MIN_FID_IMAGES = 10_000


def balanced_labels(n: int, n_classes: int = 1000, seed: int = 0) -> torch.Tensor:
    """Class labels spread evenly over ImageNet, shuffled deterministically.

    Upstream evaluates on a balanced sweep of all 1000 classes. Sampling labels uniformly at
    random instead would add class-imbalance variance to FID that the published number does not
    have, which matters when the gate is 0.2 FID wide.
    """
    labels = np.tile(np.arange(n_classes), int(np.ceil(n / n_classes)))[:n]
    np.random.default_rng(seed).shuffle(labels)
    return torch.from_numpy(labels).long()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


@torch.no_grad()
def inception_features(images_uint8: torch.Tensor, device: str, batch_size: int = 32) -> np.ndarray:
    """Pool3 Inception features via torchmetrics' TF-compatible InceptionV3 port."""
    from torchmetrics.image.fid import NoTrainInceptionV3

    net = NoTrainInceptionV3(name="inception-v3-compat", features_list=["2048"]).to(device).eval()
    feats = []
    for i in range(0, len(images_uint8), batch_size):
        batch = images_uint8[i : i + batch_size].permute(0, 3, 1, 2).to(device)
        feats.append(net(batch).cpu().numpy())
    return np.concatenate(feats)


def frechet_distance(mu1, sigma1, mu2, sigma2) -> float:
    from scipy import linalg

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", default="base", choices=sorted(MAR_SPECS))
    p.add_argument("--num-images", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-iter", type=int, default=256, help="autoregressive steps (upstream: 256)")
    p.add_argument("--cfg", type=float, default=None, help="default: the per-size published value")
    p.add_argument("--cfg-schedule", default="linear")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--out", default=None, help="default: experiments/E0.1-mar-<size>")
    p.add_argument("--compute-fid", action="store_true")
    p.add_argument("--fid-stats", default="baselines/mar/fid_stats/adm_in256_stats.npz")
    p.add_argument("--min-fid-images", type=int, default=DEFAULT_MIN_FID_IMAGES)
    p.add_argument("--save-grid", type=int, default=16, help="images in the sanity-check grid; 0 to skip")
    p.add_argument("--no-autocast", action="store_true")
    args = p.parse_args()

    out = Path(args.out or f"experiments/E0.1-mar-{args.size}")
    out.mkdir(parents=True, exist_ok=True)
    spec = MAR_SPECS[args.size]
    cfg = spec.cfg if args.cfg is None else args.cfg

    print(f"E0.1  MAR-{args.size}  cfg={cfg}  num_iter={args.num_iter}  n={args.num_images}")
    print(f"      published FID-50K {spec.published_fid} @ cfg {spec.cfg}")

    sampler = MARSampler.load(args.size, args.ckpt_dir, device=args.device)
    labels_all = balanced_labels(args.num_images, seed=args.seed)

    ledger = NFELedger()
    warmup_ledger = NFELedger()
    all_images: list[torch.Tensor] = []
    timed_seconds, timed_images = 0.0, 0

    n_batches = int(np.ceil(args.num_images / args.batch_size))
    for i in range(n_batches):
        labels = labels_all[i * args.batch_size : (i + 1) * args.batch_size]
        if not len(labels):
            break

        # The first batch pays CUDA init and autotuning; upstream also excludes it from timing.
        # Its NFE counts go to a separate ledger so the breakdown stays consistent with the clock.
        batch_ledger = warmup_ledger if i == 0 else ledger
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()

        images = sampler.generate(
            labels,
            num_iter=args.num_iter,
            cfg=cfg,
            cfg_schedule=args.cfg_schedule,
            temperature=args.temperature,
            ledger=batch_ledger,
            seed=args.seed + i,
            autocast=not args.no_autocast,
        )
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        if i > 0:
            timed_seconds += elapsed
            timed_images += len(labels)
        all_images.append(to_uint8(images).cpu())
        print(
            f"  batch {i + 1}/{n_batches}  {elapsed:.1f}s  "
            f"({elapsed / len(labels):.2f}s/img)" + ("  [warmup, untimed]" if i == 0 else "")
        )

    images_uint8 = torch.cat(all_images)

    head = ledger.token_evals.get("diffusion_head", 0)
    transformer = ledger.token_evals.get("encoder", 0) + ledger.token_evals.get("decoder", 0)
    results = {
        "experiment": "E0.1",
        "model": f"MAR-{args.size}",
        "commit": git_commit(),
        "host": platform.node(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "config": {**vars(args), "cfg": cfg, "spec": asdict(spec)},
        "n_images": int(len(images_uint8)),
        "timing": {
            "timed_images": timed_images,
            "seconds": round(timed_seconds, 3),
            "seconds_per_image": round(timed_seconds / max(1, timed_images), 4),
            "note": "first batch excluded as warmup, matching upstream",
        },
        "nfe": ledger.as_dict(),
        "nfe_split": {
            "transformer_token_evals": transformer,
            "head_token_evals": head,
            "head_over_transformer": round(head / transformer, 2) if transformer else None,
            "note": (
                "head counts include the ~100 denoising steps inside each diffloss.sample call; "
                "CFG doubles both stages and that doubling is included"
            ),
        },
    }
    print("\n" + ledger.report())
    if transformer:
        print(f"\nhead / transformer token-evals = {head / transformer:.1f}x")

    if args.compute_fid:
        if len(images_uint8) < args.min_fid_images:
            results["fid"] = {
                "status": "skipped",
                "reason": (
                    f"{len(images_uint8)} images is below --min-fid-images "
                    f"({args.min_fid_images}); FID is biased upward at small N and the number "
                    f"would not be comparable to the published FID-50K"
                ),
            }
            print(f"\nFID skipped: {results['fid']['reason']}")
        else:
            ref = np.load(args.fid_stats)
            feats = inception_features(images_uint8, args.device)
            fid = frechet_distance(
                feats.mean(0), np.cov(feats, rowvar=False), ref["mu"], ref["sigma"]
            )
            results["fid"] = {
                "status": "measured",
                "value": round(fid, 4),
                "published": spec.published_fid,
                "delta": round(fid - spec.published_fid, 4),
                "gate_passed": abs(fid - spec.published_fid) <= 0.2,
                "n_images": int(len(images_uint8)),
                "reference": args.fid_stats,
                "caveat": (
                    "feature extractor is torchmetrics inception-v3-compat; the reference stats "
                    "are ADM's TF Inception. Validate the two agree before treating the gate as met"
                ),
            }
            print(f"\nFID {fid:.4f} vs published {spec.published_fid} "
                  f"(gate {'PASS' if results['fid']['gate_passed'] else 'FAIL'})")

    (out / "results.json").write_text(json.dumps(results, indent=2, default=str))

    if args.save_grid:
        from torchvision.utils import save_image

        n = min(args.save_grid, len(images_uint8))
        grid = images_uint8[:n].permute(0, 3, 1, 2).float() / 255
        save_image(grid, out / "samples.png", nrow=int(np.ceil(np.sqrt(n))))
        print(f"\nwrote {out}/samples.png")

    print(f"wrote {out}/results.json")


if __name__ == "__main__":
    main()
