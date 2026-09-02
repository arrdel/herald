#!/usr/bin/env python
"""E0.3 -- what CSpD's acceptance test actually computes on real MAR draft/target pairs.

The fulcrum experiment for claim C1. Two stages, deliberately separated so the expensive one runs
once:

``capture`` (needs a GPU)
    Run CSpD's ``SpecDecodeRunner`` once, recording every input its acceptance test receives.
``analyze`` (no GPU)
    Sweep the temperature over the cached tensors and decompose acceptance four ways.

The temperature enters only at ``judge_accept``, after both models' distributions are fixed, so the
sweep does not need to be re-run per temperature. Six GPU runs become one. The limitation this buys
is real and reported: a cached sweep measures acceptance *at each decision point along one
trajectory*, not the end-to-end speedup or FID a different temperature would produce, because
accepting different tokens changes what gets drafted next.

Examples:
    python scripts/run_e03.py capture --draft base --target large --num-images 4
    python scripts/run_e03.py analyze --capture experiments/E0.3/capture.pt
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

from herald.models.baselines import baseline_imports
from herald.models.cspd_adapter import (
    CSPD_REPO,
    DEMO_CONFIG,
    AcceptanceCapture,
    capturing_processor,
    load_cspd_mar,
)
from herald.verifiers import (
    TokenGaussian,
    acceptance_from_log_ratio,
    cspd_log_ratio,
    exact_log_ratio,
    underflows_in_linear_space,
)

TEMPERATURES = [1.0, 2.0, 5.0, 10.0, 20.0, 42.0]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def balanced_labels(n: int, num_classes: int = 1000) -> torch.Tensor:
    return torch.arange(n) % num_classes


def do_capture(args) -> Path:
    device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("capture needs a GPU; CSpD's runner hardcodes device='cuda'")

    draft, n_draft = load_cspd_mar(args.draft, args.diff_steps, device=device)
    target, n_target = load_cspd_mar(args.target, args.diff_steps, device=device)
    print(f"draft  {args.draft}: {n_draft/1e6:.1f}M")
    print(f"target {args.target}: {n_target/1e6:.1f}M")

    capture = AcceptanceCapture()
    labels = balanced_labels(args.num_images).to(device)

    with baseline_imports(CSPD_REPO):
        import spec_decode.spec_decode_runner as runner_mod

        runner = runner_mod.SpecDecodeRunner(
            draft, target, inner_temperature=args.inner_temperature, ratio=args.ratio
        )
        with capturing_processor(runner_mod, capture), torch.no_grad():
            t0 = time.time()
            tokens = runner.run(
                bsz=args.num_images,
                num_iter=args.num_iter,
                num_draft=args.num_draft,
                cfg_schedule=args.cfg_schedule,
                labels=labels,
                temperature=1.0,
                cfg=args.cfg,
                progress=True,
            )
            torch.cuda.synchronize()
            elapsed = time.time() - t0

    print(f"generated {tuple(tokens.shape)} in {elapsed:.1f}s")
    print(f"captured {len(capture)} verification steps, {capture.n_tokens} token decisions")

    out = Path(args.out)
    path = capture.save(out / "capture.pt")
    (out / "capture_meta.json").write_text(json.dumps({
        "commit": git_commit(),
        "host": platform.node(),
        "gpu": torch.cuda.get_device_name(0),
        "draft": args.draft, "target": args.target,
        "draft_params": n_draft, "target_params": n_target,
        "num_images": args.num_images, "num_iter": args.num_iter,
        "num_draft": args.num_draft, "ratio": args.ratio, "cfg": args.cfg,
        "cfg_schedule": args.cfg_schedule, "diff_steps": args.diff_steps,
        "inner_temperature_used_during_capture": args.inner_temperature,
        "verification_steps": len(capture), "token_decisions": capture.n_tokens,
        "seconds": elapsed,
        "demo_config_for_reference": DEMO_CONFIG,
    }, indent=2))
    print(f"wrote {path}")
    return path


def do_analyze(args) -> None:
    blob = torch.load(args.capture, map_location="cpu", weights_only=True)
    draft = TokenGaussian(blob["draft_mean"], blob["draft_std"], blob["draft_log_sigma_accum"])
    target = TokenGaussian(blob["target_mean"], blob["target_std"], blob["target_log_sigma_accum"])
    x = blob["x"]
    n = int(x.shape[0])
    print(f"{n} token decisions, d={blob.get('dim')}")

    exact_lr = exact_log_ratio(x, draft, target)
    exact_rate = acceptance_from_log_ratio(exact_lr).mean().item()
    underflow = underflows_in_linear_space(exact_lr).float().mean().item()

    # The factor CSpD injects, measured rather than assumed.
    fixed_exponent = (target.log_sigma_accum - draft.log_sigma_accum).sum(-1)

    rows = []
    for temp in TEMPERATURES:
        row = {"temperature": temp}
        for name, fixed in (("no_fixed_term", False), ("as_shipped", True)):
            lr = cspd_log_ratio(x, draft, target, temperature=temp, include_fixed_term=fixed)
            row[name] = acceptance_from_log_ratio(lr).mean().item()
        rows.append(row)

    result = {
        "commit": git_commit(),
        "n_token_decisions": n,
        "exact_acceptance": exact_rate,
        "fraction_underflowing_float32_in_cspd_arithmetic": underflow,
        "measured_fixed_term_exponent": {
            "mean": float(fixed_exponent.mean()),
            "std": float(fixed_exponent.std()),
            "abs_median": float(fixed_exponent.abs().median()),
            "abs_p95": float(np.percentile(fixed_exponent.abs().numpy(), 95)),
            "max_abs": float(fixed_exponent.abs().max()),
        },
        "sweep": rows,
    }

    print(f"\nexact (lossless) acceptance      : {exact_rate:.4f}")
    print(f"tokens CSpD's own arithmetic zeroes: {underflow:.1%}")
    fe = result["measured_fixed_term_exponent"]
    print(f"measured fixed_term exponent      : mean {fe['mean']:+.3f}  "
          f"|median| {fe['abs_median']:.3f}  |p95| {fe['abs_p95']:.3f}  |max| {fe['max_abs']:.3f}")
    print(f"\n{'T':>6} {'no fixed_term':>15} {'as shipped':>12}")
    for row in rows:
        print(f"{row['temperature']:>6.0f} {row['no_fixed_term']:>15.4f} {row['as_shipped']:>12.4f}")

    out = Path(args.out) / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture", help="one GPU generation, recording the acceptance inputs")
    c.add_argument("--draft", default="base", choices=["base", "large", "huge"])
    c.add_argument("--target", default="large", choices=["base", "large", "huge"],
                   help="demo uses 'huge'; 'large' is the low-memory fallback")
    c.add_argument("--num-images", type=int, default=4)
    c.add_argument("--num-iter", type=int, default=64)
    c.add_argument("--num-draft", type=int, default=4)
    c.add_argument("--ratio", type=float, default=0.05)
    c.add_argument("--cfg", type=float, default=4.0)
    c.add_argument("--cfg-schedule", default="constant")
    c.add_argument("--diff-steps", default="100")
    c.add_argument("--inner-temperature", type=float, default=42.0,
                   help="only affects which tokens this trajectory accepts; the sweep is offline")
    c.add_argument("--out", default="experiments/E0.3")
    c.set_defaults(func=do_capture)

    a = sub.add_parser("analyze", help="sweep temperature over a capture; no GPU")
    a.add_argument("--capture", default="experiments/E0.3/capture.pt")
    a.add_argument("--out", default="experiments/E0.3")
    a.set_defaults(func=do_analyze)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
