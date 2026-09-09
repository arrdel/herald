#!/usr/bin/env python
"""E0.7 -- the draft/target separation of two real image models, measured by sampling.

The dimension sweep in this paper is a statement about ``M = ||mu_p - mu_q|| / sigma``. Until now
``M`` had never been measured for a real image-model pair, so the sweep said what happens as a
function of ``d`` without saying where a published pair sits on it. This measures it for MAR-B
against MAR-H at the KL16 token width ``d = 16``.

Why sampling rather than reading the head's parameters. MAR's per-token density is a diffusion head,
so there is no closed-form ``(mu, sigma)`` to read off: the only Gaussian in sight is the final
denoising transition, whose standard deviation is ~0.006 and whose draw is deterministic, and which
therefore describes the last step of the trajectory rather than the token distribution. (Measuring
``M`` from those final-step Gaussians gives ~138, which is a fact about the degeneracy of that step,
not about the two models -- see E0.6.) The predictive distribution a speculative verifier compares is
the trajectory marginal, and the honest way to reach it is to draw from it.

Protocol. One MAR-H generation provides the trajectory. At selected autoregressive steps the *same*
partially generated token map is fed to both models, which is exactly the speculative-decoding
setting of a shared accepted prefix, and each model's head is then drawn from ``n_draws`` times per
masked position. Per-coordinate means and standard deviations follow, and with them ``delta``,
``sigma``, ``M``, and the acceptance Proposition 2 predicts.

Example:
    CUDA_VISIBLE_DEVICES=7 python scripts/run_e07.py --cfg 3.0 --num-images 2 --n-draws 64
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
from scipy.stats import norm

from herald.models.baselines import REPO_ROOT, baseline_imports
from herald.models.mar_adapter import load_mar


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def draw_head_samples(model, z: torch.Tensor, n_draws: int, cfg: float, n_pos: int) -> torch.Tensor:
    """``n_draws`` independent draws from the diffusion head at each of ``n_pos`` conditionings.

    ``z`` is what ``sample_tokens`` hands to ``diffloss.sample``: ``(n_pos, D)`` without guidance,
    or ``(2*n_pos, D)`` with the conditional rows first and the null-class rows second. Replicating
    along the row axis keeps each replica's noise independent, because ``diffloss.sample`` draws one
    noise vector per row, so this is the same distribution the sampler would produce by running
    ``n_draws`` separate generations.
    """
    if cfg == 1.0:
        z_rep = z.repeat(n_draws, 1)
    else:
        z_rep = torch.cat([z[:n_pos].repeat(n_draws, 1), z[n_pos:].repeat(n_draws, 1)], dim=0)
    out = model.diffloss.sample(z_rep, 1.0, cfg)
    if cfg != 1.0:
        out, _ = out.chunk(2, dim=0)
    return out.reshape(n_draws, n_pos, -1)


@torch.no_grad()
def run(args) -> dict:
    device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("E0.7 needs a GPU")

    draft = load_mar("base", num_sampling_steps=args.diff_steps, device=device).eval()
    target = load_mar("huge", num_sampling_steps=args.diff_steps, device=device).eval()
    print(f"draft  MAR-B, target MAR-H, cfg={args.cfg}, {args.n_draws} draws/position")

    with baseline_imports(REPO_ROOT / "baselines" / "mar"):
        import models.mar as mar_module
        mask_by_order = mar_module.mask_by_order

    import math

    bsz, num_iter = args.num_images, args.num_iter
    torch.manual_seed(args.seed)
    labels = (torch.arange(bsz) % 1000).to(device)

    # Trajectory state, mirroring MAR.sample_tokens.
    mask = torch.ones(bsz, target.seq_len, device=device)
    tokens = torch.zeros(bsz, target.seq_len, target.token_embed_dim, device=device)
    orders = target.sample_orders(bsz)

    capture_steps = set(np.linspace(0, num_iter - 1, args.n_capture).astype(int).tolist())
    records: list[dict] = []
    t0 = time.time()

    def conditioning(model, tokens_in, mask_in):
        """``z`` for every position, from the state as it stands before the mask advances."""
        class_emb = model.class_emb(labels)
        tk, ce, mk = tokens_in, class_emb, mask_in
        if args.cfg != 1.0:
            tk = torch.cat([tokens_in, tokens_in], dim=0)
            ce = torch.cat([class_emb, model.fake_latent.repeat(bsz, 1)], dim=0)
            mk = torch.cat([mask_in, mask_in], dim=0)
        x = model.forward_mae_encoder(tk, mk, ce)
        return model.forward_mae_decoder(x, mk)

    def select(z, mask_to_pred, n_pos, idx=None):
        """Rows of ``z`` at the predicted positions, keeping the cfg row blocks aligned."""
        sel = mask_to_pred if args.cfg == 1.0 else torch.cat([mask_to_pred, mask_to_pred], dim=0)
        z = z[sel.nonzero(as_tuple=True)]
        if idx is None:
            return z
        if args.cfg == 1.0:
            return z[idx]
        return torch.cat([z[:n_pos][idx], z[n_pos:][idx]], dim=0)

    for step in range(num_iter):
        cur_tokens = tokens.clone()
        mask_now = mask                       # the state both models must condition on
        z_t = conditioning(target, tokens, mask_now)

        mask_ratio = np.cos(math.pi / 2.0 * (step + 1) / num_iter)
        mask_len = torch.Tensor([np.floor(target.seq_len * mask_ratio)]).to(device)
        mask_len = torch.maximum(
            torch.Tensor([1]).to(device),
            torch.minimum(torch.sum(mask_now, dim=-1, keepdims=True) - 1, mask_len),
        )
        mask_next = mask_by_order(mask_len[0], orders, bsz, target.seq_len)
        if step >= num_iter - 1:
            mask_to_pred = mask_now.bool()
        else:
            mask_to_pred = torch.logical_xor(mask_now.bool(), mask_next.bool())
        mask = mask_next

        n_pos = int(mask_to_pred.sum())
        z_t_sel = select(z_t, mask_to_pred, n_pos)

        if step in capture_steps and n_pos > 0:
            # Subsample positions so the cost per capture step is bounded.
            keep = min(n_pos, args.max_positions)
            idx = torch.randperm(n_pos, device=device)[:keep]
            z_tt = select(z_t, mask_to_pred, n_pos, idx)
            z_dd = select(conditioning(draft, tokens, mask_now), mask_to_pred, n_pos, idx)

            s_t = draw_head_samples(target, z_tt, args.n_draws, args.cfg, keep)
            s_d = draw_head_samples(draft, z_dd, args.n_draws, args.cfg, keep)
            records.append({
                "step": step,
                "target_mean": s_t.mean(0).cpu(), "target_std": s_t.std(0).cpu(),
                "draft_mean": s_d.mean(0).cpu(), "draft_std": s_d.std(0).cpu(),
            })
            print(f"  step {step:3d}: {keep} positions, {time.time()-t0:6.1f}s elapsed", flush=True)

        # Advance the trajectory with the target's own draw, so this is a valid MAR-H generation.
        drawn = target.diffloss.sample(z_t_sel, 1.0, args.cfg)
        if args.cfg != 1.0:
            drawn, _ = drawn.chunk(2, dim=0)
        cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = drawn
        tokens = cur_tokens.clone()

    tm = torch.cat([r["target_mean"] for r in records]).double().numpy()
    ts = torch.cat([r["target_std"] for r in records]).double().numpy()
    dm = torch.cat([r["draft_mean"] for r in records]).double().numpy()
    ds = torch.cat([r["draft_std"] for r in records]).double().numpy()

    gap = tm - dm
    sd_bar = np.sqrt(0.5 * (ts**2 + ds**2))
    z = gap / sd_bar
    M = np.linalg.norm(z, axis=1)
    predicted = 2.0 - 2.0 * norm.cdf(M / 2.0)

    rng = np.random.default_rng(0)
    bidx = rng.integers(0, len(predicted), size=(10_000, len(predicted)))
    ci = (float(np.percentile(predicted[bidx].mean(1), 2.5)),
          float(np.percentile(predicted[bidx].mean(1), 97.5)))

    res = {
        "commit": git_commit(), "host": platform.node(),
        "gpu": torch.cuda.get_device_name(0),
        "draft": "mar_base", "target": "mar_huge",
        "cfg": args.cfg, "n_draws": args.n_draws, "diff_steps": args.diff_steps,
        "num_images": bsz, "num_iter": num_iter, "seed": args.seed,
        "n_positions": int(len(M)), "token_dim": int(tm.shape[1]),
        "seconds": time.time() - t0,
        "per_coordinate_gap_over_sigma": {
            "mean": float(np.abs(z).mean()), "median": float(np.median(np.abs(z))),
            "p95": float(np.percentile(np.abs(z), 95)),
        },
        "sigma_target_mean": float(ts.mean()), "sigma_draft_mean": float(ds.mean()),
        "separation_M": {
            "mean": float(M.mean()), "median": float(np.median(M)),
            "p05": float(np.percentile(M, 5)), "p95": float(np.percentile(M, 95)),
        },
        "M_over_sqrt_d": float((M / np.sqrt(tm.shape[1])).mean()),
        "acceptance_predicted_gaussian": {"mean": float(predicted.mean()), "ci95": ci},
        "acceptance_at_mean_M": float(2.0 - 2.0 * norm.cdf(M.mean() / 2.0)),
        "shared_covariance_check": {
            "mean_abs_log_sd_ratio": float(np.abs(np.log(ts / ds)).mean()),
            "median_sd_ratio": float(np.exp(np.median(np.log(ts / ds)))),
        },
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"results_cfg{args.cfg}.json").write_text(json.dumps(res, indent=2))

    pc, sm = res["per_coordinate_gap_over_sigma"], res["separation_M"]
    print(f"\n{res['n_positions']} positions, d={res['token_dim']}, cfg={args.cfg}")
    print(f"sigma: target {res['sigma_target_mean']:.4f}  draft {res['sigma_draft_mean']:.4f}")
    print(f"per-coordinate |delta|/sigma : mean {pc['mean']:.4f}  median {pc['median']:.4f}")
    print(f"separation M                 : mean {sm['mean']:.3f}  median {sm['median']:.3f}"
          f"  [p5 {sm['p05']:.2f}, p95 {sm['p95']:.2f}]")
    print(f"M/sqrt(d)                    : {res['M_over_sqrt_d']:.4f}")
    ap = res["acceptance_predicted_gaussian"]
    print(f"predicted acceptance         : {ap['mean']:.4f}  "
          f"95% CI [{ap['ci95'][0]:.4f}, {ap['ci95'][1]:.4f}]")
    print(f"wrote {out / f'results_cfg{args.cfg}.json'}")
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-images", type=int, default=2)
    p.add_argument("--num-iter", type=int, default=64)
    p.add_argument("--n-draws", type=int, default=64)
    p.add_argument("--n-capture", type=int, default=8)
    p.add_argument("--max-positions", type=int, default=16)
    p.add_argument("--cfg", type=float, default=3.0)
    p.add_argument("--diff-steps", default="100")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="experiments/E0.7-mar-separation")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
