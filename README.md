# HERALD

**H**igh-throughput **E**mbedding-space **R**isk-controlled **A**cceleration of **L**atent **D**iffusion

Speculative sampling for continuous-token generative models, where drafting and verification
both happen in the JEPA embedding space rather than in data-token space.

---

## The one-paragraph version

Continuous autoregressive generators (MAR, D-JEPA) beat their discrete-token counterparts on
image quality, but they are slow: every token requires a full diffusion/flow trajectory through
a denoising head. Speculative decoding is the standard fix, and it works beautifully for discrete
LLMs — but the discrete machinery (softmax ratios, top-*m* sets, tree verification) does not
survive the move to continuous space. The single existing continuous method, CSpD, has to
introduce an *inner temperature of 42* into its acceptance ratio to get any acceptance at all.
We claim this is not an implementation detail but a symptom: **exact speculative verification in
continuous space is defeated by dimensionality**, and the field has been hiding this behind a
relaxation that silently voids the losslessness guarantee. HERALD proposes a different contract.
Instead of pretending to preserve the target's density over 16-dimensional VAE tokens, we
preserve a *learned low-dimensional semantic statistic* of the output, and we back it with a
finite-sample, distribution-free guarantee instead of an asymptotic one. The machinery to do
this is already sitting unused inside every D-JEPA checkpoint.

## The gap, stated precisely

D-JEPA trains three components jointly: a context encoder, an **EMA target encoder**, and a
**feature predictor** whose output `z` is supervised by *both* a diffusion loss and a JEPA
prediction loss against the target encoder's embeddings.

At inference time, `sample_tokens()` uses `z` only as conditioning for the denoising MLP. The
target encoder is never called. The JEPA loss is never evaluated.

> **D-JEPA trains a verifier and then discards it at inference.
> We claim it is exactly the verifier that speculative sampling needs.**

This gives two things for free, with no new parameters and no new training:

| Role | What we reuse | Why it is free |
| --- | --- | --- |
| **Drafter** | the predictor output `z` | Already computed once per iteration for *all* masked positions, in a single parallel transformer pass. The expensive part is the per-token denoising head that consumes it. |
| **Verifier** | the EMA target encoder + JEPA loss | Already trained to score exactly the quantity "does this token's embedding match what the predictor expected?" Its training-time residual distribution is a ready-made calibration set. |

## Three claims, in order of how much we believe them

**C1 (diagnostic, high confidence).** Acceptance probability under exact continuous speculative
decoding decays sharply with token dimension *d*. CSpD's reported speedups depend on
`inner_temperature=42.` and `ratio=0.15`; with the temperature set to 1 (the genuinely lossless
setting) acceptance collapses. This is measurable in an afternoon and is publishable on its own
as a negative result about the current state of continuous speculative decoding.

**C2 (mechanism, medium confidence).** Verifying a *k*-dimensional semantic projection
(k ≈ 64–768) instead of the *d*-dimensional token restores high acceptance, because the density
ratio concentrates far more slowly in the projected space. The JEPA predictor supplies both the
projection and the reference distribution.

**C3 (method, to be earned).** HERALD achieves ≥2× fewer NFE than CSpD at matched FID/CMMD,
while carrying a *stronger* guarantee than CSpD actually delivers: distribution-free,
finite-sample control of semantic deviation via conformal risk control, rather than an exactness
claim that the temperature hack invalidates.

### What would falsify this

- C1 fails if acceptance at temperature 1 is already high — then CSpD is genuinely lossless and
  the motivation evaporates. **Check this first, before writing any method code.**
- C2 fails if acceptance in semantic space does not improve over token space at matched
  guarantee strength, or if the improvement is entirely explained by the projection discarding
  the information that mattered.
- C3 fails if few-step distillation (consistency models, shortcut models) dominates on the
  speed/quality frontier, making speculation the wrong tool. This is the most likely failure
  mode and we benchmark against it from day one rather than at rebuttal time.

## Why the guarantee change is a feature, not a retreat

Speculative decoding's selling point is losslessness. Weakening it needs justification, so:

1. The exact guarantee is over the *VAE token* distribution, which is already an arbitrary
   lossy encoding of the image. Exactness with respect to it is not exactness with respect to
   anything a user cares about.
2. CSpD does not actually deliver the exact guarantee it advertises (see C1).
3. Conformal risk control gives a *finite-sample* statement with no distributional assumptions,
   whereas the classical acceptance rule's guarantee is exact only when the densities are exact —
   which for a learned diffusion head they are not.

So the honest comparison is not "exact vs. approximate" but "an unverifiable exactness claim about
an uninteresting space vs. a verifiable coverage claim about a meaningful one."

## Status

Pre-experimental. Nothing has been run yet. See `docs/RESEARCH_PLAN.md` for the phase plan and
`docs/EXPERIMENT_LOG.md` for results as they land. `docs/RELATED_WORK.md` records what is already
taken, with verification notes.

## Setup

```bash
conda activate herald
pip install -e ".[dev]"
```

Baselines are cloned under `baselines/` (git-ignored):

```bash
bash scripts/setup_baselines.sh     # clone repos
bash scripts/download_checkpoints.sh # MAR + D-JEPA weights, KL16 VAE
```

Hardware assumed: 8× RTX A6000 (48 GB each), driver 530 / CUDA 12.1.
Every phase-1 and phase-2 experiment is inference-only on public checkpoints.

## Layout

```
herald/
├── drafters/      predictor-as-drafter, few-step flow drafts
├── verifiers/     token-space (CSpD-exact) and semantic-space acceptance rules
├── calibration/   conformal risk control, nonconformity scores
├── models/        adapters onto MAR / D-JEPA checkpoints
├── metrics/       FID, CMMD, NFE accounting, acceptance-rate instrumentation
└── data/          ImageNet-256 latents, time-series testbeds
docs/              plan, related work, running experiment log
baselines/         upstream repos, unmodified (git-ignored)
experiments/       one directory per run, config + results + notes
```

## Testbeds

**Primary — ImageNet 256×256, inference-only.** MAR-B/L/H and D-JEPA-B/L/H are public. This is
where the headline numbers live and where CSpD is directly comparable.

**Secondary — time-series (ETTh1, UCI Electricity).** Trainable from scratch in hours, so it is
the only place we can sweep token dimension *d* with full control. C1's dimension-scaling curve
comes from here; ImageNet only gives us two or three points on it.

## Key references

Verified against the actual papers, not summaries. Full notes in `docs/RELATED_WORK.md`.

- D-JEPA — [arXiv:2410.03755](https://arxiv.org/abs/2410.03755), ICLR 2025. The predictor and
  target encoder we repurpose.
- CSpD — [arXiv:2411.11925](https://arxiv.org/abs/2411.11925). The only continuous speculative
  decoding method; our direct baseline and the target of C1.
- MAR — [arXiv:2406.11838](https://arxiv.org/abs/2406.11838). Diffusion loss for continuous AR;
  D-JEPA and CSpD both build on this codebase.
- ARC-Decode — ICML 2026. Conformal risk control for speculative decoding, **discrete LLMs only**.
  Closest prior art on the guarantee axis; differentiated by space, not by tool.
- SPLICE — [arXiv:2605.00126](https://arxiv.org/abs/2605.00126). JEPA latents + flow matching +
  adaptive conformal inference for time-series. Our secondary testbed inherits its setup.
- Beyond Generative Priors — [arXiv:2605.24631](https://arxiv.org/abs/2605.24631). JEPA as a
  density proxy for guidance; evidence that JEPA embeddings carry usable density information.
