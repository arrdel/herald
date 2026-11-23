# Experiment Log

Append-only. One entry per run, newest last. Record what was run, what came out, and what it
changes — including runs that failed or contradicted the plan. **Negative results are logged with
the same care as positive ones**; the value of the log is that it is trustworthy.

Template:

```
## <date> — <experiment id> — <one-line result>
**Ran:** command / config / commit
**Setup:** model, data, hardware, seeds
**Result:** numbers, with the quality metric attached to any speed claim
**Reading:** what it means
**Changes:** what in the plan this confirms, kills, or reroutes
```

---

## 2026-08-29 — E-pre.1 — Synthetic probe supports C1; CSpD's temperature=42 accepts 99.9% of drafts from a model that visibly disagrees

**Ran:** `tests/test_verifiers.py::test_acceptance_decays_with_dimension`, plus an inline sweep
over `herald.verifiers.TokenSpaceVerifier`. Pure synthetic Gaussians, no models loaded.

**Setup:** Draft `N(0, I_d)` vs. target `N(0.5·1, I_d)` — a fixed per-dimension mean shift of
half a standard deviation. 20k samples per cell. Acceptance is the classical
`min(1, p(x)/q(x))` under `x ~ q`.

**Result:**

Acceptance vs. token dimension, at temperature 1 (the genuinely lossless setting):

| d | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| acceptance | 0.800 | 0.723 | 0.615 | 0.475 | 0.316 | 0.156 | 0.044 |

Acceptance vs. CSpD's `inner_temperature`, at d=16, **with the draft and target models held
fixed**:

| temperature | 1 | 2 | 5 | 10 | 20 | 42 |
| --- | --- | --- | --- | --- | --- | --- |
| acceptance | 0.316 | 0.631 | 0.920 | 0.979 | 0.995 | **0.999** |

**Reading:** Both halves of the C1 argument show up cleanly in the idealized case. Acceptance
decays roughly geometrically in *d* because each dimension contributes an independent factor to
the density ratio — so a per-dimension disagreement that is individually mild compounds. And the
temperature knob moves acceptance from 32% to 99.9% **without changing the models at all**, which
is the point: a high acceptance rate at temperature 42 carries almost no information about
whether draft and target agree.

The KL between these two distributions at d=16 is 2.0 nats, which is a real disagreement, and
the tempered rule accepts it 999 times in 1000.

**Caveats, stated plainly.** This is synthetic. Real MAR draft/target pairs are far more similar
than a 0.5σ shift, so the temperature-1 acceptance rate on real models could be much higher than
0.316 and C1 could still fail. Isotropic Gaussians with a uniform shift are also the most
favorable case for the dimensional argument; correlated tokens would compound more slowly.
Nothing here substitutes for E0.3.

**Changes:** No change to the plan — this is what the plan predicted, which is mild evidence the
plan is worth executing. Confirms E0.3 is correctly identified as the fulcrum and worth running
first. It does *not* license writing method code yet.

---

## 2026-09-01 — E0.1 (partial) — The denoising head is 98.3% of wall time while doing 4.5× *fewer* token evaluations than the transformer; the plan's "step 2 dominates" is right about time and wrong about volume

**Ran:** `python scripts/run_e01.py --size base --num-images 8 --batch-size 4 --num-iter 256
--save-grid 8 --out experiments/E0.1-mar-base` at commit `9192abf`. Config and raw numbers in
`experiments/E0.1-mar-base/results.json`.

**Setup:** MAR-B (208M, matching the published parameter count on a strict `load_state_dict`),
public `mar-base.safetensors` + KL16, ImageNet 256. Upstream evaluation settings: `num_iter=256`,
`cfg=2.9`, `cfg_schedule=linear`, `temperature=1.0`, autocast on. Seed 0, classes 0–7. One RTX
A6000, **shared with another user's job holding the GPU at 100% utilization**. 8 images generated,
4 timed (first batch discarded as warmup, as upstream does).

**Result:**

Wall-clock, per component:

| component | seconds | % of time | token-evals | calls |
| --- | --- | --- | --- | --- |
| diffusion_head | 363.7 | **98.3%** | 204,800 | 256 |
| decoder | 3.6 | 1.0% | 392,192 | 256 |
| encoder | 2.7 | 0.7% | 524,288 | 256 |

Throughput 93.25 s/image. Head/transformer token-evals = **0.22**. Per token evaluation the head
costs roughly 260× more wall time than the transformer.

No FID. The reproduction gate is **not** met — see below.

**Reading:** The free-drafter premise survives on the axis that matters: the transformer produces
`z` for every masked position using 1.7% of the run, so reading a tensor that is already computed
really is close to free. But the *mechanism* stated in section 1 of the plan is wrong. The plan
argues the head dominates because it "runs 64 × ~100 × (tokens per step) times", which is a claim
about volume, and by volume the head is the *smaller* consumer — the transformer does 4.5× more
token evaluations. The head dominates because its 100 denoising steps are **sequential**, executed
at a batch of one or two tokens at `num_iter=256`, whereas the transformer does all 320 positions
in one batched pass. This is a latency/utilization effect, not an arithmetic one.

That distinction has teeth for how HERALD may claim a speedup. An NFE reduction on the head is not
automatically a proportional wall-time win, and a reviewer who counts token evaluations rather than
seconds will find the transformer is the larger consumer. Speed claims should therefore be made in
wall-time and in *sequential head invocations*, with raw NFE reported alongside rather than as the
headline. The standing rule about attaching quality metrics to speed claims now needs a companion:
say which cost model a speedup is measured in.

Suggestive but confounded: the earlier smoke run at `num_iter=32` reached 39.8 s/image against
93.25 s/image at `num_iter=256`, for the same 25,600 head evaluations per image. That is consistent
with batching inside the head being the dominant lever, but the two runs used batch size 2 and 4
respectively, so it is not a clean measurement. Worth isolating before it is used for anything.

**Caveats, stated plainly.** The GPU was contended throughout, so absolute wall-times are not
usable as throughput numbers; the percentage split is more robust because both stages contend
equally, but it is not immune either. Four timed images is enough for a deterministic property of
the sampling loop and nothing else. This entry says nothing about output quality: the samples are
class-faithful on visual inspection (`experiments/E0.1-mar-base/samples.png`) and that is not a
quality metric.

**Two instrumentation bugs found and fixed**, both of which would have produced plausible wrong
numbers rather than obvious ones:

1. `instrument_mar_sampler` wrapped only `forward_encoder`/`forward_decoder`, which is D-JEPA's
   naming. MAR calls them `forward_mae_encoder`/`forward_mae_decoder`, so on MAR the transformer
   was never recorded at all and the head would have appeared to be 100% of everything.
2. The head's NFE ignored the ~100 denoising steps inside each `diffloss.sample` call, understating
   it by two orders of magnitude. The step count is now read off `gen_diffusion.num_timesteps` and
   the code refuses to guess it.

A third, smaller correction: transformer inputs were counted in *sequences* while the head was
counted in *token-diffusion-steps*, which made the ratio meaningless (it read as 400× in the smoke
run). Both are now in token evaluations, which is what produced the 0.22 above.

**Changes:**
- Section 1 of `RESEARCH_PLAN.md` updated: the head's dominance is serialization, not volume.
- E0.1 is **not** complete. The FID-50K gate needs 50k samples; at the measured ~93 s/image on a
  contended GPU that is roughly 54 GPU-hours for MAR-B alone, so it is queued for when the machine
  frees up. The runner refuses to report FID below 10k images rather than emit a small-sample
  number that would look like a failed reproduction.
- No change to E0.3's priority. It remains the fulcrum and is still unrun.

---

## 2026-09-02 — E0.3a — CSpD's acceptance rule has a *second* defect, independent of the temperature, and the schedule alone bounds it at e^81

> **Partly superseded the same day — see "E0.3a (corrected)" below.** The characterisation of the
> second defect in this entry is wrong: the accumulated-variance factor *does* belong in the rule
> (CSpD shares one noise draw, so it verifies a trajectory), and the actual defect is that CSpD
> applies its **reciprocal**. The magnitudes below are unaffected; the error exponent doubles.
> Left unedited because a log that quietly rewrites its own mistakes is not a log.

**Ran:** `python scripts/analyze_cspd_fixed_term.py`. Zero GPU, zero model weights — the result
follows from CSpD's diffusion schedule and its own source. Raw numbers in
`experiments/E0.3a-fixed-term/results.json`. Source reading recorded in `docs/RELATED_WORK.md`;
faithful reimplementation and decomposition in `herald/verifiers/cspd_exact.py`.

**Setup:** `create_diffusion(timestep_respacing="100", noise_schedule="cosine")` as `DiffLoss`
builds it, KL16 token dimension 16.

**Result:**

First, the call path for the known defect is now *verified* rather than inferred, which was an
open caveat: `SpecDecodeRunner.run` → `DistributionProcessor(temperature=42.)` →
`speculative_sample` → `judge_accept` → both `x1_pdf_unormalize` and `x2_pdf_unormalize`, each
dividing the standardized residual by the temperature. There is no proposal-only reading.

Second, and new: `judge_accept` multiplies the ratio by
`fixed_term = exp((x2_log_var - x1_log_var).sum(-1))`, which does not belong in the density ratio
of the two final-step Gaussians. Tracing what fills those fields through
`gaussian_diffusion.py::p_sample_loop_progressive_with_noise`, `x*_log_var` is **not** a
log-variance — it is `Σ log σ` accumulated over every denoising step except the last. So the factor
is a product over 16 channels × 99 steps = **1,584 terms**.

Because `learn_sigma=True` (`ModelVarType.LEARNED_RANGE`), each step's `log σ` is a model-predicted
interpolation between `posterior_log_variance_clipped` and `log β`, so the achievable range is
fixed by the schedule:

| quantity | value |
| --- | --- |
| per-step \|log σ\| spread, median | 0.0159 |
| per-step \|log σ\| spread, max | 1.4301 |
| summed over the 99 accumulated steps | 5.05 |
| worst-case exponent (16 channels) | **80.8**, i.e. `fixed_term` up to e^81 ≈ 10^35 |
| exponent at an assumed 5% fraction gap | 4.0, i.e. `fixed_term` ≈ 57× |
| exponent needed to merely double acceptance | 0.693 |

On synthetic Gaussians an exponent of 4.0 moves acceptance from 0.32 to 0.91 **at temperature 1**,
with the two final-step distributions untouched.

**Reading:** The C1 story is stronger and more interesting than "they set a temperature to 42."
There are two independent departures from the exact rule, and the arithmetic says the *undocumented*
one may be the larger. A 5% disagreement between draft and target on their predicted variance
fraction — utterly unremarkable between two differently-sized networks — is enough to inflate the
acceptance ratio ~57×, which saturates `min(1, ratio)` for most tokens before the temperature does
anything at all. The temperature is visible in the source and has at least a plausible reading as a
deliberate relaxation. This one reads as a bug.

A third, smaller finding: `judge_accept` composes raw `torch.exp` calls in linear space, and at
d=16 both numerator and denominator routinely underflow float32 to zero, giving `0/0`. Anyone
measuring temperature-1 acceptance using upstream's own arithmetic risks reporting a numerical
artifact as a dimensional effect. Our harness works in log space and flags affected tokens.

**Caveats, stated plainly.** The 80.8 is a **bound**, not a measurement: it assumes the two models
pin their fractions to opposite ends of the range at every step and channel. The 57× uses an
**assumed** 5% gap and is explicitly labelled as such in the output — it is not measured, and it is
exactly what E0.3 replaces. The factor is signed, so depending on which model predicts larger
variances it can deflate acceptance as easily as inflate it; the direction for a real MAR-B/MAR-H
pair is unknown. Nothing here establishes that CSpD's *reported* speedups are wrong, only that the
quantity its acceptance test computes is not the density ratio its guarantee is stated about.

**Reproducibility note (verified).** `SpecDecodeRunner` is never imported by `main_mar.py`; its only
use in the repo is `demo/run_mar.ipynb`. There is no evaluation entry point, FID script, or
throughput benchmark for the speculative path, so E0.2 cannot "reproduce" the 2.33× — the harness
has to be written first. The demo configuration is draft `mar_base`, target `mar_huge`,
`ratio=0.05` (class default is 0.15), `num_draft=4`, `cfg=4.0`, `cfg_schedule="constant"`.

**Changes:**
- C1 is now a two-part claim: exact continuous acceptance is defeated by dimensionality, *and* the
  only published continuous method does not implement the exact rule in two separate ways.
- E0.3 redesigned to be GPU-frugal (see plan): capture the acceptance inputs from **one** generation
  run and sweep temperature offline, rather than one run per temperature. Six runs become one.
- E0.3 gains a deliverable: measure the real `frac_gap` and the real `fixed_term` distribution, and
  report the four-way decomposition (`exact`, `temperature_only`, `fixed_term_only`, `as_shipped`).
- E0.2's framing changes from "reproduce" to "build the missing harness, then measure".

---

## 2026-09-02 — E0.3b — the closed form for continuous acceptance, and why it weakens our own C1

**Ran:** no experiment. A related-work sweep turned up a paper the 2026-08-29 sweep missed, and its
central formula forced a correction to the plan. Recorded here because it changes a claim, and a
claim changing is a result.

**Source:** *Accelerating Speculative Diffusions via Block Verification* (arXiv 2606.13426), read in
full. Its Eq. 45 gives expected acceptance for Gaussian sample verification in closed form:
`E_q[min(1, p/q)] = 1 − TV = 2 − 2Φ(M/2)`, `M = ‖m_p − m_q‖/σ`. Implemented as
`analytic_acceptance_shared_covariance` and cross-validated against our own `exact_log_ratio` by
200k-sample Monte Carlo at four (d, gap) settings, agreeing to 5e-3.

**Result.** With per-coordinate gap `δ = ‖m_p − m_q‖/(σ√d)`, `M = δ√d`. At δ=0.25σ acceptance runs
0.90 / 0.62 / 0.32 / 0.05 for d = 1 / 16 / 64 / 256. At MAR's d=16: δ=0.1σ → 0.84, δ=0.25σ → 0.62,
δ=0.5σ → 0.32, δ=1.0σ → 0.05.

**Reading, and it goes against us.** C1 has been stated as "exact continuous speculative decoding is
defeated by dimensionality." At the dimension CSpD verifies in — d=16, one KL16 token — that is
overstated. Sixteen dimensions is not a curse. If MAR-B and MAR-H agree to within 0.1σ per
coordinate, exact acceptance is 84% and there was nothing to hide. The √d penalty bites at block or
image scale, not per token.

C1 therefore reduces to an empirical question about **δ**, which nobody has measured and which is
now the most important number in the project. E0.3 measures it directly. Two things survive
regardless: δ varies across the generation, so acceptance is a schedule and a single average will
hide where it fails; and E0.3a's finding is independent of δ, because CSpD's rule is not computing
this ratio at all.

**Scoop assessment.** The paper also publishes the acceptance collapse (in the σ→0 limit, for
diffusion-step speculation) and a training-free "Free Drafter". Neither kills HERALD, but both
constrain how it can be positioned: the collapse is no longer ours to discover, and the free-drafter
rhetoric now needs an explicit distinction (their drafter reuses the target's own score inside a
diffusion sampler; ours is D-JEPA's separately-trained predictor with the EMA encoder as verifier).
Temperature-as-lenience is also *not* novel — their Appendix I discusses it, citing Leviathan and
Tran-Thien (2023). What remains CSpD-specific is the undocumented magnitude and the arithmetic
defect. Full assessment in `docs/RELATED_WORK.md`.

**Process note.** The 2026-08-29 sweep missed a June 2026 paper that is arguably the closest work in
the field, because the query set was scoped to continuous-token AR and this is diffusion-step
speculation. Widened going forward.

**Changes:** new §1b in the plan with the revised C1 and its falsifier; `δ turns out to be small`
added as the top risk; scoop risk marked partly realised. Transform-domain verification for E1.2b
searched and provisionally clear — nothing found putting a wavelet/DCT statistic inside a
speculative acceptance criterion.

---

## 2026-09-02 — E0.5 — KL16's gain is near-isotropic within an image, but varies 3× *between* images, which is the bigger problem

**Ran:** `python scripts/run_e05.py --n-latents 4 --crop 256 --k 64 --oversample 16 --power-iters 1
--device cpu`. Results in `experiments/E0.5-kl16-jacobian/results.json`. On CPU at ~190 s/latent
because all eight GPUs were allocated elsewhere; the run needs ~1 GB and would take seconds on a
free card. Gate-exempt (frozen public VAE).

**Setup:** local Jacobian spectrum of the KL16 decoder via randomized SVD with matrix-free JVP/VJP,
top 64 of 4096 latent directions, at four latents obtained by re-encoding E0.1's MAR samples.

**Result:**

| latent | anisotropy (top-64) | effective rank | σ_max | σ_min |
| --- | --- | --- | --- | --- |
| 0 | 4.08 | 60.69 | 23.61 | 5.79 |
| 1 | 3.05 | 61.44 | 15.99 | 5.25 |
| 2 | 3.58 | 60.56 | 19.54 | 5.46 |
| 3 | 2.60 | 61.75 | 47.38 | 18.24 |

Median anisotropy 3.31, median effective rank 61.06 of 64. Linearization relative error (measured
finite-difference gain vs. the Jacobian's prediction, at ε=1e-2) median 0.054, max 0.076.

**Reading — the question asked.** Within a single latent, the decoder is close to isotropic over the
directions that matter: gain varies only ~3.3× across the top 64, and effective rank is 61 of 64, so
those directions carry near-equal response rather than one dominating. An isotropic threshold
`‖E_ema(x̂) − z‖ ≤ τ` is therefore a defensible first choice, and the Jacobian-weighted nonconformity
score can stay on the shelf. The linearization is also valid at the ε the method will see (~5%
error), so if we ever do need weighting, the Jacobian is a trustworthy basis for it.

**Reading — the question not asked, which matters more.** σ_max ranges 15.99 to 47.38 across four
images, a factor of 3.0. That is as large as the within-image anisotropy. The same latent-space
threshold τ therefore buys pixel deviations differing ~3× depending on which image is being
generated. Conformal risk control still delivers its guarantee, but the guarantee is **marginal over
images, not conditional on one** — some images systematically absorb larger pixel error than the
average implies. Worth deciding deliberately in Phase 2 rather than discovering in E2.5: options are
to normalize the score by a cheap per-image gain estimate, or to accept marginal validity and report
the spread honestly.

**Caveats.** Four latents is a pilot, and the between-image spread is exactly the quantity four
samples estimate worst — it needs more images before being leaned on. Randomized SVD recovers the
**top** of the spectrum, so "anisotropy 3.31" is a lower bound over the top 64 of 4096 directions
and says nothing about the remaining 4032; the full condition number could be far larger, and this
measurement cannot rule out a badly-attenuated tail. The latents come from re-encoding MAR's own
samples rather than real photographs, so they sit where the method operates but are not ImageNet.

**Changes:** none to the method. E0.5's design question is answered "isotropic is fine for now"; the
between-image gain spread is logged as a Phase-2 design item.

---

## 2026-09-02 — E1.1 — acceptance collapses as `d^0.52` while per-coordinate disagreement stays flat. The dimensional mechanism is real.

**Ran:** `python scripts/run_e11.py --dims 4 8 16 32 64 128 --tok-steps 2000 --mar-steps 4000
--device cpu`. ETTh1 (7 channels, 17,420 hours), window 96, patch 8 → 12 tokens. Per `d`: a patch
VAE, then a target MAR (width 256, depth 6) and a draft MAR (width 128, depth 4) on its latents,
architectures held fixed across the sweep so the capacity gap does not drift. 39 min on CPU, since
all eight GPUs remained allocated elsewhere. Results in `experiments/E1.1-dsweep/results.json`,
figure at `docs/figures/e11_acceptance_vs_d.png`.

**Result:**

| d | exact acceptance | δ (mean gap) | \|log σ_p/σ_q\| | equal-cov. analytic | M_eff | M_eff/√d |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 0.6420 | 0.129 | 0.065 | 0.898 | 0.93 | 0.465 |
| 8 | 0.4048 | 0.181 | 0.083 | 0.798 | 1.67 | 0.589 |
| 16 | 0.1761 | 0.225 | 0.101 | 0.653 | 2.71 | 0.676 |
| 32 | 0.0734 | 0.218 | 0.098 | 0.537 | 3.58 | 0.633 |
| 64 | 0.0170 | 0.195 | 0.102 | 0.436 | 4.78 | 0.597 |
| 128 | 0.0038 | 0.197 | 0.096 | 0.264 | 5.79 | 0.512 |

Acceptance falls **170×** from d=4 to d=128. Inverting each rate to an effective Mahalanobis
distance gives a power-law fit of **M_eff ∝ d^0.519**, against the √d law's predicted 0.500, with
M_eff/√d flat at 0.579 ± 0.071.

**Reading.** This is the plot C1 is built around, and it behaves as §1b predicts. The important
control is the second column: **δ is flat from d=16 onward** (0.225, 0.218, 0.195, 0.197), as is the
scale gap (~0.10). So the collapse across the upper half of the sweep is not the draft getting worse
at larger d — per-coordinate agreement is unchanged and acceptance still falls 46×. That is the
dimensional mechanism isolated, which acceptance alone could never have shown. Between d=4 and d=16
δ does grow, so the early decay mixes the two effects and should not be quoted as purely dimensional.

**The equal-covariance closed form is badly optimistic, and that strengthens C1.** Eq. 45 of
arXiv 2606.13426, evaluated at the measured δ, predicts 0.264 at d=128 where the truth is 0.0038 —
a factor of 70. The effective per-coordinate divergence is ~2.8× the mean gap alone (0.579 vs
0.209). The missing part is scale and shape: learned draft and target disagree on variance, not just
location, and a mean-gap-only formula cannot see it. Anyone estimating continuous acceptance from
mean differences will substantially overestimate it.

**At d=16 — MAR's actual token dimension — exact acceptance is 0.176.** That is the directly
C1-relevant number from this testbed, and it is low enough that a method would need to relax
something. It is not MAR, and it should not be quoted as if it were, but it is the first measured
evidence that δ is not small in the regime E0.3 will probe.

**Caveats, and one is serious.** *The training budget is fixed at 4,000 steps for every d.* Larger-d
models have more to learn on the same budget, so some of the decay could be relative undertraining
rather than dimension. The flat δ argues against this being the whole story — an undertrained draft
should disagree *more*, and δ does not rise — but it is not a proof. **Settled by E1.1b/E1.1c
below: the confound is refuted, and the penalty survives at matched δ.** Beyond it: one dataset,
one seed, one arbitrary capacity gap (256/6 vs 128/4); the tokenizer is refit per d, so the latent
distribution is not held fixed; and the head is a Gaussian mixture rather than a diffusion head, a
deliberate choice (exact densities are the point) that makes the absolute rates non-transferable to
MAR.

**Changes:** none to the method. C1's dimensional half now has direct empirical support in a
controlled setting, with the mechanism — not just the trend — verified. E0.3 remains the test that
matters, since it measures δ on real MAR pairs.

---

## 2026-09-02 — E1.1b / E1.1c — the training-budget confound is refuted, and at *matched* model agreement dimension still costs ~60×

**Ran:** two convergence traces on the E1.1 testbed at the extremes `d ∈ {16, 128}`, each training a
draft and a target and measuring acceptance at matched step counts from saved checkpoints.
**E1.1b**, the long run: `--mar-steps 16000 --checkpoint-every 2000` (4× E1.1's budget), 65 min CPU,
`experiments/E1.1b-convergence/`. **E1.1c**, the fine early trace: `--mar-steps 4000
--checkpoint-every 250 --threads 8`, 16 min CPU, `experiments/E1.1c-earlytrain/`. Offline analysis
in `scripts/analyze_e11b.py`; figures `docs/figures/e11b_convergence.png` and `e11c_earlytrain.png`.

**Result 1 — the confound is refuted.** E1.1 fixed the budget at 4,000 steps for every `d`, so the
worry was that large-`d` models were simply undertrained. They were not. Within E1.1's budget both
dimensions **plateau early**, `d=16` by ~750 steps and `d=128` by ~500:

| step | 250 | 500 | 750 | 1000 | 2000 | 3000 | 4000 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| acceptance, d=16 | 0.3403 | 0.2319 | 0.1981 | 0.1877 | 0.2003 | 0.1750 | 0.1793 |
| acceptance, d=128 | 0.0444 | 0.0048 | 0.0044 | 0.0050 | 0.0017 | 0.0019 | 0.0020 |

Undertraining at large `d` predicts `d=128` still climbing at 4,000 and the two curves converging.
Neither happens. The prediction was committed to in the script docstring before the run.

**Result 2 — past E1.1's budget acceptance keeps *falling*, because the models overfit apart.**
Over 2k→14k steps acceptance goes 0.1777 → 0.0714 at `d=16` and 0.0051 → 0.0007 at `d=128`, while
validation NLL *rises* throughout (target `d=16`: 13.5 → 26.3) and per-coordinate disagreement grows
(δ 0.227 → 0.342 at `d=128`). Draft and target overfit to *different* solutions, so training longer
makes them disagree more. E1.1's 4,000-step numbers were therefore, if anything, generous to
large `d` — the honest correction runs against us in the safe direction.

**Result 3 — the matched-δ comparison, which is the real prize.** Because δ drifts over training,
each dimension sweeps out a *range* of model agreement, so acceptance can be compared at δ held
equal rather than at budget held equal. Interpolating `d=128`'s acceptance (in log space, no
extrapolation) to each `d=16` checkpoint's δ:

| | pairs | δ range | d=16 → d=128 penalty |
| --- | --- | --- | --- |
| E1.1c (4k budget) | 15 | 0.19–0.29 | 16×–117×, **median 62×** |
| E1.1b (16k budget) | 7 | 0.26–0.33 | 55×–92×, **median 64×** |

Two independent runs at different budgets agree on ~60×. E1.1 could only argue this indirectly, from
δ happening to be flat across the sweep; here the matching is direct, and it is the cleanest
statement of the dimensional mechanism we have.

**Result 4 — at each model's own best-validation checkpoint**, acceptance is 0.2164 at `d=16`
(draft@1500, target@1000) and 0.0010 at `d=128` (draft@1500, target@2750). *This 216× should not be
quoted as the headline*: best-val selects **different budgets for draft and target**, and a mismatched
pair disagrees more for reasons that have nothing to do with dimension — note 0.0010 is below every
matched-step value in the `d=128` trace. The matched-δ ~60× is the defensible number.

**A defect found and fixed in our own instrument.** Writing the first test for `measure_acceptance`
revealed that its `seed` argument only seeded the mask: `MixtureSameFamily.sample()` takes no
generator, so the draw of `x` — and therefore the acceptance estimate — followed ambient global RNG
state. Repeat measurements of the same model pair differed by ~5e-4. Now forked and seeded.
**E1.1 was re-measured under the fixed sampler** (`run_e11.py --measure-only`, inference only):
every dimension moved within a few SEM, and the headline 170× d=4→d=128 collapse reads 181×. The
result stands; `experiments/E1.1-dsweep/results_remeasured.json` keeps both so they can be compared.

**Reading.** The one serious hole in E1.1 is closed, and closed in a way that strengthens it rather
than merely defending it: the dimensional penalty survives holding per-coordinate agreement fixed,
which is the control that distinguishes "acceptance falls with `d`" from "the draft is worse at
large `d`". C1's dimensional half is now supported by a direct matched comparison rather than by an
observation that a nuisance variable happened not to move.

**Caveats.** Still one dataset, one seed, one capacity gap, one tokenizer refit per `d` — E1.1's
structural caveats are untouched, and only the two extreme dimensions were traced. The plateau is
established within 4,000 steps but the models never reach a genuine optimum: validation NLL turns
over early and then degrades, so "converged" here means "stopped improving", not "well trained".
Matched-δ interpolation assumes acceptance depends on the checkpoint only through δ, which is not
exactly true — the scale gap moves too, and it moves in the same direction, so some of the 60× may
be scale rather than location disagreement. Finally, CPU training is not bit-reproducible across
thread counts: an independent `d=16` run at 8 threads gave 0.1731 at step 4000 against 0.1494 here.
The trend is robust to that; individual cells are not.

**Changes:** `measure_acceptance` moved into `herald/eval/` so E1.1, E1.1b, and the E1.2/E1.3 ladder
cannot drift apart, with the RNG fix and six tests. `TrainConfig.checkpoint_every` added. No change
to the method or to C1's statement.

---

## 2026-09-02 — E0.3a (corrected) — the accumulated-variance factor *belongs*; CSpD applies its reciprocal. Weaker claim, stronger position.

**Trigger.** A scoop-check sweep of the transform-domain verification literature (run to clear
E1.2b) turned up **Fast-ARDiff** (arXiv 2512.08537), which builds on CSpD and whose §S3 states the
continuous acceptance ratio as `p/q = Σ · p_θ(x₀|x₁ᵖ)/q_θ(x₀|x₁ᵍ)` with
`Σ = ∏_{t≥2} √(|Σ_q,t| / |Σ_p,t|)`, applying it in Algorithm 1 as part of the *correct* criterion.
That is an independent group endorsing the factor E0.3a called spurious. Both readings cannot be
right, so I settled it against the source rather than against search results.

**Ran:** no compute. Re-read `spec_decode_runner.py` and `gaussian_diffusion.py`, and derived the
trajectory ratio. New tests in `tests/test_cspd_exact.py` pin the result (77 tests pass).

**Result — Fast-ARDiff is right about the factor, and we were wrong.** CSpD draws `noise` and
`inter_noise` **once** and shares them between draft and target (`run()`, lines 101–103 — the same
fact E0.4's prerequisite rests on). What is being verified is therefore the whole denoising
trajectory, whose likelihood ratio is `∏_t p(x_{t-1}|x_t) / q(x_{t-1}|x_t)`. Each Gaussian
transition contributes a normalizer `(|Σ_q,t|/|Σ_p,t|)^{1/2} = ∏_c σ_q,t,c / σ_p,t,c`. A factor of
this shape belongs. E0.3a's "nothing calls for that factor" was simply wrong.

**But the defect is real, and now sharper: CSpD applies the factor upside down.**

| | accumulated factor |
| --- | --- |
| correct (trajectory ratio) | `∏_t (σ_q,t / σ_p,t)` |
| CSpD, `exp((x2_log_var − x1_log_var).sum(-1))` with x1=draft=q, x2=target=p | `∏_t (σ_p,t / σ_q,t)` |

The shipped factor is the reciprocal of the intended one, so **the acceptance ratio is wrong by
`fixed_term²`** and every error exponent in the original entry doubles: worst case 80.8 → **161.5**
(e^161), and the assumed-5%-gap case 4.04 → **8.08**, a ~3,200× error rather than 57×.

Two details make this a sign slip rather than a modelling choice, both visible in the one line:
the *same* normalizer for the final step appears in the *same* expression the right way round
(`x1_std.prod(-1) / x2_std.prod(-1)`, i.e. `σ_q,1/σ_p,1`), and the accumulated subtraction is
target-minus-draft where every other scale term is draft-over-target. Note also that
`gaussian_diffusion.py` line 582 returns `"log_variance": 0.5 * out["log_variance"]` — so the
accumulator holds `Σ log σ`, not `Σ log σ²`, which is why the power comes out right even though the
direction does not.

**Reading.** This is a weaker claim than the one it replaces and a much stronger position to defend.
"Upstream added a term with no place in the ratio" invites a one-line rebuttal from anyone holding
the Fast-ARDiff derivation; "upstream's term is correct in form and inverted in sign, and here is
the code, the derivation, and a test" does not. It also means the C1 story now has an independent
witness: Fast-ARDiff derives the criterion CSpD *should* implement, which makes the gap between
them a documented discrepancy rather than our interpretation. **This is exactly the failure mode the
project's own rule about reading source rather than abstracts is meant to catch — the trap here was
that we did read the source, but reconstructed the intended derivation from the final-step
Gaussians alone rather than from the shared-noise trajectory the code actually implements.**

**Caveats.** Fast-ARDiff's §S3 is a secondary source for CSpD's intent; the derivation above is
mine and does not depend on it, but I have not found the corresponding statement in the CSpD paper
itself. The bound remains a bound: 161.5 assumes both models pin their variance fractions to
opposite ends of the range at every step and channel. The factor is signed, so the inversion can
deflate acceptance as easily as inflate it, and the direction for a real MAR-B/MAR-H pair is still
unknown — that is E0.3's job.

**Changes:**
- `cspd_log_ratio` gains `fixed_term_sign`; `decompose_acceptance` gains a
  **`corrected_trajectory`** rule, so E0.3 will separate the sign error from the legitimate
  trajectory term instead of lumping them together. E0.3's deliverable is now a five-way
  decomposition.
- `fixed_term_magnitude` documents that the error exponent is twice the returned value.
- `RELATED_WORK.md`: E0.3a note corrected; **Fast-ARDiff added as a first-order competitor** —
  4.38×–4.88× on ImageNet 256 with FID preserved (MAR-H 4.47× at FID 1.61) is the frontier C3 must
  beat, in exactly our setting.
- C1's second half is restated: the only published continuous-token AR method departs from the exact
  rule in two undocumented ways — temperature 42, and an **inverted** trajectory normalizer.

---

## 2026-09-02 — E1.2a — acceptance is set by the *verified* dimension `k`, not the token dimension `d`. Six values of `d` collapse onto one curve.

**Ran:** `python scripts/run_e12a.py --dims 4 8 16 32 64 128 --n-projections 5 --eval-batches 40`.
**No training and no GPU** — this reuses E1.1's saved tokenizers and draft/target pairs and is pure
inference, 16 min on CPU. For each `d`, verification is applied to `A x` for a random `k × d` matrix
with orthonormal rows, `k` sweeping powers of two up to `d`, five independent draws of `A` per `k`.
The draft still proposes the *full* token; only its projection is checked. Machinery in
`herald/verifiers/projected.py`, results in `experiments/E1.2a-random-projection/`, figure at
`docs/figures/e12a_random_projection.png`.

**Correctness anchor, passed.** A square orthogonal `A` is a bijection with unit Jacobian, so at
`k = d` the projected test must reproduce raw verification *exactly*. It does, at every dimension:
0.6425 / 0.4069 / 0.1743 / 0.0736 / 0.0178 / 0.0035 against E1.1's re-measured values to four
decimals. Any acceptance gain at `k < d` is therefore the projection discarding disagreement, not an
error in the pushforward density.

**Result.** Acceptance as a function of `k` (columns are the token dimension; `k = d` cells are the
no-projection control and are excluded from the summary):

| k | d=4 | d=8 | d=16 | d=32 | d=64 | d=128 | mean over `k<d` | CoV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.8885 | 0.8789 | 0.8544 | 0.8808 | 0.8787 | 0.8901 | **0.879** | 1.3% |
| 2 | 0.8096 | 0.7899 | 0.7633 | 0.7586 | 0.7920 | 0.7945 | **0.785** | 2.3% |
| 4 | *0.6425* | 0.6218 | 0.5968 | 0.6134 | 0.6321 | 0.6400 | **0.621** | 2.4% |
| 8 | — | *0.4069* | 0.3635 | 0.4255 | 0.4352 | 0.4199 | **0.411** | 6.8% |
| 16 | — | — | *0.1743* | 0.2072 | 0.2087 | 0.2471 | **0.221** | 8.4% |
| 32 | — | — | — | *0.0736* | 0.0770 | 0.0817 | **0.079** | 2.9% |

**The token dimension almost drops out.** At `k=4`, acceptance is 0.62 whether the token has 4
dimensions or 128 — a 32× range of `d` moving acceptance by 2.4%. This is a much stronger form of
the E1.1/E1.1b claim than either produced. E1.1 showed acceptance falling with `d`; E1.1b showed the
fall survives holding model agreement fixed; E1.2a shows the governing variable is *the dimension of
the space the test is performed in*, and that the token's own dimension matters only insofar as it
is that space. The mechanism is now isolated about as cleanly as this testbed allows.

**What the null control costs us, stated plainly.** The corresponding gains over raw verification
are 1.0× / 1.5× / 3.4× / 8.3× / 35.6× / 180.4× at `k=4` for `d` = 4 → 128. **A random projection with
no structure whatsoever recovers acceptance from 0.0035 to 0.64 at `d=128`.** So "verifying a
low-dimensional statistic restores acceptance" — C2 as previously worded — is now *established*, and
established by the null control, which means it is not evidence for anything HERALD proposes. Any
transform-domain (E1.2b) or learned (E1.3) statistic has to be judged against this curve at matched
`k`, never against raw verification. This is exactly what Fearnhead & Prangle (JRSS-B 2012) predict
for ABC and what Grouped Speculative Decoding (ICCV 2025) proves for the discrete case; we now have
it measured in our own setting.

**Caveats, and the largest one is structural.** *Acceptance here is not free.* Everything in the null
space of `A` is unverified — draft and target may disagree arbitrarily in `d − k` directions and the
token is still accepted. This experiment measures only the acceptance side of that trade and says
nothing about the resulting sample quality, so the numbers above are an upper bound on what any
`k`-dimensional rule can deliver and *not* a speedup. The fidelity side is the whole difficulty, and
it is what makes the choice of statistic (E1.2b, E1.3) matter at all: at matched `k` every rule sits
at roughly the same acceptance, so a structured statistic can only win on *what it discards*. That
reframes the ladder — the interesting axis is fidelity at matched `k`, not acceptance at matched `k`.
Also: spread over draws of `A` grows with `d` (max std 0.007 at `d=4`, 0.049 at `d=128`), so a single
random projection is a noisy baseline at high `d` and five draws is on the low side. Same single
dataset, seed, and capacity gap as E1.1, and these are the 4,000-step checkpoints E1.1b showed to be
slightly past their best validation NLL.

**Changes:**
- `herald/verifiers/projected.py` added: uniform-on-Stiefel `random_projection`, exact pushforward
  `project_mixture` (a projected diagonal Gaussian has a *full* `k × k` covariance), and
  `projected_log_ratio`. Ten tests, anchored on the `k = d` invariance.
- `measure_acceptance` gains a `projection` argument, so E1.1, E1.1b and the whole E1.2/E1.3 ladder
  share one estimator and report all statistics in the space actually being verified.
- **C2 needs restating.** "Verifying a low-dimensional learned semantic statistic restores
  acceptance" is now partly true-but-vacuous: the *low-dimensional* half is established and free,
  and the *learned semantic* half carries the entire claim. C2 should be rewritten in terms of
  fidelity at matched `k`. Flagged here; not rewritten unilaterally.

---

## 2026-09-02 — E1.2-frontier — the acceptance E1.2a called "free" costs ~0.4 nats per discarded dimension. Acceptance is set by `k`; fidelity is set by `d − k`.

**Ran:** `python scripts/run_e12_frontier.py --dims 16 64 --n-projections 3 --eval-batches 20`.
No training, no GPU, 31 s on CPU, on E1.1's checkpoints. Each rule runs a full speculative step:
accept the draft's proposal on the projected statistic, and **on rejection draw from the true
residual `(p − q)₊`** by rejection sampling with `p` as the proposal. Metric is *excess NLL under the
target* — how many nats less likely the emitted tokens are under `p` than genuine `p` samples.
Harness in `herald/eval/fidelity.py`, results in `experiments/E1.2-frontier/`, figure at
`docs/figures/e12_frontier.png`.

**Why the residual sampler matters more than it looks.** With the exact rule and a correct residual,
`q·min(1,p/q) + (1−Ā)·(p−q)₊/Z = p` exactly, so the anchor must read zero. It does: **−0.0031 ±
0.0520** at d=16 and **+0.1251 ± 0.1375** at d=64, with the rejection sampler converging on 100% of
tokens. Had rejection simply resampled from `p` — the one-line version, and what a real
implementation often does — even exact verification would have scored as lossy and every number
below would have been measuring the harness instead of the rule.

**Result.**

| k | d=16 acceptance | d=16 excess NLL | ÷(d−k) | d=64 acceptance | d=64 excess NLL | ÷(d−k) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.8475 | 6.40 | **0.426** | 0.8675 | 27.61 | **0.438** |
| 2 | 0.7419 | 5.35 | **0.382** | 0.7770 | 24.10 | **0.389** |
| 4 | 0.5960 | 3.80 | **0.317** | 0.6216 | 18.77 | **0.313** |
| 8 | 0.3531 | 1.42 | **0.178** | 0.4094 | 11.30 | **0.202** |
| 16 | — | — | — | 0.1965 | 4.45 | 0.093 |
| 32 | — | — | — | 0.0686 | 1.09 | 0.034 |
| exact | 0.1613 | **−0.003** | — | 0.0138 | **+0.125** | — |

**Two variables, cleanly separated.** E1.2a found acceptance is a function of the verified dimension
`k` alone. This finds the fidelity cost is, to a good approximation, **`(d − k) × c(k)`** — the
per-unverified-dimension cost matches across a 4× range of `d` at every shared `k` (0.426 vs 0.438,
0.382 vs 0.389, 0.317 vs 0.313, 0.178 vs 0.202). So:

> **Acceptance is governed by the dimension you verify. Fidelity is governed by the dimension you
> *don't*.**

**This is the correction E1.2a needed, and it changes the outlook.** E1.2a's headline — a random
projection lifts acceptance from 0.0035 to 0.64 at d=128 — read as though low-dimensional
verification were nearly free. It is not: at d=64, `k=4` buys acceptance 0.62 for **18.8 nats per
token**. Worse, the cost grows *linearly in `d`* while the acceptance gain does not, so the naive
"just verify a few dimensions" strategy gets steadily worse exactly where it looks most attractive.
A low-dimensional acceptance rule is not a free lunch at scale; it is a linearly expensive one.

**And it makes C2 falsifiable in a useful way.** Since acceptance at matched `k` is the same for
every rule, a structured or learned statistic can only win by having a **lower `c(k)`** — by
discarding directions that do not matter. That is now a single measurable number, and it is exactly
the claim the JEPA framing makes. E1.2b and E1.3 are scored on `c(k)`, and if their `c(k)` matches a
random projection's, the framing is decoration and the falsifier in the plan has fired.

**Caveats.** Excess NLL is a cross-entropy gap, `CE(p̃,p) − H(p)`, which equals `KL(p̃‖p) + H(p̃) −
H(p)`; it is zero iff `p̃ = p` and it is the right *direction*, but it is not itself a divergence,
and a rule that emitted an over-*concentrated* distribution could in principle score well on it. A
proper `KL(p̃‖p)` estimate is computable from the same quantities and is the obvious follow-up. NLL
in nats/token is not comparable across `d` (different-dimensional densities), which is why the
`÷(d−k)` column and not the raw column is the cross-`d` claim. Only two dimensions and three draws
of `A` per `k`; spread across draws is visible at `k=8`, d=64 (9.3 to 13.7). Same checkpoints,
dataset and capacity gap as E1.1, and the linear-in-`(d−k)` fit rests on four shared `k` values, so
it is a strong regularity rather than an established law.

**Changes:**
- `herald/eval/fidelity.py` added: `residual_sample` (true `(p−q)₊` rejection sampler, skipping
  positions that were accepted) and `measure_fidelity`. Six tests, anchored on losslessness.
- The ladder's scoring axis is now explicit: **`c(k)`, the per-discarded-dimension fidelity cost.**
  E1.2b/E1.3 report it, and neither is worth running on acceptance alone.

---

## Next up

- **E1.2b** The transform-domain statistic, scored on `c(k)` against E1.2-frontier's random-projection
  curve. Literature-cleared 2026-09-02; read GSD and CASCADE in full first, and frame against
  PCG/FreqCa/ABC as prior rather than as insight.
- **`KL(p̃‖p)` estimator** to replace the cross-entropy gap as the fidelity metric. Computable from
  quantities `measure_fidelity` already forms; closes the one real hole in E1.2-frontier.
- **E0.3** The temperature sweep on real models — the capture run. Blocks everything downstream.
  Harness built and tested; blocked only on GPU availability, with all 8 A6000s fully allocated to
  another user (47.7 GB each, ~1.4 GiB free) for this entire session.
- **E0.4** The pixel-domain audit, on E0.3's samples, in the same sitting.
- **E0.2** Reproduce CSpD's 2.33× at its defaults.
- **E0.1b** The FID-50K gate for MAR-B/L and D-JEPA-B, when GPUs are free.

---

## 2026-09-03 — E1.2b — five subspace rules, one frontier. Choosing *which* `k` directions to verify cannot beat choosing *how many*.

**Ran:** `python scripts/run_e12b.py --dims 16 64 --eval-batches 60 --n-projections 3 --threads 8`,
then `python scripts/analyze_e12b.py`. Commit `f723a64`. CPU only, on E1.1's saved checkpoints.

**Setup:** Five choices of the verified subspace at matched `k`, all reduced to orthonormal bases
(only the row space affects the ratio, proved and tested): `random` (the E1.2a null control), `pca`
(top principal directions of the token distribution), `signal_top` (top eigenvectors of `E[JᵀJ]` for
the decoder Jacobian — the directions that move the waveform most), `dct_low` (the same restricted
to the low-frequency DCT band of the decoded patch — the transform-domain statistic, linearized
through the Jacobian so the pushforward stays exactly Gaussian), and `signal_bottom`, an
**anti-control** that verifies the least important directions and must come out worst if the choice
of subspace matters at all.

**Result:** it does not matter. Scored against the random curve interpolated to the *same*
acceptance (never extrapolated):

| | cost range across the sweep | median deviation | max deviation | max as % of range |
|---|---|---|---|---|
| d=16, excess NLL | 5.812 nats | 0.077 | 0.357 | **6.1%** |
| d=64, excess NLL | 26.325 nats | 0.773 | 1.366 | **5.2%** |
| d=16, energy distance | 1.727 | 0.085 | 0.350 | 20.3% |
| d=64, energy distance | 1.396 | 0.051 | 0.282 | 20.2% |

The anti-control ranks **best** at several points (1.26 nats *below* the random curve at d=64, k=8)
and `signal_top` ranks worst (1.37 nats above at k=2). A rule built with full knowledge of the
decoder is indistinguishable from one designed to be as bad as possible, once acceptance is held
fixed. The rules are nonetheless genuinely different: at d=64, k=4 acceptance ranges 0.14 (pca) to
0.79 (anti-control), a 5.6× spread.

**Reading:** there is a conservation identity behind this, and it is why the search was doomed. For
shared-covariance Gaussians, verifying orthonormal `A` separates the pushforwards by `‖Aδ‖` while
what leaks out is governed by `‖A⊥δ‖`, and orthonormality forces `‖Aδ‖² + ‖A⊥δ‖² = ‖δ‖²`. Acceptance
and fidelity are two readings of one conserved budget fixed by the *pair*; a subspace choice only
picks a point on a single curve. This also explains E1.2a: a random `k`-subspace captures `k/d` of
`‖δ‖²` regardless of `d`, so acceptance comes out a function of `k` alone. It refines E1.2a too —
that finding holds only for *random* subspaces, which are exchangeable by rotational symmetry; a
structured subspace can deliberately sit where `p` and `q` agree and buy a much higher rate at fixed
`k`, and gains nothing by it.

**A metric had to be replaced.** The first signal-domain metric was excess squared error of the
decoded waveform against the real window. Under it *every* rule scored **negative** excess
distortion and the anti-control scored best of all (−0.064 vs −0.015 for `signal_top`). The draft is
the smaller model, its samples sit closer to the conditional mean, and squared error against a
single reference is minimised by the conditional mean rather than by a correct sample — so the
metric rewarded variance shrinkage and would have crowned a rule that always emitted the mean.
Replaced with a signal-domain **energy distance** `2E‖X−Y‖ − E‖X−X'‖ − E‖Y−Y'‖`, which is zero
exactly when the decoded distributions agree and is *increased* by mode collapse. Under it the
ordering is the one the construction demands: exact +0.060±0.250, `signal_top` +0.069, random
+0.337, anti-control **+1.313**. Both metrics are now reported on every rule, and the exact anchor
holds on both.

**Changes:** kills the E1.2b/E1.3 program as originally framed — there is no `c(k)` to beat, because
`c(k)` is the same for every rule. C2 is restated a third time, now as a conservation result rather
than a search for a better statistic, and it is *stronger* for it: an impossibility claim with a
proof for the tractable case and an adversarial control in the measurement. Reroutes the positive
claim to E1.4: the only lever left is the divergence itself.

---

## 2026-09-03 — E1.4 — distillation moves the frontier that projection can only slide along

**Ran:** `python scripts/run_e14.py --dims 16 64 --stages 3 --steps-per-stage 600 --eval-batches 40
--n-projections 2 --threads 8`. Commit `f723a64`. CPU only.

**Setup:** the constructive corollary of E1.2b. If the frontier is fixed by `‖δ‖/σ`, a property of
the *pair*, then only changing the pair can move it. The draft is distilled against the frozen
target with a **mode-covering** objective (`KL(p‖q)`, estimated by drawing from the target and
scoring those draws under the draft). Direction chosen deliberately: reverse KL would let the draft
collapse onto one mode and score well while being a poor proposal, which is exactly the failure
speculative sampling punishes.

**Result:**

| d | acceptance before | after 1,800 steps | gain | fidelity cost | projection cost for the same rate |
|---|---|---|---|---|---|
| 16 | 0.1744 | 0.1979 | 1.13× | 0 (exact, lossless) | 0.179 nats/token |
| 64 | 0.0182 | 0.0245 | 1.34× | 0 (exact, lossless) | 0.061 nats/token |

Excess NLL stays at zero within error at every stage, as it must: the rule remains exact, and
losslessness is guaranteed analytically, not measured hopefully.

**Reading:** modest in absolute terms and not presented as a competitive speedup — the point is the
*direction*. Distillation is the one intervention tested that moves the curve rather than the
position on it, which is what the conservation law predicts and what all five subspace rules could
not do. The gain being small at d=64 is consistent with the same √d mechanism: moving `M = δ√d`
requires shrinking `δ` everywhere at once, and that gets harder as `d` grows. That is a further
statement of the difficulty, not a way around it.

**Changes:** supplies C3's positive half. The correct target for future work is the *pair*, not the
acceptance rule — which is exactly the HERALD thesis, since a D-JEPA-style predictor and EMA target
encoder are a coupled pair produced as a by-product of training. Two bugs fixed that only appear off
CPU: `distillation_loss` built its masking generator on CPU (crashes on CUDA batches), and
`dct_matrix` built on CPU while the Jacobian was on device. Both now follow the batch/tensor device,
with a regression test for the first.

---

## 2026-09-03 — submission — ICLR 2027 paper written against the corrected claim set

**Ran:** `submission/` built with `tectonic -X compile main.tex` (no TeX on the experiment host; a
static tectonic binary and a self-contained `iclr2027_conference.sty` reimplementation were used).

**Result:** main text exactly 9 pages, references pp. 9–10, appendix 10 pages, 20 total. Compiles
with no errors and no undefined references or citations.

**Reading:** the paper leads with the conservation result rather than with a method. Structure: the
measured collapse (E1.1 + convergence controls), the conservation law with proof and the five-rule
test including the anti-control, the metric that had to be replaced, distillation as the lever that
works, and the corrected CSpD criterion. The negative result is the contribution; the falsifiers are
stated explicitly in an appendix, including the one that *fired* during the work (squared error was
gameable, and the anti-control exposed it).

---

## 2026-09-03 — E1.2b-gpu / E1.4-gpu — scaled sweeps on ai-lab2, and the trend that reversed a caveat

**Ran:** both experiments re-run on an RTX 4090 (`adelechinda@ai-lab2`) at roughly 5× the CPU budget:
E1.2b over `d ∈ {4,8,16,32,64,128}` × 5 rules (1,346 s), E1.4 over the same six dimensions at 10,000
distillation steps each, five measurement stages (5,377 s). Both wrote `results.json` and then died
in their plotting call — the host's conda env is Python 3.9 and `zip(strict=)` needs 3.10+. No data
was lost; figures were regenerated locally with `--plot-only`.

**Result — conservation (E1.2b):** the collapse holds across a 32× range of `d`, and *tightens*:

| d | cost range (nats) | median dev | max dev | max as % of range |
|---|---|---|---|---|
| 4 | 0.240 | 0.027 | 0.038 | 15.9% |
| 8 | 1.560 | 0.017 | 0.112 | 7.2% |
| 16 | 5.768 | 0.050 | 0.356 | 6.2% |
| 32 | 11.341 | 0.331 | 0.950 | 8.4% |
| 64 | 26.305 | 0.616 | 1.476 | 5.6% |
| 128 | 48.016 | 0.777 | 1.933 | **4.0%** |

The d=16 and d=64 rows replicate the CPU run to within 0.06 nats and 0.6 percentage points, which is
an independent confirmation on different hardware and a different RNG stream. The anti-control now
ranks *best* at the three largest dimensions, and at d=128, k=16 the rules span a 30× acceptance
range (0.018 pca to 0.550 anti-control) while still tracing one curve.

**Result — distillation (E1.4):** the gain grows monotonically with dimension.

| d | before | after 10,000 steps | gain | projection cost for the same rate |
|---|---|---|---|---|
| 4 | 0.6453 | 0.7198 | 1.12× | 0.188 nats/token |
| 8 | 0.4018 | 0.4728 | 1.18× | 0.321 |
| 16 | 0.1779 | 0.2507 | 1.41× | 0.613 |
| 32 | 0.0722 | 0.1084 | 1.50× | 0.390 |
| 64 | 0.0176 | 0.0459 | 2.61× | 0.366 |
| 128 | 0.0036 | 0.0208 | **5.72×** | 0.739 |

Exact anchors are within 1.2 SEM of zero at every `d`, residual convergence 100% throughout.

**Reading:** this reverses the caveat written after the CPU run. That run saw a smaller effect at
d=16 than at d=64 and read it as "moving `M = δ√d` gets harder as `d` grows". With six dimensions and
a 5× budget the trend is unambiguous and points the other way: acceptance is `2 − 2Φ(M/2)`, a
Gaussian tail that is exponentially sensitive to `M` once `M` is large, so a fixed proportional
shrink of `δ` buys a few percent at d=4 and a multiple at d=128. The collapse that defeats
speculative sampling is the same thing that makes closing the pair pay. Projection cannot follow,
because Corollary 1 pins it to a curve that `δ` fixes. The earlier reading was under-powered, not
wrong about the mechanism.

**Changes:** paper updated throughout to the six-dimension sweep — `tab:collapse` restructured to one
row per dimension, `tab:e14` to six rows, a new right-hand panel showing gain against `d`, and the
`√d`-difficulty caveat in §5 replaced by the trend. Appendix tables are now generated from
`results.json` rather than transcribed. Main text stays at 9 pages; appendix 11.
---

## 2026-09-09 — E0.5b — The KL16 decoder's sensitivity spectrum decays 20× more slowly than the DCT spectrum of the images it produces, which is the image-domain statement of the conservation result

**Ran:** `python scripts/plot_e05_spectrum.py --results experiments/E0.5-kl16-jacobian/results.json
--samples experiments/E0.1-mar-base/samples.png --out submission/figures/e05_kl16_spectrum.png`.
No new GPU work: this re-reads E0.5's stored Jacobian spectra and adds one derived quantity. All
eight A6000s were held at 100% by another user's job throughout, so nothing new could have been run
anyway.

**Setup:** the four E0.5 latents (top 64 singular values of the KL16 decoder's local Jacobian, of
4096 latent directions) against the 2D DCT of the luminance of the same four MAR-B samples. Both
spectra are sorted descending, truncated to 64 components, and normalised by their leading
component, which is the only way the two are comparable — they are spectra of different objects and
only the *shape of the decay* is being compared.

**Result (derived, not newly measured).** At rank 64, the median normalised magnitude is **0.0146**
for the image DCT coefficients and **0.3038** for the decoder's singular values: the sensitivity
spectrum is roughly **20× shallower**. Recomputed from E0.5's stored values, `σ_max` across the four
latents spans 15.99 to 47.38.

**Reading.** This is the empirical half of the argument the paper makes analytically. Transform
coding is profitable because a natural image's energy really is concentrated in a few coefficients,
and the DCT curve shows that concentration on these very images. Projected verification would need
the *decoder* to concentrate its perceptual response in the same way, and it does not: at the rank
where the image has given up 98.5% of its leading magnitude, the decoder has given up 70%. So the
transform-coding intuition fails twice over — Proposition 2 says a good basis could not help even if
one existed, and this says the structure it would exploit is not there to begin with. The two are
complementary, and it is worth being explicit that this measures decoder geometry, **not** where the
draft and target disagree, which remains unmeasured on real image models.

**Caveats.** Inherits all of E0.5's: four latents, top 64 of 4096 so the anisotropy is a lower
bound, and latents re-encoded from MAR's own samples rather than photographs. The DCT side adds one
more — the comparison is between a full-resolution pixel-domain transform and a latent-space
Jacobian, so it is a statement about decay shape and nothing finer.

**Changes:** paper reframed around the image-processing story for a TMLR submission (see below).
`§5 The Real Image Tokenizer Has No Band to Keep` is new and carries this figure plus E0.5's table;
the conservation section gained a `Why transform coding does not transfer` passage naming
Eq. 4 as Parseval's theorem applied to the draft–target gap rather than to the signal; `dct-low` is
promoted from one rule among five to the transform-coding rule under test. Venue changed from ICLR
2027 to TMLR: `iclr2027_conference.sty` removed, official `tmlr.sty`/`tmlr.bst` vendored, main
content held to exactly 12 pages so the two-week review deadline still applies. Appendix 12 pages.

---

## 2026-09-09 — E0.6 / E0.7 / E0.8 — Reviewer overhaul: the theory generalises without Gaussians, the random-projection derivation was invalid, and CSpD's test fires on a near-degenerate distribution

**Context:** a detailed external review judged the paper "borderline / weak reject" for a top venue,
with the core objection that several claims were broader than the theorem and experiments support.
This entry records the three measurements made in response and the claim changes they forced.

### Verified the review's arithmetic before acting on it

Three of its criticisms are correct and are now fixed in the paper.

- **`M ≈ 1.35`, not `1.7`.** `A = 2 − 2Φ(M/2) = 0.5` needs `Φ(M/2) = 0.75`, so `M = 1.349`.
  `M = 1.7` corresponds to `A = 0.395`. The appendix design rule was wrong by that margin.
- **The `0.519` exponent is not distinguishable from `0.5`.** Log-log fit over the six measured `M`
  gives slope `0.5194`, se `0.0525`, 95% CI `[0.374, 0.665]`, `p = 0.73` against `0.5`. The paper no
  longer reports it as a departure.
- **`181×` was right, its presentation was not.** Unrounded, `0.642470 / 0.003548 = 181.06`; the
  four-decimal table rounds to `183.6`. The caption now says so.

### E0.8 — the analytic reference, and the random-projection law done properly (no GPU)

`python scripts/run_e08.py`

The old derivation substituted `E||Aδ||² = (k/d)||δ||²` into the acceptance formula. That is invalid
because `f(z) = 2 − 2Φ(√z/2σ)` is nonlinear; measured over the sweep the substitution errs by
**2.8–8.8% (median 5.4%)**. Doing it correctly *strengthens* the result rather than repairing it.
With `Z = ||Aδ||²/||δ||² ~ Beta(k/2, (d−k)/2)` and the per-coordinate gap held fixed so `M = c√d`,
the verified separation `M²Z = c²·dZ ⇒ c²·χ²_k`, **a limit containing no `d` at all**. So
"acceptance is a function of `k` alone" is a theorem with an `O(1/d)` correction, not a coincidence.
Evaluating the exact Beta expression over `d ∈ {16,…,4096}` gives a CV across dimensions of
**0.15% at k=1, 0.47% at k=4, 1.7% at k=32**, against the `1.3–12.3%` measured on trained models —
the excess being what heteroscedastic mixtures and finite samples should add.

Also: a training-free dimension sweep to `d = 4096` at the measured `δ/σ = 0.578`, with a direct
Monte Carlo of the accept/reject rule confirming the closed form to **6.5e-4**. This is the clean
reference the trained sweep is a noisy sample of, and it answers the "dimension is not the only
variable" objection without needing the tokenizer to be held fixed.

New figure `e08_feasibility.png`: acceptance contours over `(d, δ/σ)`. Holding `A = 0.5` needs
per-coordinate agreement within **0.34σ at d=16** but **0.021σ at d=4096**.

### E0.6 — CSpD's acceptance test on a real MAR-B/MAR-H pair (GPU 7, ~28 s)

`python scripts/run_e03.py capture --draft base --target huge --num-images 4 --num-iter 64`
then `python scripts/run_e06.py`

1008 real token decisions at `d = 16`. **The distributions the shipped test compares are nearly
degenerate.** Final-step posterior `σ = 0.0064` for both models, agreeing to a median ratio of
`1.000`; the proposed token equals the draft's posterior mean to numerical precision, because the
last denoising step adds no noise. Median gap `0.107`, i.e. **16.5σ per coordinate**, giving
`M ≈ 138` and exact acceptance that rounds to **0.0000**.

**Reading — and what this is not.** This is a fact about the final denoising transition, not about
the two models. It does *not* measure the draft/target separation the paper's theory concerns, which
is the trajectory marginal. What it does explain is the sampling temperature: at the shipped value of
42 the effective separation falls to `M ≈ 3.3`, which is the only regime in which that test
discriminates at all. It went into the CSpD appendix, not the main text.

### E0.7 — the measurement the review actually asked for: written, not yet run

`scripts/run_e07.py` samples both heads `n_draws` times per position at *shared* contexts, which is
the only honest route to the trajectory marginal since a diffusion head has no closed-form `(μ, σ)`.
Lint-clean and ready. **Blocked on GPU:** all eight A6000s returned to 100%/48 GB from another user's
job minutes after the E0.6 capture finished. This remains the single highest-value open experiment.

### Claim changes in the paper

- **Title** → *The Dimensionality Barrier in Continuous Speculative Sampling*. "Acceptance is
  conserved" was indefensible: projected acceptance demonstrably moves from 0.0035 to 0.64.
- **New Proposition 2 (no assumptions):** `KL(p‖q) = KL(p_A‖q_A) + E_{p_A}[KL(p(u|s)‖q(u|s))]` by the
  KL chain rule under a bijection. Gaussian conservation is now a corollary of it. This answers the
  "why should a Gaussian identity tell me anything general" objection: projection *never* destroys
  divergence, it converts marginal into conditional, for arbitrary densities.
- **"No subspace can beat any other" → "a rule selects a point on the frontier, it cannot move the
  frontier."** The old phrasing was contradicted by our own data (0.018 vs 0.550 at `d=128, k=16`).
- **"the only lever is the divergence" → "within the class of orthonormal linear rules studied
  here."** Nonlinear/adaptive/learned statistics are explicitly left open.
- **"dimension is the only variable" → "a dimension-indexed family of matched systems,"** with the
  tokenizer retraining stated outright and the analytic reference carrying the isolating argument.
- **KL16: "has no band to keep" → "no evidence of a compact band among the top 64 of 4096."**
- **Excess NLL is labelled a zero-anchored cross-entropy discrepancy, not a divergence.**
- **Contributions 6 → 3**; CSpD audit demoted to an appendix and reframed as an audit of one factor
  rather than a verdict, explicitly noting we did not reproduce its speedups.
- **Related work** gained FAR (arXiv 2504.18391, verified) and De-MAR (ICCV 2025 pp. 18024–18033,
  verified), and the motivation no longer implies continuous AR is slow for want of a remedy.

**Page budget:** main body is 12 pages plus ~60% of page 13, over the 12-page bar for TMLR's
two-week review window. Held there deliberately: the overflow is the new proposition, the
feasibility figure and the scoping the review demanded, and figure rescaling was absorbed by float
reflow without recovering space. 113 tests pass; build is clean with no unresolved references.

**Still open (review P1):** seeds, a second dataset, the oracle discrepancy-aligned subspace,
bootstrap CIs on frontier deviations, expanded KL16 sampling, and distillation continued to a useful
acceptance regime.

## 2026-09-10 — E0.7 / E1.2c / E1.5 — the review's remaining empirics, and a discrepancy between the theory's emitter and ours

Second pass over the same reviewer's list, this time the items that needed either a proof or a GPU.
Four experiments, two of which changed what the paper claims.

### The emitter the theory describes is not the one we measure (no GPU, `scripts/check_projected_emitter.py`)

The review asked us to define the projected output distribution explicitly as
`p̃_A(s,u) = p_A(s) q(u|s)` and derive its leakage. Doing that surfaced a problem worth more than the
edit itself: **`herald.eval.fidelity._emit` does not implement that sampler.** It applies the
projected acceptance test but repairs rejections from the *full-space* residual `(p−q)₊`, not the
projected one, so its emitted law is not `p̃_A`. Writing the reviewer's proposition without checking
would have attributed to our experiment a decomposition belonging to a different algorithm.

Measured both on a shared-covariance Gaussian pair (σ=1, per-coordinate gap 0.578, 4e5 samples/cell):

| d | k | acc measured | acc predicted | xNLL of `p̃_A` | ‖A⊥δ‖²/2σ² | xNLL of ours |
|---|---|---|---|---|---|---|
| 16 | 4 | 0.564 | 0.563 | 2.012 | 2.005 | 1.068 |
| 64 | 16 | 0.248 | 0.248 | 8.007 | 8.018 | 2.041 |
| 128 | 16 | 0.247 | 0.248 | 18.694 | 18.709 | **4.701** |

Two findings. The idealised emitter's excess NLL reproduces the conservation identity's leakage term
to three decimals, which is a direct numerical confirmation that the conditional term of the chain
rule is the divergence of a *real sampler* and not a bookkeeping remainder. And our repair leaks
**strictly less** everywhere, by 4× at `d=128, k=16`, because it corrects a rejected position in
every coordinate rather than only the verified ones. The direction is favourable: every frontier we
report is traced under the *more* forgiving scheme, so the obstruction is understated, not inflated.
Documented as its own appendix section rather than buried.

### New Proposition 2: acceptance is one minus a total variation (no GPU)

The review's other must-fix was a general bridge from the KL decomposition to acceptance. It turns
out to be exact rather than an inequality. Under proposals from `q`, the projected test accepts with
probability `∫min(p_T, q_T) = 1 − TV(p_T, q_T)`, and the data-processing inequality gives
`TV(p_T,q_T) ≤ TV(p,q)`, hence `Ā_T ≥ Ā_exact` for **any** measurable statistic. This makes the null
control a theorem: every statistic weakly raises acceptance however badly chosen, so the random
projection's 180× gain was guaranteed before we ran it. Equality holds iff `{p>q}` is (up to a
`p=q` set) an event of `σ(T)`, which is *weaker* than `p/q` being `σ(T)`-measurable — the first draft
of the statement had the stronger, wrong condition.

### E1.2c — the oracle-disagreement subspace does not escape the frontier (GPU 6, ~9 min)

`python scripts/run_e12c_oracle.py --out experiments/E1.2c-oracle`

The obvious objection to E1.2b is that all five rules are heuristics. This one is not: rank
directions by the measured `E[δδᵀ]` and verify the top `k`. By construction it is the best rank-`k`
subspace for capturing the disagreement, so it upper-bounds any linear rule of that rank.

**The disagreement really is concentrated** — one direction carries 28–40% of `E‖δ‖²`, four carry
78–91% — so the rule has genuine structure to exploit. It still buys nothing. Largest departure from
the random curve at matched acceptance, as a fraction of the cost range: 0.9% (d=4), 5.2% (8), 3.1%
(16), 4.7% (32), 5.3% (64), **2.8% (128)** — all inside the 4.0–15.9% envelope the five heuristics
occupy, and every one on the *wrong* side (the oracle is if anything slightly worse than a random
subspace). Perfect knowledge of where two models disagree does not locate an escape.

### E1.5 — distillation shrinks the divergence it is supposed to shrink (GPU 5, ~50 min)

`python scripts/measure_distillation_divergence.py --out experiments/E1.5-divergence`

E1.4 measured the consequence (acceptance up) and asserted the mechanism (divergence down). A draft
that merely inflated its variance would have produced the same table. Re-ran the identical
trajectory — same checkpoints, schedule, seeds — recording `KL(p‖q)` by Monte Carlo and `M`:

| d | KL before | KL after | drop | M before | M after | acc before | acc after |
|---|---|---|---|---|---|---|---|
| 4 | 0.514 | 0.322 | 37.3% | 0.320 | 0.212 | 0.6497 | 0.7170 |
| 16 | 4.731 | 3.740 | 20.9% | 1.075 | 0.830 | 0.1764 | 0.2479 |
| 64 | 19.281 | 14.490 | 24.8% | 1.922 | 1.551 | 0.0180 | 0.0434 |
| 128 | 37.860 | 28.710 | 24.2% | 2.750 | 2.144 | 0.0033 | 0.0189 |

The chain is now observed link by link. Acceptance rises monotonically across all five stages at
every `d`; KL and M fall monotonically apart from sub-1% final-stage wiggles at `d ∈ {32,64,128}`.
Re-measured acceptance agrees with E1.4's independent run (0.0033→0.0189 vs 0.0036→0.0208).

**The interesting number is the flatness.** The divergence reduction is ~20–25% at *every* width
while the acceptance it buys grows 1.12× → 5.72×. That is the Gaussian tail, and it is the sharpest
evidence for §6's argument that we have.

**Caveat recorded, not hidden:** computing `M` from the mixture's *total* variance folds
between-component spread into the scale, making `M` smaller than the value governing acceptance, so
`2−2Φ(M/2)` overpredicts badly (0.372 vs a measured 0.0189 at `d=128`). `M` is therefore read only as
a before/after indicator here; `KL(p‖q)` carries the argument because it assumes nothing.

### E0.7 — MAR-B vs MAR-H, trajectory-marginal, finally run (GPUs 4 and 7, ~2 and ~6 min)

`CUDA_VISIBLE_DEVICES=7 python scripts/run_e07.py --cfg 3.0 --num-images 4 --n-capture 12 --max-positions 24`

The single highest-value open experiment from the last entry; the GPUs freed up. This is the
measurement E0.6 could not make.

| | guidance 3.0 (188 positions) | guidance 1.0 (64 positions) |
|---|---|---|
| σ target / draft | 0.4007 / 0.3984 | 0.4823 / 0.4841 |
| per-coordinate \|δ\|/σ, median | 0.280 | 0.221 |
| separation `M`, mean / median | 1.850 / 1.683 | 1.400 / 1.256 |
| predicted acceptance (95% CI) | **0.397** [0.369, 0.425] | 0.513 [0.473, 0.550] |

**`M ≈ 1.9`, not 138.** The final-step figure was an artefact of a degenerate transition, exactly as
E0.6 argued but could not demonstrate. Three consequences. Shared covariance — the assumption
Proposition 4 needs — holds to within 0.6% for this pair, so the identity is near-exact here rather
than approximate. Exact speculative verification of *one* KL16 token is viable, at ~0.4 acceptance.
And the pair lands where the feasibility map put it: the map calls for 0.34σ at acceptance 0.5, the
measurement gives 0.28–0.38σ. The map is now calibrated against a published pair rather than only
against our testbed. Guidance costs acceptance (1.0 → 3.0 moves `M` from 1.40 to 1.85), because it is
applied to both models but sharpens the larger one more.

This is a prediction from a measured `M` under a Gaussian model, **not** an end-to-end speculative
run; the limitations section now says so explicitly, and the old "remains unmeasured" sentence is
gone.

### Paper changes

- **New Proposition 2** (TV/data-processing) and the explicit emitter `p̃_A` in Eq. 3, with
  Proposition 3's remainder rewritten as `KL(p‖p̃_A)` — the review's two theory must-fixes.
- **Corollary 1 scoped** to shared isotropic covariance in its statement, with a paragraph saying
  what survives without it (Props 2–3) and what does not (the single curve).
- **Bootstrap CIs** on the collapse statistic, in Table 2 and a new appendix section. At `d=128`,
  4.0% [3.3, 5.6]; the intervals at `d=4` and `d=128` are disjoint, so the tightening with dimension
  is resolved rather than noise. Bootstrapping a maximum is upward-biased and the appendix says so,
  which is why the energy-distance point estimate at `d=128` falls below its own interval.
- **Wording**, per the review: "moves the frontier" → "shifts the attainable acceptance at zero
  fidelity cost"; the related-work exactness claim narrowed to "the family of exact speculative
  accelerations that preserve the target distribution by construction"; the conclusion's absolute
  phrase replaced with the scoped one and the nonlinear-statistic question raised as the open problem.
- **CSpD appendix renamed** to "Audit of a Prior Continuous Speculative-Decoding Criterion".

**Page budget — decided: stay at 13 pages.** The additions cost about a page. Trimming duplicated
null-control prose in §4.1 (now a theorem in §4.2) and compressing the new paragraphs recovered most
of it, leaving the main body at 13 pages rather than 12, overflowing by roughly 26 lines. TMLR has no
hard limit: 12 pages of main content earns reviewers a 2-week deadline and anything longer a 4-week
one, with no effect on the acceptance criteria. Since the extra page is precisely the content the
review demanded, the slower window is the better trade and no further cutting is planned.

**Still open (review P1):** seeds and a second dataset.

---

## E0.5b — KL16 decoder Jacobian at N=50 (2026-09-10)

**Question.** The review called the four-latent spectrum a pilot, and rightly: the between-image
spread is exactly the quantity four samples estimate worst. Does the "no compact perceptual band"
reading survive an order of magnitude more latents?

**Setup.** `scripts/generate_mar_samples.py` draws 50 MAR-B samples with E0.1's settings
(cfg 2.9, 256 iters, seed 0) and stores them as float32 rather than as an 8-bit montage, so the
sample count is no longer capped by the grid and the images are not quantised on the way in.
`scripts/run_e05.py` gained `--images` to consume that tensor, and now reports IQRs rather than
min/max alone. On GPU the Jacobian pass is 14 s per latent against ~200 s on CPU, so all 50 run in
about 13 minutes. Sanity check: re-running latent 0 through the montage path reproduces the original
anisotropy to 4.05 vs 4.08.

**Result — the headline holds, and tightens.**

| quantity | N=4 | N=50 (median, IQR) |
| --- | --- | --- |
| anisotropy over top-64 | 3.31 | **3.16** (2.70–3.51) |
| effective rank of 64 | 61.06 | **61.55** (61.11–62.19) |
| DCT magnitude at rank 64 | 1.5% | **1.7%** |
| Jacobian magnitude at rank 64 | 30% | **32%** |

88% of latents have anisotropy below 4 and 92% have effective rank at least 60, so this is a property
of the decoder rather than of four lucky images.

**Two numbers moved, and both are now stated honestly in §5.**

- *Between-image scale.* `σ_max` runs 17.7–136.5 across 50 images, a 7.7× spread, against the 3×
  the four-image pilot showed. The paper said "threefold different pixel error"; it now says nearly
  eightfold. This strengthens the incidental finding — a latent-space tolerance is marginal over
  images, not conditional on one — but it is a case of the pilot having understated a problem.
- *Linearisation check.* The old caption claimed median relative error 0.054, max 0.076. Pooled over
  50 latents it is median 0.093, max 0.431, which looks worse until it is split by direction: the
  leading direction is median 0.018 (the 0.431 is a single outlier), while the 64th direction sits a
  consistent +12% — the finite-difference gain *exceeds* the Jacobian's prediction in all 50 latents.
  So the true `σ_64` is slightly larger than measured, the reported anisotropy is an overestimate of
  how uneven the decoder is, and the isotropy conclusion is on the safe side. §5 now says this rather
  than quoting a pooled number that hides the split.

**Paper changes.** Table 4 is now a summary over 50 latents (median / IQR / range) instead of four
per-latent rows, which is both more informative and one line shorter; §5's prose, the DCT contrast,
and the "what it does not say" caveats are updated, and the "four latents is a pilot" hedge is gone.

---

## E1.6 — Multi-seed (3 seeds) and 3-fold expanding chronological CV sweep (2026-11-23)

**Ran:** `scripts/run_e16_cv_seeds.py` across 8× RTX A6000 GPUs (~10 min wall-clock for 30 tasks):
- **Multi-seed suite:** `seeds ∈ {0, 1, 2}` on Fold 3 (standard 70/10/20 ETTh1 split: 12,099 train / 1,647 val / 3,389 test windows) across `d ∈ {4, 8, 16, 32, 64, 128}`.
- **Chronological CV suite:** 3-fold expanding-origin CV via `herald.data.expanding_cv_bundles` (`seed = 0`, all 6 dims), fitting each fold's standardizer strictly on its training prefix before windowing so test blocks (3,389 windows each) are future and mutually disjoint:
  - Fold 1: 5,131 train / 1,647 val / 3,389 test (months 9–12)
  - Fold 2: 8,615 train / 1,647 val / 3,389 test (months 13–16)
  - Fold 3: 12,099 train / 1,647 val / 3,389 test (months 17–20)

Each run trained a fresh tokenizer (2k steps), target/draft MAR (4k steps), measured exact collapse, then ran 10k steps of mode-covering KL distillation with exact `KL(p||q)`, `M`, and `Excess NLL` audits. Configs and per-task JSONs in `experiments/E1.6-cv-seeds/results.json`.

**Result — multi-seed (Fold 3, mean ± std over 3 seeds):**

| d | collapse acc. | before → after (distill) | gain | KL drop | Excess NLL after |
| --- | --- | --- | --- | --- | --- |
| 4 | 0.6271 ± 0.0075 | 0.6275 → 0.6908 | 1.10 ± 0.01× | 32.4 ± 3.2% | +0.004 ± 0.021 |
| 8 | 0.3874 ± 0.0169 | 0.3880 → 0.4637 | 1.20 ± 0.03× | 27.0 ± 0.7% | -0.005 ± 0.032 |
| 16 | 0.1881 ± 0.0131 | 0.1898 → 0.2534 | 1.33 ± 0.01× | 23.6 ± 2.3% | -0.008 ± 0.023 |
| 32 | 0.0702 ± 0.0082 | 0.0727 → 0.1228 | 1.69 ± 0.13× | 24.1 ± 4.8% | +0.046 ± 0.073 |
| 64 | 0.0195 ± 0.0024 | 0.0188 → 0.0511 | 2.74 ± 0.35× | 24.5 ± 0.8% | +0.056 ± 0.055 |
| 128 | 0.0036 ± 0.0017 | 0.0033 → 0.0194 | **6.23 ± 1.70×** | 26.1 ± 2.5% | -0.065 ± 0.419 |

**Result — 3-fold expanding chronological CV (`seed = 0`, mean ± std over 3 folds):**

| d | collapse acc. | before → after (distill) | gain | KL drop |
| --- | --- | --- | --- | --- |
| 4 | 0.5963 ± 0.0496 | 0.5949 → 0.6565 | 1.10 ± 0.01× | 28.2 ± 6.6% |
| 8 | 0.3493 ± 0.0717 | 0.3463 → 0.4156 | 1.20 ± 0.03× | 25.4 ± 3.4% |
| 16 | 0.1648 ± 0.0684 | 0.1644 → 0.2162 | 1.33 ± 0.07× | 21.5 ± 0.9% |
| 32 | 0.0615 ± 0.0359 | 0.0592 → 0.1038 | 1.95 ± 0.51× | 25.1 ± 2.4% |
| 64 | 0.0179 ± 0.0099 | 0.0164 → 0.0459 | 2.84 ± 0.33× | 26.1 ± 4.7% |
| 128 | 0.0031 ± 0.0023 | 0.0027 → 0.0123 | **4.76 ± 0.64×** | 24.7 ± 2.9% |

**Caveats:** Fold 1 has only 5,131 training windows and exhibits higher baseline gap (`δ = 0.313` at `d=16`, `0.411` at `d=128`), lowering baseline acceptance across folds slightly while preserving both the dimensional collapse (~192× across folds) and the monotonic dimension-growing distillation payoff (`1.10×` → `4.76×`).

