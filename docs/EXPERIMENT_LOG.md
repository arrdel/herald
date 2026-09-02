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

## Next up

- **E0.3** The capture run, blocked only on GPU availability — all 8 A6000s have been fully
  allocated to another user (47.7 GB each, ~985 MiB free) for the whole session. Harness first.
- **E0.1b** The FID-50K gate for MAR-B/L and D-JEPA-B, when GPUs are free.
- **E0.2** Reproduce CSpD's 2.33× at its defaults.
- **E0.3** The temperature sweep on real models. Blocks everything downstream.
- **E0.4** The pixel-domain audit, on E0.3's samples, in the same sitting.
