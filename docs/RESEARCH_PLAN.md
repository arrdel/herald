# HERALD — Research Plan

Living document. Amend it when evidence arrives; do not quietly drift from it.

---

## 0. Thesis

Continuous-token generative models are slow because each token needs a full denoising
trajectory. Speculative sampling should fix this, but the only existing continuous method
(CSpD) buys its acceptance rate with an `inner_temperature=42.` relaxation that voids its
losslessness claim. We argue the underlying obstruction is dimensional, not incidental, and that
the right move is to change the *space* in which verification happens: from the *d*-dimensional
VAE token to a low-dimensional JEPA semantic embedding, with a distribution-free finite-sample
guarantee replacing the unattainable exactness claim.

The components needed are already trained and already discarded inside every D-JEPA checkpoint.

---

## 1. Where the compute actually goes

From `baselines/djepa-imagenet/models/djepa.py::sample_tokens`, per generation step:

1. `forward_encoder` + `forward_decoder` → `z` for **all** masked positions. One transformer
   pass, batched over the sequence. Cheap per token.
2. `diffloss.sample(z, ...)` → runs the denoising MLP for ~100 steps, **per token**.

**Measured in E0.1** (MAR-B, `num_iter=256`, `cfg=2.9`, one A6000): the head is **98.3% of wall
time**, the encoder and decoder together 1.7%. Step 1 already produces a semantic prediction of
every token that step 2 then spends nearly all of the run computing.

**But not for the reason originally written here.** The first version of this section argued from
volume — the head "runs `64 × ~100 × (tokens per step)` times". By volume the head is the *smaller*
consumer: 204,800 token evaluations against the transformer's 916,480, a ratio of 0.22. The head
dominates because its ~100 denoising steps run **sequentially at a batch of one or two tokens**,
while the transformer covers all 320 positions in a single batched pass. Per token evaluation the
head costs about 260× more wall time. This is a serialization and utilization effect.

**Implication, unchanged.** The drafter is free: reading `z` costs 1.7% of the run, and we are not
adding a draft model but reading a tensor that is currently thrown away. This is the structural
difference from CSpD, which trains and loads a second MAR.

**Implication, new and load-bearing.** Because the cost is serialization rather than arithmetic, an
NFE reduction does not convert into a proportional wall-time win, and raw NFE is the wrong headline
number: counted in token evaluations the transformer is the larger consumer, and a reviewer who
checks will find that. Report speedups in wall-time and in *sequential head invocations*, with NFE
alongside. Any acceptance rule that reduces head invocations without reducing their serialization
will underdeliver against its NFE figure.

---

## 1b. How hard is exact continuous acceptance, quantitatively — and a correction to C1

Added 2026-09-02. For two Gaussians sharing a covariance `σ²I`, expected acceptance has a closed
form: `E_q[min(1, p/q)] = 1 − TV = 2 − 2Φ(M/2)` with `M = ‖m_p − m_q‖/σ`. This is Eq. 45 of
*Accelerating Speculative Diffusions via Block Verification* (arXiv 2606.13426), cross-validated
against our own implementation by Monte Carlo in `tests/test_cspd_exact.py`. Writing the
per-coordinate gap as `δ = ‖m_p − m_q‖/(σ√d)` gives `M = δ√d`, so at fixed per-coordinate agreement
acceptance decays in `√d`:

| per-coordinate gap δ | acceptance at d=1 | d=16 | d=64 | d=256 |
| --- | --- | --- | --- | --- |
| 0.25 σ | 0.90 | **0.62** | 0.32 | 0.05 |

and at MAR's `d=16`: δ=0.05σ → 0.92, δ=0.1σ → 0.84, δ=0.25σ → 0.62, δ=0.5σ → 0.32, δ=1.0σ → 0.05.

**This is a correction, and it cuts against us.** C1 has been stated as "exact continuous
speculative decoding is defeated by dimensionality." At the dimension CSpD actually verifies in —
`d=16`, one KL16 token at a time — that is too strong. Sixteen dimensions is not a curse: if
MAR-B and MAR-H agree to within 0.1σ per coordinate, exact acceptance is 84% and CSpD would have
had no reason to relax anything. The `√d` penalty only bites at the block or image level.

So C1's fate rests entirely on an empirical quantity nobody has measured: **δ, the per-coordinate
disagreement between draft and target on real MAR pairs.** That is now the single most important
number in the project, and E0.3 measures it directly. Stating C1 as a dimensionality argument
without δ would be arguing from a mechanism whose magnitude we have not checked.

Two things do survive independently of δ. First, δ is not constant across the generation: both the
per-token variance and the model disagreement change as the sequence fills in, so acceptance is a
*schedule*, not a scalar, and reporting a single average hides where it fails. Second, E0.3a's
finding stands whatever δ turns out to be — CSpD's rule is not computing this ratio at all.

**Revised C1, to be confirmed or killed by E0.3:** *exact acceptance in continuous token space is
governed by `δ√d/σ`; on real MAR draft/target pairs δ is large enough that exact acceptance is
impractical, and the only published method conceals this behind two undocumented departures from
the exact rule.* If E0.3 finds δ small and exact acceptance high, the first half is dead, the
second half survives as a correctness note, and the project retreats to the free-drafter
contribution as the risk table already provides for.

---

## 2. Method sketch

Notation: `z_i` = predictor output at masked position *i*; `E_ema` = EMA target encoder;
`x_i` = sampled token latent; `d` = token dim (16 for KL16 VAE); `k` = semantic dim.

**Draft.** For a block *B* of masked positions, form `x̂_B` with a cheap sampler off `z_B`:
either the predictor mean directly, or 1–2 Euler steps of the flow-matching head (vs. ~100 for
the target). Cost is one transformer pass, already paid.

**Verify.** Rather than testing `p(x̂_i)/q(x̂_i)` in *d* dimensions per token, embed the drafted
block and test it as a whole:

```
s(B) = D( E_ema(x̂_B) ,  z_B )
```

where `D` is the same discrepancy the model was trained on (`models/jepaloss.py`, smooth-L1 /
cosine). Accept the block if `s(B) ≤ τ`.

**Calibrate.** `τ` comes from conformal risk control on a held-out calibration set, chosen so
that the expected semantic deviation of accepted output from target-model output is `≤ α` with
finite-sample validity. On rejection, fall back to the exact target sampler for that block, so
quality degrades gracefully to the unaccelerated model rather than to garbage.

**What is genuinely new here** (see `RELATED_WORK.md` for the checks behind each):

- Using the JEPA predictor as a zero-cost drafter. ASSD does something structurally analogous
  for discrete any-subset ARMs via attention-mask switching; nobody has done it in continuous
  space, and nobody has used a JEPA predictor for it.
- Verifying in a learned semantic space instead of the token space. All of SJD, GSD, PathSpec,
  SSD, ARC-Decode verify over a codebook; CSpD verifies in token space. Neither option exists
  for us and neither is what we want.
- Conformal risk control over a *continuous semantic* nonconformity score. ARC-Decode is the
  prior art and is discrete-token, LLM-only, JS-divergence-on-logits.

---

## 3. Phases

Experiments tagged **[IP]** form the pixel-domain track. They exist because the claim that our
semantic statistic "preserves what users care about" is currently asserted from the JEPA framing
rather than measured, and because FID and CMMD are structurally incapable of measuring it. The
track is additive: every [IP] experiment either sharpens an existing gate or strengthens an
existing control, and none of it requires the method to work.

### Phase 0 — Reproduce and instrument (no new ideas)

Nothing gets built until the baselines run and the diagnostic lands.

- **E0.1** Run MAR-B/L and D-JEPA-B unmodified. Reproduce reported FID within tolerance.
  Record wall-time and a per-component NFE breakdown (transformer passes vs. denoising-head
  evaluations). *Gate: reproduction within 0.2 FID of published.*
- **E0.2** Run CSpD at its default `inner_temperature=42., ratio=0.15` and measure the claimed
  2.33× speedup. *Reframed after reading the source:* this is not a reproduction, because there is
  nothing to reproduce from. `SpecDecodeRunner` is never imported by `main_mar.py`; its only use in
  the repo is `demo/run_mar.ipynb`, and the repo ships no FID script or throughput benchmark for
  the speculative path. The harness has to be written before the number can be checked at all.
- **E0.3a** **Done (2026-09-02), no GPU.** Bound CSpD's spurious accumulated-variance factor from
  its diffusion schedule alone. Found a *second* departure from the exact acceptance rule,
  independent of the temperature and undocumented: `judge_accept` multiplies the ratio by
  `exp(Σ_steps Σ_channels (log σ₂ − log σ₁))`, a product of 1,584 terms bounded at e^81. A 5%
  disagreement between draft and target on predicted variance suffices for ~57×, where 2× would
  already saturate `min(1, ratio)`. See the log entry and `herald/verifiers/cspd_exact.py`.
- **E0.3** **The critical experiment.** Sweep `inner_temperature ∈ {1, 2, 5, 10, 20, 42}` on real
  MAR draft/target pairs. Report the four-way decomposition — `exact`, `temperature_only`,
  `fixed_term_only`, `as_shipped` — so the two defects are attributed separately, plus the measured
  distribution of the `fixed_term` exponent that E0.3a could only bound.

  *Design, revised for a GPU budget.* The temperature enters only at `judge_accept`, after both
  models' distributions are fixed (verified by call path). So one generation run can record every
  acceptance input, and the entire sweep becomes arithmetic on cached tensors: **six GPU runs
  collapse to one.** `scripts/run_e03.py capture` does the GPU half, `analyze` the free half.

  *What that trade costs, stated up front.* A cached sweep gives acceptance at each decision point
  along **one** trajectory. It does not give end-to-end speedup or FID per temperature, because
  accepting different tokens changes what gets drafted next. Acceptance rate is what C1 is about,
  so this is the right economy — but the speedup and quality columns need the expensive version and
  must not be quietly reported from the cheap one.

> **E0.3 is the fulcrum of the whole project.** If acceptance at temperature 1 is already high,
> C1 is dead, CSpD is genuinely lossless, and the motivation collapses. Run it in week 1. Do not
> write method code before it reports.
>
> E0.3a has *not* pre-empted this. It shows the shipped rule computes something other than the
> density ratio; it says nothing about what the exact ratio evaluates to on real model pairs. That
> number is still the one that decides the project.

- **E0.4 [IP]** **Pixel-domain audit of the temperature sweep.** Reuses E0.3's samples, adds no
  new sampling. At each temperature, decode both the accepted-draft latents and the latents the
  target sampler would have produced *from the same seed and the same reparameterization noise*,
  then compare them per image rather than in aggregate. Report paired PSNR / SSIM / MS-SSIM /
  LPIPS, the radially-averaged
  power spectrum of the residual, and where the residual concentrates spatially under a
  gradient-magnitude partition into edge, texture, and flat regions.

  *Prerequisite — verified 2026-09-02, the pairing is sound.* The paired decode is only meaningful
  if draft and target share reparameterization noise, so that the two outputs differ solely through
  the acceptance decision rather than through sampling variance. Confirmed in
  `spec_decode/spec_decode_runner.py`: `run()` draws `noise` and `inter_noise` **once** (lines
  101–103), before either model is called, and passes the same tensors into every
  `sample_tokens_step` for both draft and target. `p_sample_with_noise` then consumes
  `inter_noise[i]` as its only stochastic input, so at matched `(image, token, step)` both models
  see identical noise. `orders` is likewise drawn once and shared. E0.4's residual therefore
  measures acceptance cost, not sampling variance.

  *Why this is not redundant with the FID already in E0.3.* FID compares two 50k-sample bags and
  is insensitive by construction to per-image deviation that leaves the marginal distribution
  intact. CSpD's entire defense of temperature 42 is that FID barely moves. A paired decode
  measures exactly what FID cannot see, so this is the difference between "the outputs are still
  ImageNet-like" and "the outputs are the ones the target model would have produced."

  *Falsifier.* If paired LPIPS and the residual spectrum are flat across `temperature ∈ [1, 42]`,
  then the relaxation genuinely costs nothing perceptible, C1 weakens considerably even if
  acceptance rates move, and the argument has to retreat to the free-drafter contribution.

- **E0.5 [IP]** **KL16 decoder sensitivity.** Measure how a latent perturbation of norm ε maps to
  pixel deviation as a function of *direction*: the local Jacobian spectrum of the VAE decoder
  (randomized SVD, batched over tokens — `baselines/jepa-guidance` already has a randomized-SVD
  implementation with error bounds to borrow the approach from), plus the spatial and spectral
  profile of the decoded response along the top and bottom singular directions.

  *Why this is a method question and not a report section.* Every acceptance rule in this project
  thresholds an isotropic norm in latent space. If decoder gain varies strongly by direction, an
  isotropic threshold is the wrong shape, and the correct nonconformity score is Jacobian-weighted
  — a concrete, testable design consequence that changes `herald/calibration/`.

  *Falsifier.* If the Jacobian spectrum is close to flat, there is no reweighting to do, and E0.5
  reduces to a one-paragraph justification for keeping the isotropic score. Report it either way.

### Phase 1 — Establish the dimensional obstruction

Every Phase-1 sweep has the same shape: hold the model fixed, change *the space the acceptance
test lives in*, and measure acceptance at matched guarantee strength. E1.2 and E1.3 form a ladder
of controls running from no structure to learned structure, so that any gain credited to the JEPA
statistic has had to survive the cheaper explanations first.

- **E1.1** Time-series testbed (ETTh1, UCI Electricity), MAR-style continuous AR trained from
  scratch. Sweep token dim `d ∈ {4, 8, 16, 32, 64, 128}`. Plot exact-acceptance rate vs. *d*.
  Predicted: sharp decay. *This is the plot the paper is built around.*
- **E1.2a** Same sweep, verifying a random *k*-dim projection. The null control: isolates "low
  dimension helps" from "*this* projection helps."
- **E1.2b [IP]** Same sweep, verifying a **transform-domain statistic** — wavelet or
  steerable-pyramid subband energies, DCT coefficient statistics, the local mean/variance/
  covariance triple that SSIM is built from, radially-averaged spectral bands. Structured,
  interpretable, training-free, and matched in dimension *k* to E1.3. This is the strong control.
- **E1.3** Same, with the JEPA semantic projection.

Read the ladder as two gaps rather than one. **E1.2a → E1.2b** is the value of structure;
**E1.2b → E1.3** is the value of *learning* the structure. The original plan measured only the
second gap and measured it against the weakest available baseline, which would have overstated it.

*Falsifier, committed to in advance.* If E1.3 ≈ E1.2b, the JEPA framing is decoration. The honest
paper then becomes "a transform-domain acceptance rule accelerates continuous autoregressive
generation," with the D-JEPA reuse narrative dropped — a smaller result, but a true one and still
a contribution to an empty cell in the related-work table. Log it the day it happens rather than
tuning E1.3 until it wins.

*Testbed note.* E1.2b's transforms are defined on images, whereas E1.1–E1.3 run on the
time-series testbed where *d* can be swept freely. Use the 1-D analogues there (DWT subbands, DCT,
local moments, spectral bands), and repeat E1.2b against E1.3 on ImageNet latents in Phase 2 with
the genuinely 2-D transforms. Do not silently report the 1-D result as if it were the image one.

### Phase 2 — HERALD on ImageNet, inference-only

- **E2.1** Predictor-as-drafter, exact token-space verification. Isolates the free-drafter
  contribution from the semantic-verification contribution.
- **E2.2** Add semantic-space verification with a fixed threshold. Sweep block size.
- **E2.2b [IP]** The E1.2b-vs-E1.3 comparison on ImageNet latents with true 2-D transforms
  (separable DWT / steerable pyramid over the 16×16 token grid, block DCT, local moment maps).
  Token dim is fixed at `d=16` here, so this varies the semantic dim *k* and the choice of
  statistic rather than *d* — it is not a second dimension sweep. It has to be run anyway: this
  is the setting the headline numbers live in, and a transform-versus-learned verdict established
  only on 1-D time series would not transfer on its own.
- **E2.3** Add conformal calibration. Verify empirical coverage matches nominal α across
  ImageNet classes, and check it holds under class-conditional shift (the exchangeability
  assumption is doing real work here — probe it, don't assume it).
- **E2.4** Frontier plot: FID/CMMD vs. wall-time against MAR, D-JEPA, CSpD, **and few-step
  distillation**. Distillation is the honest competitor and the most likely way this project
  loses; it goes on the same axes from the start.
- **E2.5 [IP]** Run the E0.4 pixel audit on HERALD's own accepted blocks, as a function of the
  conformal risk level α and block size. This closes the loop the whole reframing rests on: we
  claim to control semantic deviation, so we have to show what that buys and what it costs in the
  pixel domain, at each α, against the target sampler at matched seed.

  *Falsifier, and the one most likely to fire.* If accepted blocks are semantically faithful but
  systematically lose high-frequency detail, then "preserves semantics, softens texture" is the
  real result. That is a weak selling point for image generation and it must be reported as a
  fidelity tradeoff curve rather than buried — the Jacobian-weighted score from E0.5 is the first
  remedy to try, not a reason to stop measuring.

### Phase 3 — Only if phases 0–2 hold

Video (V-JEPA 2 latents, where the semantic/pixel gap is widest and the argument is strongest),
and the SPLICE-style time-series application with an end-to-end conformal guarantee.

---

## 4. Metrics

Report all of these together; each one alone is gameable.

| Metric | Why |
| --- | --- |
| FID + CMMD | CMMD because FID is unreliable at the sample counts we can afford |
| NFE, split by component | The honest compute number; wall-time alone hides batching effects |
| Wall-time, fixed batch and hardware | What anyone actually cares about |
| Acceptance rate, per block size | The mechanism |
| Empirical coverage vs. nominal α | Whether the guarantee is real |
| Semantic deviation from target output | Direct measurement of what we claim to preserve |
| Paired PSNR / SSIM / MS-SSIM | Per-image agreement with the target sampler at matched seed |
| Paired LPIPS | Perceptual agreement; expect it to disagree with SSIM on generative output, and report both when it does |
| Residual power spectrum | Whether acceptance trades away a specific frequency band — high-frequency detail is the suspect |
| Residual spatial localization | Whether error concentrates on edges and texture or spreads over flat regions |

**Standing rule.** Any speedup claim is reported with its quality number attached, at matched
sampling settings, in the same table. A speedup without its FID is not a result.

**Third standing rule.** State which cost model a speedup is measured in. E0.1 found the denoising
head is 98.3% of wall time while performing 4.5× *fewer* token evaluations than the transformer,
so "NFE" and "seconds" are not interchangeable here and a result quoted in one will not hold in the
other. Wall-time and sequential head invocations are the primary figures; raw NFE goes beside them,
never instead of them.

**Second standing rule.** Distributional and paired metrics answer different questions and neither
substitutes for the other. FID and CMMD ask whether the output is still ImageNet-like; the paired
metrics ask whether it is the image the target model would have produced. Speculative decoding's
entire claim is about the second, so a table reporting only the first is not evidence for it.

---

## 5. Known risks

| Risk | Mitigation |
| --- | --- |
| **E0.3 kills C1** | Run it first. If CSpD is genuinely lossless at temp 1, pivot to the D-JEPA free-drafter contribution alone, which stands independently. |
| **Distillation dominates** | On the same plot from Phase 2. If it wins outright, the contribution becomes the diagnostic (C1) plus the negative result, published honestly and small. |
| **Semantic guarantee reads as a cop-out** | Pre-empt with the E0.3 evidence that the exact guarantee is not currently being delivered by anyone. Lead the paper with the diagnostic. |
| **Exchangeability violated under CFG** | CFG changes the sampling distribution, so a calibration set gathered at one CFG scale may not transfer. Test explicitly in E2.3; calibrate per-CFG-scale if needed. |
| **Scoop risk** | The space is moving fast (SSD, PathSpec, ARC-Decode all recent). Re-run the related-work sweep monthly; log it in `RELATED_WORK.md` with dates. **Partly realised 2026-09-02:** arXiv 2606.13426 published the closed-form acceptance collapse (for σ→0, in diffusion-step speculation) and a training-free "Free Drafter", and the 2026-08-29 sweep missed it. C1 repositioned around dimensionality and the continuous-token AR setting; widen the query set beyond continuous-token AR. |
| **δ turns out to be small** | The new top risk, from §1b. If MAR draft/target agree to within ~0.1σ per coordinate, exact acceptance at d=16 is ~84%, C1's dimensionality half dies, and CSpD's relaxation looks unnecessary rather than concealing. E0.3 measures δ directly and is cheap; run it before anything depends on the answer. |
| **Semantic acceptance costs high-frequency detail** | Genuinely likely, and it would damage C3. E0.4 detects it on CSpD before we build anything; E2.5 quantifies it for HERALD. If it fires, try the Jacobian-weighted score from E0.5, and failing that report an explicit semantic-fidelity vs. detail-fidelity tradeoff curve instead of a free speedup. |
| **Transform-domain control matches JEPA** | Not project-ending. Retarget to the transform rule, which is cheaper, training-free, and interpretable; the acceleration result survives and the "reuse the discarded verifier" narrative does not. Decide this on E1.2b/E2.2b, not on which story reads better. |
| **Transform-domain verification is already published** | Unverified as of the 2026-08-29 sweep, which did not look for it. Check before building E1.2b — search the image-quality-assessment and perceptual-metric literature as well as speculative decoding, since the idea could arrive from either side. |

---

## 6. Immediate next actions

1. ~~Finish checkpoint downloads~~ — done.
2. ~~E0.1 — instrument the NFE split~~ — done (partial; FID gate deferred, see E0.1b).
3. ~~E0.3a — bound the accumulated-variance factor~~ — done, no GPU.
4. **E0.3 capture — the temperature sweep.** Everything downstream is conditional on it. Harness is
   built and tested (`scripts/run_e03.py`); blocked only on a GPU. All eight A6000s on this host
   have been fully allocated to another user for the whole session (~985 MiB free of 48 GB).
5. E0.4 — the pixel audit, on E0.3's samples, in the same sitting.
6. E0.5 — KL16 decoder Jacobian spectrum. Gate-exempt, cheap, and it fits in the memory that *is*
   free, so it is the natural thing to run while waiting on the capture.
7. Related-work check on transform-domain verification before any E1.2b code.

**Standing note on the GPU budget.** Prefer designs that separate an expensive collection pass from
cheap offline analysis, as E0.3 now does. Before adding a GPU run, ask whether the quantity depends
on the model or only on what the model already produced.

Do not start Phase 2 code before E0.3 reports. E0.4 and E0.5 are exempt from that gate: they are
analysis of Phase-0 outputs and of a frozen public VAE, and neither presupposes that HERALD works.
