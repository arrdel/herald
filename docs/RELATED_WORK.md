# Related Work — verified landscape

Every entry here was checked against the paper or the source repo, not against a summary.
When adding an entry, record what you actually verified and how.

**Last full sweep:** 2026-08-29

---

## 1. The three pillars

### JEPA + generation

**D-JEPA — Denoising with a Joint-Embedding Predictive Architecture.**
[arXiv:2410.03755](https://arxiv.org/abs/2410.03755) · ICLR 2025 ·
[code](https://github.com/D-JEPA/djepa-imagenet) · cloned to `baselines/djepa-imagenet`

Context encoder, EMA target encoder, feature predictor. Two losses: per-token diffusion (or
flow-matching) loss on the masked tokens, plus a JEPA prediction loss regressing the predictor
output toward the EMA target features. Reinterprets masked image modeling as generalized
next-token prediction. Public checkpoints: B (212M, FID 3.40), L (687M, 2.32), H (1.4B, 2.04),
all without CFG.

*Verified in source:* `models/djepa.py::sample_tokens` calls `forward_encoder` →
`forward_decoder` → `diffloss.sample`. It never calls `forward_ema_encoder` and never evaluates
`jepaloss`. **The target encoder and the JEPA loss are trained and then unused at inference.**
This is the opening HERALD is built on.

**MAR — Autoregressive Image Generation without Vector Quantization.**
[arXiv:2406.11838](https://arxiv.org/abs/2406.11838) · [code](https://github.com/LTH14/mar) ·
cloned to `baselines/mar`

Introduced the per-token diffusion loss over continuous tokens. Both D-JEPA and CSpD are forks
of this codebase, so it is the common substrate and the natural ablation floor.

### Speculative sampling, continuous

**CSpD — Continuous Speculative Decoding for Autoregressive Image Generation.**
[arXiv:2411.11925](https://arxiv.org/abs/2411.11925) ·
[code](https://github.com/MarkXCloud/CSpD) · cloned to `baselines/CSpD`

The only continuous-space speculative decoding method. Contributions: an acceptance criterion
for diffusion output distributions, denoising trajectory alignment (via shared reparameterization
noise), token pre-filling, and acceptance-rejection sampling with an analytic upper bound to
dodge the normalizing integral. Claims 2.33× with preserved output distribution.

*Verified in source* — `spec_decode/spec_decode_runner.py`:

```python
class DistributionProcessor:
    def __init__(self, ..., temperature=42.):
    def x1_pdf_unormalize(self, x):
        return torch.exp(-0.5 * (((x - self.x1_mean) / self.x1_std / self.temperature) ** 2).sum(-1))
```

The acceptance ratio is computed on Gaussian densities whose exponent is divided by a
`temperature` defaulting to **42**. `SpecDecodeRunner` additionally defaults to `ratio=0.15`,
so only a fraction of steps are speculated at all.

Dividing the exponent by 42² flattens both densities toward uniform, driving the ratio toward 1
and acceptance toward certain — regardless of whether draft and target agree. The losslessness
proof holds for the untempered ratio; **the shipped default is not the object the proof covers.**

**Caveat (a) resolved, 2026-09-02 — the temperature is applied at the acceptance test.** Traced
by call path, not inferred: `SpecDecodeRunner.run` (line 190) constructs
`DistributionProcessor(..., temperature=self.temperature)` and line 200 calls
`processor.speculative_sample(draft_samples)`; `speculative_sample` (line 62) calls
`judge_accept`; `judge_accept` (lines 36–37) calls both `x1_pdf_unormalize` and
`x2_pdf_unormalize`, each of which divides the standardized residual by `self.temperature`. There
is no proposal-only interpretation available. Caveat (b) still stands — E0.3 must measure.

**New, 2026-09-02 — a second and independent deviation: the accumulated-variance factor.**
`judge_accept` returns

```python
p_x * self.fixed_term * self.x1_std.prod(-1) / (q_x * self.x2_std.prod(-1))
#     ^^^^^^^^^^^^^^^ = exp((x2_log_var - x1_log_var).sum(-1))
```

The correct ratio of the two final-step Gaussians is `p_unnorm * prod(s1) / (q_unnorm * prod(s2))`.
The `fixed_term` is an extra multiplicative factor on top of it.

**Corrected 2026-09-02 (second pass).** An earlier version of this note said the factor "has no
place in it." That was too strong, and the correction matters because it is the difference between a
claim a reviewer can dismiss and one they cannot. CSpD draws `noise` and `inter_noise` **once** and
shares them between the two models, so the object being verified is the whole denoising trajectory,
not the final-step Gaussian in isolation. The trajectory likelihood ratio is
`∏_t p(x_{t-1}|x_t) / q(x_{t-1}|x_t)`, and each Gaussian transition contributes a normalizer
`(|Σ_q,t| / |Σ_p,t|)^{1/2} = ∏_c σ_q,t,c / σ_p,t,c`. **A factor of this shape belongs.**
Fast-ARDiff (arXiv 2512.08537, §S3) derives it independently and states it in exactly this
direction, so anyone who has read that paper will have seen the factor endorsed.

**The defect is that CSpD applies it inverted.** The correct accumulated factor is
`∏_t (σ_q,t / σ_p,t)`; `fixed_term = exp((x2_log_var − x1_log_var).sum(-1))` with `x1 = q` (draft)
and `x2 = p` (target) is `∏_t (σ_p,t / σ_q,t)`, the reciprocal. The acceptance ratio is therefore
wrong by `fixed_term²`.

Two things make this a sign slip rather than a modelling choice, and both are visible in the single
line above. The *same* normalizer for the final step sits in the *same* expression the right way
round — `x1_std.prod(-1) / x2_std.prod(-1)`, i.e. `σ_q,1 / σ_p,1`. And the accumulated subtraction
is target-minus-draft where every other scale term in the expression is draft-over-target.
Pinned by tests in `tests/test_cspd_exact.py`; `cspd_log_ratio(..., fixed_term_sign=-1)` gives the
correct orientation, and `decompose_acceptance` now reports it as `corrected_trajectory` so the
sign error is separated from the legitimate trajectory term.

Despite the name, `x*_log_var` is not the log-variance of the final-step Gaussian. Tracing what
fills it: `MAR.get_distribution()` returns `diffloss.gen_diffusion.manager`, and in
`diffusion/gaussian_diffusion.py::p_sample_loop_progressive_with_noise` the manager is populated as

```python
if i == 0:                                  # final step
    self.manager.variance = out["vars"]     # = exp(0.5*log_variance) * temperature  -> a STD
else:                                       # every other step
    self.manager.log_var += out["log_variance"]   # = 0.5 * log_variance -> log SIGMA, accumulated
```

So `x*_log_var` is a sum of `log σ` over ~99 intermediate denoising steps and all 16 channels, and
`x*_std` is the final-step standard deviation (the attribute name says variance; the value is not).
`fixed_term` is therefore `exp(Σ_steps Σ_channels (log σ₂ − log σ₁))`, a product of roughly 1,584
per-step-per-channel ratios.

This is **not** inert. `diffusion/__init__.py::create_diffusion` defaults to `learn_sigma=True`
→ `ModelVarType.LEARNED_RANGE`, and `DiffLoss` builds its network with
`out_channels = target_channels * 2` precisely so the variance is predicted per token. Draft and
target are different networks, so their intermediate variance schedules differ and the factor
drifts away from 1, compounding over ~99 steps.

Reimplemented and decomposed in `herald/verifiers/cspd_exact.py`, with a test showing the factor
alone changes the accept decision when draft and target are the *same* distribution — the exact
ratio is 1 at every token, and CSpD's rule still rejects.

**Also new — the shipped rule cannot be evaluated at temperature 1 in its own arithmetic.**
`judge_accept` composes raw `torch.exp` calls in linear space. At `d=16` the quadratic form is
routinely large enough that numerator and denominator both underflow float32 to exactly zero,
giving `0/0`. Any temperature-1 acceptance measured with upstream's own code therefore risks
reporting a numerical artifact as a dimensional effect. Our E0.3 harness works in log space and
flags which tokens upstream would have zeroed (`underflows_in_linear_space`).

**Reproducibility note.** `SpecDecodeRunner` is never imported by `main_mar.py`; the only usage in
the repo is `demo/run_mar.ipynb`. There is no evaluation entry point, FID script, or throughput
benchmark for the speculative path, so the reported 2.33× cannot be reproduced from the repo
without writing the harness. The demo's configuration is draft `mar_base`, target `mar_huge`,
`ratio=0.05` (the class default is 0.15), `num_draft=4`, `cfg=4.0`, `cfg_schedule="constant"`.

#### Block Verification for Speculative Diffusions (arXiv 2606.13426) — **found 2026-09-02, missed by the 2026-08-29 sweep. The most consequential competitor so far.**

The 2026-08-29 sweep did not surface this and should have; it predates the sweep. Read in full, not
from the abstract. It changes what C1 can claim and how the "free drafter" framing has to be
positioned, so it is recorded in detail.

**What it is.** Speculative sampling for *diffusion models* (U-Net DDPM and LDM, CIFAR-10 through
ImageNet-256), speculating over **denoising steps of a whole image** rather than over
autoregressive tokens. Three contributions: (i) a `Decomposition` residual sampler
(Proposition 3.1) that reduces exact residual sampling in ℝ^d to a tractable **1-D** problem;
(ii) block verification adapted from LLMs, maximizing joint acceptance over a draft block;
(iii) the **Free Drafter**, a training-free self-speculative drafter that reuses scores already
computed during the parallel verification pass. Reported gains are modest — up to 6.3% over
existing speculative methods, 1.08×–2.34× over DDPM depending on step count.

**Where it collides with C1, and how much.** Their Eq. 45 gives the per-sample acceptance rate for
Gaussian sample verification in closed form:

> `Acceptance(k+1) = 2 − 2Φ( ½ · M(m_q, m_p; σ²I) )`, where `M` is the Mahalanobis distance.

and they observe that as `σ → 0` the distance diverges and **acceptance goes to zero**, "independent
of the form of the drafter."

This is the same formula C1 rests on, and it is now published. Since `M = ‖m_q − m_p‖ / σ`, a
per-coordinate mean gap δ over dimension d gives `M = δ√d / σ`, so acceptance decays in **both**
the σ→0 direction they analyze and the d→∞ direction HERALD is about. Honest reading:

- **Weakened.** "Exact acceptance collapses in continuous speculative sampling" is no longer an
  unpublished observation. C1 cannot be stated as a discovery of the collapse.
- **Still standing.** They analyze the noise-scale limit, for diffusion-step speculation, in pixel
  or 64×64×3 latent space. HERALD's claim is the *dimensionality* direction, in **continuous-token
  autoregressive** models (MAR/D-JEPA, d=16 per token × 256 tokens), where the per-token verifier is
  a different object entirely. Neither the setting nor the limit is the same.
- **Untouched, and now the sharper part of C1.** That the only published continuous-token AR method
  departs from the exact rule in two specific, undocumented ways (temperature 42 and the *inverted*
  accumulated-variance factor) is not in this paper or any other. E0.3a stands, in its corrected
  form.
- **Genuinely useful to us.** Eq. 45 gives a closed-form acceptance curve to plot the E0.3
  measurement against, at zero GPU cost. A measured curve that tracks the analytic one is a much
  stronger result than a measured curve alone. Caveat: Eq. 45 assumes draft and target share a
  covariance `σ²I`, which CSpD's per-token learned variances violate, so it is a reference curve
  rather than a prediction.

**Where it collides with the drafter story.** "Free Drafter" and HERALD's "the drafter is already
trained and thrown away" occupy overlapping rhetorical ground, and a reviewer will ask. The
mechanisms are not the same: theirs reuses the target model's own score from the verification pass
within a diffusion sampler (self-speculation, no second network); HERALD's is D-JEPA's *feature
predictor* with the *EMA target encoder* as verifier — two separately trained networks that the
upstream method discards at inference. Different model family, different objects. The distinction is
real but must be drawn explicitly rather than assumed.

**Also relevant.** Temperature-as-lenience is **not** novel and we must stop implying it is: their
Appendix I discusses it directly, citing Leviathan's `lq/p` relaxation and Tran-Thien (2023) on the
optimal lossy form. What remains specific to CSpD is the *magnitude* (42, undocumented) and the
separate arithmetic defect.

They also dismiss CSpD's family in passing — the MAR line "relies on rejection sampling with the
primary model as its proposal, which ends up being computationally inefficient."

**Actions.** Reposition C1 around dimensionality and the continuous-token AR setting, citing Eq. 45
rather than competing with it. Add their `Decomposition` sampler to the Phase-2 baseline set: it is
an exact residual sampler and its existence undercuts any argument that exactness is impractical
*per se*. Re-run the sweep monthly, and treat the 2026-08-29 miss as evidence the query set was too
narrow — it was scoped to continuous-token AR and missed diffusion-step speculation.

#### Fast-ARDiff (arXiv 2512.08537) — **found 2026-09-02. A first-order competitor in exactly our setting, and it bears on E0.3a.**

Zou, Ma, Huang, Yu, Zhao (USTC; JD). Builds directly on CSpD for MAR, TransDiff and NextStep-1. Two
mechanisms: a training-time loss forcing the draft's attention-map entropy to match the target's
(reported to cut the rejection rate by >40%), and an inference-time rule that stops speculating when
shallow-layer attention entropy falls below a calibrated threshold. With two-stage diffusion
distillation: **4.38×–4.88× on ImageNet 256 with FID essentially preserved** (MAR-L 1.78 → 1.77 at
4.38×; MAR-H 4.47× at FID 1.61).

- **This is the number C3 has to beat**, and it reports FID alongside speedup as we demand of
  ourselves. It is the strongest published speed/quality frontier in the continuous-token AR
  setting.
- **It also endorses CSpD's accumulated-variance factor.** Its §S3 states the acceptance ratio with
  `Σ = ∏_{t≥2} √(|Σ_q,t| / |Σ_p,t|)` and Algorithm 1 applies it in both the acceptance test and the
  residual sampler. This forced a correction to our E0.3a claim: the factor *belongs* (it is the
  trajectory normalizer), and CSpD's defect is that it applies the **reciprocal**. See the corrected
  note above. A reviewer who has read Fast-ARDiff would otherwise have caught this.
- **Not a threat to E1.2b.** Its gating statistic is a scalar attention entropy deciding *whether to
  speculate*, not a statistic being *verified*. No transform domain, no guarantee.

#### Transform-domain verification (E1.2b prerequisite) — clear in the literal form, but not an all-clear. Re-swept 2026-09-02 (18 queries).

E1.2b's specific construction — verifying a wavelet subband energy, DCT statistic, SSIM local-moment
triple, steerable-pyramid statistic or spectral band of a *continuous* token as the accept/reject
test in a draft-and-verify loop — remains unoccupied. Nothing found on either the speculative-
decoding side or the image-quality-assessment side. **Build it.** Three qualifications, each of
which changes how it must be written up rather than whether to run it:

- **The coarse-grained-verification construction is published, repeatedly, in discrete spaces, and
  we cannot claim it.** **PCG** (arXiv 2511.13732) defines a coarse random variable over acoustic
  similarity groups, splits probability mass across overlapping groups, runs rejection sampling on
  the *group label*, and proves exactness at the coarse level — structurally the theorem HERALD
  wants, and its related-work section explicitly claims the general move. **LANTERN**
  (arXiv 2410.03355, ICLR 2025) and **LANTERN++** (2502.06352) aggregate target mass over latent
  neighbours under a TV bound, and report untempered visual-AR acceptance at **0.0402** — useful
  external corroboration of the collapse premise. **Grouped Speculative Decoding** (ICCV 2025)
  proves clustering reduces TV and so raises acceptance. **CASCADE** (2605.07230) defines the
  equivalence class in the *verifier's* feature space. Our claim is the continuous-space
  instantiation: the coarsening is a dimension-reducing map rather than a partition of a finite
  vocabulary, so the aggregation is an intractable integral rather than a sum, and the
  acceptance-vs-dimension scaling has no analogue in a codebook. State that difference explicitly.
- **Transform-domain statistics already drive in-loop shortcut decisions.** **FreqCa**
  (2510.08669), **SpectralCache** (2603.05315) and **SeaCache** (2602.18993) decompose diffusion
  features by frequency and use per-band criteria to decide cache-or-recompute. No drafter, no
  verifier, no guarantee — the decision is a deterministic threshold on a heuristic. But "frequency
  bands of the representation tell you when computation can be shortcut" is established prior, not
  our insight. Our contribution is the *guarantee*, not the observation.
- **Cite ABC for the motivating logic.** Fearnhead & Prangle (JRSS-B 2012) prove ABC acceptance
  scales as `h^d` in the dimension of the matched statistic, and the field's twenty-year answer is
  to accept on a low-dimensional near-sufficient summary instead. No scoop — nothing to do with
  speculative decoding — but the *idea* "acceptance collapses with dimension, so accept on a
  summary" is not new as statistics, and a well-read reviewer will say so. It also predicts E1.2a:
  a random projection reduces `d` and therefore *must* raise acceptance, which is precisely why the
  null control exists and why E1.2b's bar is "beats a random projection at matched `k`".

Also surfaced: **SemanticSpec** (arXiv 2602.03708) runs the acceptance decision on learned semantic
probabilities from draft and target hidden states — verify-in-semantic-space with a learned
projection, i.e. **E1.3's idea in the text modality**. Logged here as a threat to E1.3's framing,
not E1.2b's. **COOL-SD** (2601.09212, AAAI 2026) and **Cactus** (2604.04987) give the general
relaxed-acceptance efficiency-vs-TV frontier our frontier plot will be compared against.
**MuLo-SD** (CVPR 2026, 2601.05149) is multi-scale in the *drafter*, not the verifier.
**Portilla & Simoncelli** (IJCV 2000) is the ancestor of E1.2b's premise — that a modest set of
subband statistics defines a perceptual equivalence class — and should be cited as the reason for
choosing *those* statistics, which pre-empts "why these 16 numbers and not any other 16".

*Coverage.* Web search only, no Scholar/Semantic Scholar/arXiv full-text. Two papers read in full
(PCG, Fast-ARDiff); the rest from abstracts plus retrieved method excerpts, so those verdicts are
**provisional** under our own read-the-source rule. Read GSD and CASCADE in full before writing
E1.2b. PCG's author list and venue could not be verified.

### Speculative sampling, discrete — why none of it transfers

All of the following depend on a categorical distribution over a finite codebook. Continuous
tokens have no codebook, no softmax, no top-*m*, no analytic ratio, so none of these methods port
over. Collectively they show the discrete side is crowded and the continuous side is nearly empty.

| Work | Idea | Blocker for us |
| --- | --- | --- |
| SJD (2410.01699) | Self-speculation via Jacobi iteration; previous verification output is the next draft | Codebook logits |
| Speculative Coupled Decoding (2510.24211) | Couples draft sampling across Jacobi iterations; ~3.8× image, ~10× video | Discrete coupling |
| PathSpec / PathRelax (2606.10492) | Parallel-path tree relaxation | Codebook-space similarity |
| SSD (2606.20543) | 2D spatial drafting, whole rows at once | Discrete verification |
| ASSD (ICLR 2026) | Any-subset ARM is its own drafter and density oracle via attention-mask switching | **Closest structural analogue to our free drafter** — but discrete, and it needs a fine-tune to get the joint-density head |
| SpecHub (2411.05289) | Multi-draft optimal transport; documents the O(\|V\|^{k+1}) blowup | Confirms the acceptance/dimensionality tension in the discrete setting |

### Risk-controlled acceptance

**ARC-Decode — Accelerated Decoding with Risk-Bounded Acceptance.** ICML 2026 ·
[code](https://github.com/NeuraLiying/ARC-Decode)

Closest prior art on the guarantee axis, and the one to cite carefully. Training-free relaxed
acceptance on top of EAGLE-3: a Local Tolerance Score calibrated by conformal risk control,
relaxing rejections whose expected local JS divergence is bounded. Up to 1.6× over EAGLE-3.

*Differentiation.* Discrete LLM tokens; nonconformity score is JS divergence between categorical
logits; risk is local distributional deviation at a token. HERALD is continuous, scores a
learned semantic embedding residual, and controls deviation of a *semantic statistic of a block*.
Same statistical tool, different object. **Do not claim conformal speculative decoding as novel** —
claim it in continuous semantic space.

**EARS (2512.13194)** — entropy-scaled adaptive acceptance threshold. Heuristic, no
finite-sample guarantee. **When Is a Draft Accepted? (2606.30265)** — KL-based acceptance
certificates for relaxed/greedy/tree rules; useful theory scaffolding, discrete.

### JEPA latents as a generative substrate

**SPLICE — Latent Diffusion over JEPA Embeddings for Conformal Time-Series Inpainting.**
[arXiv:2605.00126](https://arxiv.org/abs/2605.00126)

JEPA encoder → 64-dim latent → conditional bridge with four sampling modes → hourly-conditioned
decoder → Adaptive Conformal Inference. Flow-matching variant matches DDIM in 5–10 Euler steps
(5–10×). 93–95% empirical coverage. Pooled encoder transfers across feeds.

Relevant twice over: it is direct evidence that a 64-dim JEPA latent is a sufficient generative
substrate for a real domain, and its setup is the template for our secondary testbed. Note the
distinction — SPLICE puts conformal bands on the *output*; HERALD puts conformal control on the
*acceptance decision*.

**Beyond Generative Priors: Minority Sampling with JEPA-Guided Diffusion.**
[arXiv:2605.24631](https://arxiv.org/abs/2605.24631) · ICML 2026 ·
[code](https://github.com/soobin-um/jepa-guidance) · cloned to `baselines/jepa-guidance`

JEPA-SCORE = log-sum of the encoder Jacobian's singular values, via randomized SVD with error
bounds; used to steer diffusion toward semantically rare regions. Envelope theorem avoids
backprop through the SVD.

Load-bearing for us: it establishes that **JEPA embeddings carry usable density information**,
which is the premise behind verifying in that space. Different goal (rarity-seeking guidance vs.
acceleration), but if the density signal were not there, C2 would have no basis.

**DLLM-JEPA.** [arXiv:2606.00091](https://arxiv.org/abs/2606.00091) · ICML 2026

JEPA for masked diffusion LMs; two views from one input via different mask rates, 33% fewer
training FLOPs than LLM-JEPA. Training-time work, orthogonal to our inference-time question, but
it is the other half of the "Speculative JEPA Diffusion" umbrella and a natural Phase-3 direction
if we want a text modality.

---

## 2. The gap, in one table

| | Discrete tokens | Continuous tokens |
| --- | --- | --- |
| **Exact verification** | Solved; many variants | CSpD only, and its default relaxation voids exactness |
| **Self/free drafting** | SJD, ASSD, PathSpec | **Empty** |
| **Tree / block verification** | SSD, PathSpec | **Empty** |
| **Risk-controlled acceptance** | ARC-Decode, EARS | **Empty** |
| **Semantic-space verification** | **Empty** | **Empty** |

HERALD targets the last three cells in the continuous column, using components that already
exist in D-JEPA checkpoints.

---

## 3. Sweep log

| Date | Scope | Outcome |
| --- | --- | --- |
| 2026-08-29 | Initial sweep: D-JEPA, CSpD, SPLICE, DLLM-JEPA, JEPA-guidance, discrete spec-decode family, conformal spec-decode | All papers in the source brief verified real. Gap confirmed. CSpD temperature-42 finding recorded. |

**Open, not yet searched.** The 2026-08-29 sweep covered speculative decoding and JEPA generation
only. It did **not** look for prior work on verifying or accepting in a *transform domain*
(wavelet / DCT / SSIM-statistic / spectral), which E1.2b now depends on being novel. Search both
sides — the speculative-decoding literature and the image-quality-assessment and perceptual-metric
literature, where the idea could plausibly have arrived first under different vocabulary. Do this
before writing E1.2b code, and record the outcome here whichever way it lands.
