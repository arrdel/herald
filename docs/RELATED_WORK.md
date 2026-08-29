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

This is the single most important observation in the project so far, and it is the motivation
for E0.3. Two caveats before leaning on it publicly: (a) confirm the temperature is applied at
acceptance and not only inside a proposal step, by tracing the call path, and (b) run the sweep
and measure, rather than arguing from the source alone.

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
