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

With `num_iter=64` the transformer runs 64 times total; the denoising head runs
`64 × ~100 × (tokens per step)` times. Step 2 dominates, and step 1 already produces a semantic
prediction of every token it is about to spend step 2 computing.

**Implication.** The drafter is free. We are not adding a draft model; we are reading a tensor
that is currently thrown away. This is the structural difference from CSpD, which trains and
loads a second MAR.

Confirm this empirically before building on it — see E0.

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

### Phase 0 — Reproduce and instrument (no new ideas)

Nothing gets built until the baselines run and the diagnostic lands.

- **E0.1** Run MAR-B/L and D-JEPA-B unmodified. Reproduce reported FID within tolerance.
  Record wall-time and a per-component NFE breakdown (transformer passes vs. denoising-head
  evaluations). *Gate: reproduction within 0.2 FID of published.*
- **E0.2** Run CSpD at its default `inner_temperature=42., ratio=0.15`. Reproduce the claimed
  2.33× speedup.
- **E0.3** **The critical experiment.** Sweep `inner_temperature ∈ {1, 2, 5, 10, 20, 42}`.
  Record per-token acceptance rate, end-to-end speedup, and FID at each setting.

> **E0.3 is the fulcrum of the whole project.** If acceptance at temperature 1 is already high,
> C1 is dead, CSpD is genuinely lossless, and the motivation collapses. Run it in week 1. Do not
> write method code before it reports.

### Phase 1 — Establish the dimensional obstruction

- **E1.1** Time-series testbed (ETTh1, UCI Electricity), MAR-style continuous AR trained from
  scratch. Sweep token dim `d ∈ {4, 8, 16, 32, 64, 128}`. Plot exact-acceptance rate vs. *d*.
  Predicted: sharp decay. *This is the plot the paper is built around.*
- **E1.2** Same sweep, but verify a random *k*-dim projection instead of the full token. Isolates
  "low dimension helps" from "the learned projection helps."
- **E1.3** Same, with the JEPA semantic projection. The gap between E1.2 and E1.3 is the value of
  *learning* the statistic. If E1.3 ≈ E1.2, the JEPA framing is decoration and we should say so.

### Phase 2 — HERALD on ImageNet, inference-only

- **E2.1** Predictor-as-drafter, exact token-space verification. Isolates the free-drafter
  contribution from the semantic-verification contribution.
- **E2.2** Add semantic-space verification with a fixed threshold. Sweep block size.
- **E2.3** Add conformal calibration. Verify empirical coverage matches nominal α across
  ImageNet classes, and check it holds under class-conditional shift (the exchangeability
  assumption is doing real work here — probe it, don't assume it).
- **E2.4** Frontier plot: FID/CMMD vs. wall-time against MAR, D-JEPA, CSpD, **and few-step
  distillation**. Distillation is the honest competitor and the most likely way this project
  loses; it goes on the same axes from the start.

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

**Standing rule.** Any speedup claim is reported with its quality number attached, at matched
sampling settings, in the same table. A speedup without its FID is not a result.

---

## 5. Known risks

| Risk | Mitigation |
| --- | --- |
| **E0.3 kills C1** | Run it first. If CSpD is genuinely lossless at temp 1, pivot to the D-JEPA free-drafter contribution alone, which stands independently. |
| **Distillation dominates** | On the same plot from Phase 2. If it wins outright, the contribution becomes the diagnostic (C1) plus the negative result, published honestly and small. |
| **Semantic guarantee reads as a cop-out** | Pre-empt with the E0.3 evidence that the exact guarantee is not currently being delivered by anyone. Lead the paper with the diagnostic. |
| **Exchangeability violated under CFG** | CFG changes the sampling distribution, so a calibration set gathered at one CFG scale may not transfer. Test explicitly in E2.3; calibrate per-CFG-scale if needed. |
| **Scoop risk** | The space is moving fast (SSD, PathSpec, ARC-Decode all recent). Re-run the related-work sweep monthly; log it in `RELATED_WORK.md` with dates. |

---

## 6. Immediate next actions

1. Finish checkpoint downloads (`scripts/download_checkpoints.sh`).
2. E0.1 — reproduce MAR-B and D-JEPA-B, instrument the NFE split.
3. **E0.3 — the temperature sweep.** Everything downstream is conditional on it.

Do not start Phase 2 code before E0.3 reports.
