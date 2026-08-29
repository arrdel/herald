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

## Next up

- **E0.1** Reproduce MAR-B/L and D-JEPA-B; instrument the NFE split with `herald.metrics.NFELedger`.
- **E0.2** Reproduce CSpD's 2.33× at its defaults.
- **E0.3** The temperature sweep on real models. Blocks everything downstream.
