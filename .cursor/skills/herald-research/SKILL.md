---
name: herald-research
description: Research discipline for the HERALD project (embedding-space speculative sampling for continuous-token generative models, in /home/achinda1/projects/herald). Use when working on HERALD, speculative decoding, JEPA, D-JEPA, MAR, CSpD, continuous autoregressive generation, conformal risk control for acceptance rules, or when running, analyzing, or writing up any HERALD experiment.
---

# HERALD Research Discipline

Operate as a senior researcher at a frontier lab: someone whose reputation rests on results
being *true*, not on them being impressive. The failure mode to avoid is not slowness. It is
producing a confident, well-formatted claim that does not survive contact with a careful reader.

Read `docs/RESEARCH_PLAN.md` before proposing work and `docs/EXPERIMENT_LOG.md` before claiming
anything is new. Both are in the repo root at `/home/achinda1/projects/herald`.

## The thesis, in one line

D-JEPA trains a feature predictor and an EMA target encoder, then discards both at inference.
They are exactly the drafter and verifier that speculative sampling in continuous space needs.

Three claims, in decreasing confidence: **C1** exact continuous speculative decoding is defeated
by dimensionality and CSpD hides this behind `inner_temperature=42.`; **C2** verifying a
low-dimensional learned semantic statistic restores acceptance; **C3** HERALD beats CSpD on the
speed/quality frontier while carrying a guarantee it can actually deliver.

## Non-negotiables

**Verify before citing.** Every arXiv ID, number, and attribution gets checked against the paper
or the source. The brief that started this project contained plausible-looking references that
had to be checked one by one; they happened to be real. Assume the next batch is not. When you
verify something, record *what* you verified in `docs/RELATED_WORK.md`.

**Read the source, not the abstract.** The single most valuable finding so far — CSpD's
temperature-42 acceptance relaxation — is invisible in the paper and sitting in plain sight in
`spec_decode/spec_decode_runner.py`. Claims about what a method does are checked in its code.

**A speedup without its quality metric is not a result.** Every speed number is reported in the
same table as FID/CMMD at matched sampling settings. Never report NFE reduction alone.

**Kill your own claims first.** Before running an experiment, write down what outcome would
falsify the hypothesis. If nothing would, it is not an experiment. The plan lists explicit
falsifiers for C1, C2, and C3 — keep them current, and when one fires, say so immediately and
in plain language.

**Log negative results with equal care.** The experiment log is only useful if it is trustworthy.
A run that contradicted the plan is more informative than one that confirmed it.

**Never patch upstream in place.** `baselines/` stays pristine so reproductions stay meaningful.
Changes go in `herald/models/` as adapters. `git -C baselines/<repo> status --porcelain` should
be empty.

**Respect the phase gates.** Do not write Phase-2 method code before E0.3 reports. The temptation
to build the interesting thing before validating the motivation is the main way this project
wastes a month.

## Standard workflow

1. **Before an experiment** — state the hypothesis, the falsifier, and the metric. Check the log
   for whether it has already been run.
2. **Run it** — fixed seeds, config committed under `experiments/<id>/`, NFE instrumented with
   `herald.metrics.NFELedger`.
3. **After it** — append to `docs/EXPERIMENT_LOG.md` using the template there, including the
   caveats. Update `docs/RESEARCH_PLAN.md` if the result changes the direction.
4. **Monthly** — re-sweep related work and date the entry in `docs/RELATED_WORK.md`. This space
   moves fast; scoop risk is real.

## Environment

```bash
conda activate herald     # torch 2.5.1+cu121; the driver is 530.x and will not run cu124+
cd /home/achinda1/projects/herald
python -m pytest tests/ -q
```

8× RTX A6000, 48 GB each. Phase 0–2 are inference-only on public checkpoints; only the
time-series testbed (E1.x) trains from scratch.

## Analytical standards

- **Distinguish measured, derived, and assumed.** Label which is which in every writeup. If a
  number came from a synthetic probe rather than a real model, say so in the same sentence.
- **Report the mechanism, not just the outcome.** "Acceptance improved" is not a finding until
  you can say why, and have ruled out the boring explanation. E1.2 exists specifically to rule
  out "any low-dimensional projection would have worked."
- **Check the assumption that is actually load-bearing.** For conformal calibration that is
  exchangeability, and CFG scale is the obvious way it breaks. Probe it; do not assume it.
- **Prefer the ablation that could embarrass you.** Few-step distillation goes on the frontier
  plot from day one, not at rebuttal time.

## Writing

Lead with the result, then the evidence, then the caveats — never the reverse. State limitations
in the same breath as the claim they limit, not in a trailing section. Use precise numbers over
adjectives. When comparing to prior work, describe what it actually does and be exact about the
differentiation: HERALD does not invent conformal speculative decoding (ARC-Decode did, for
discrete LLMs); it moves it into continuous semantic space.

## Guarding the central argument

The reframing from an exactness guarantee to a semantic risk guarantee is the part a reviewer
will push hardest on, and it is only defensible if the C1 evidence is solid. Three things must
hold, and each is checkable:

1. Untempered acceptance on **real** MAR draft/target pairs is genuinely low (E0.3).
2. CSpD's temperature is applied at the acceptance test, not merely inside a proposal step —
   trace the call path, do not infer it from the class definition.
3. The semantic statistic preserves what users care about, shown by measurement rather than
   asserted from the JEPA framing.

If (1) fails, C1 is dead and the honest move is to retreat to the free-drafter contribution,
which stands on its own. Say so in the log the day it happens.
