"""NFE accounting for continuous-token samplers.

Wall-time alone conflates algorithmic savings with batching effects, so every speedup claim in
this project is reported alongside a per-component NFE breakdown. The split that matters is
transformer passes (shared across all masked positions in a step) versus denoising-head
evaluations (paid per token per diffusion step) -- the latter dominates in MAR/D-JEPA sampling.
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch


@dataclass
class NFELedger:
    """Counts model evaluations, keyed by component.

    ``token_evals`` weights each call by the number of tokens it processed, which is the
    quantity that actually scales cost for the per-token denoising head.
    """

    calls: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    token_evals: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    seconds: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def record(self, component: str, n_tokens: int = 1, elapsed: float = 0.0) -> None:
        self.calls[component] += 1
        self.token_evals[component] += int(n_tokens)
        self.seconds[component] += elapsed

    @contextmanager
    def timed(self, component: str, n_tokens: int = 1):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.record(component, n_tokens, time.perf_counter() - start)

    @property
    def total_token_evals(self) -> int:
        return sum(self.token_evals.values())

    def merge(self, other: NFELedger) -> None:
        for comp, n in other.calls.items():
            self.calls[comp] += n
        for comp, n in other.token_evals.items():
            self.token_evals[comp] += n
        for comp, s in other.seconds.items():
            self.seconds[comp] += s

    def as_dict(self) -> dict:
        return {
            "calls": dict(self.calls),
            "token_evals": dict(self.token_evals),
            "seconds": dict(self.seconds),
            "total_token_evals": self.total_token_evals,
        }

    def report(self) -> str:
        rows = ["component            calls    token-evals    seconds    %time"]
        total_s = sum(self.seconds.values()) or 1.0
        for comp in sorted(self.seconds, key=lambda c: -self.seconds[c]):
            rows.append(
                f"{comp:<20} {self.calls[comp]:>6} {self.token_evals[comp]:>14} "
                f"{self.seconds[comp]:>10.2f} {100 * self.seconds[comp] / total_s:>7.1f}"
            )
        rows.append(f"{'TOTAL':<20} {'':>6} {self.total_token_evals:>14} {total_s:>10.2f}")
        return "\n".join(rows)


#: The two forks name their transformer stages differently. MAR uses ``forward_mae_encoder`` /
#: ``forward_mae_decoder``; D-JEPA renamed them to ``forward_encoder`` / ``forward_decoder``.
#: Wrapping only one convention silently records nothing on the other fork, which would leave the
#: transformer half of the split empty and make the denoising head look like 100% of the work.
_ENCODER_NAMES = ("forward_mae_encoder", "forward_encoder")
_DECODER_NAMES = ("forward_mae_decoder", "forward_decoder")


def _token_count(t) -> int:
    """Token evaluations represented by a transformer input.

    A sequence tensor ``(batch, positions, width)`` is ``batch * positions`` token evaluations, not
    ``batch``. Counting rows would put the transformer in units of *sequences* while the denoising
    head is counted in *token-diffusion-steps*, and the resulting ratio -- which is what the
    free-drafter claim is argued from -- would be meaningless.
    """
    if not torch.is_tensor(t):
        return 1
    if t.dim() >= 3:
        return int(t.shape[0] * t.shape[1])
    return int(t.shape[0])


def diffusion_steps_per_call(diffloss) -> int:
    """How many denoising evaluations one ``diffloss.sample`` call actually performs.

    ``DiffLoss.sample`` runs ``gen_diffusion.p_sample_loop``, which evaluates the denoising MLP
    once per respaced timestep -- around 100 by default. Counting the call as a single evaluation
    per token understates the head by two orders of magnitude and would make the free-drafter
    argument look far weaker than it is, so the multiplier is read off the diffusion object rather
    than assumed.
    """
    gen = getattr(diffloss, "gen_diffusion", None)
    steps = getattr(gen, "num_timesteps", None)
    if not isinstance(steps, int) or steps < 1:
        raise ValueError(
            "could not read the denoising step count from diffloss.gen_diffusion.num_timesteps; "
            "refusing to report an NFE breakdown that would silently be ~100x too small"
        )
    return steps


def instrument_mar_sampler(model, ledger: NFELedger):
    """Patch a MAR/D-JEPA model in place so sampling populates ``ledger``.

    Returns a callable that restores the original methods. Handles both forks' naming.

    Under classifier-free guidance every stage runs on a doubled batch, and that doubling is
    recorded rather than divided out: it is real compute that any honest speedup comparison has to
    account for.
    """
    originals: dict[str, object] = {}
    # Methods that live on the class rather than the instance must be *deleted* on restore, not
    # reassigned. Reassigning leaves an instance attribute shadowing the class method, so a second
    # instrument/restore cycle would wrap the already-wrapped copy and double-count.
    was_instance_attr: dict[str, bool] = {}

    def wrap_first(names: tuple[str, ...], component: str) -> None:
        for name in names:
            fn = getattr(model, name, None)
            if fn is None:
                continue
            originals[name] = fn
            was_instance_attr[name] = name in vars(model)

            def wrapped(*args, _fn=fn, _component=component, **kwargs):
                t = args[0] if args else next(iter(kwargs.values()))
                with ledger.timed(_component, _token_count(t)):
                    return _fn(*args, **kwargs)

            setattr(model, name, wrapped)
            return

    wrap_first(_ENCODER_NAMES, "encoder")
    wrap_first(_DECODER_NAMES, "decoder")

    diffloss = getattr(model, "diffloss", None)
    if diffloss is not None and hasattr(diffloss, "sample"):
        originals["diffloss.sample"] = diffloss.sample
        inner = diffloss.sample
        steps = diffusion_steps_per_call(diffloss)

        def wrapped_sample(z, *args, **kwargs):
            with ledger.timed("diffusion_head", z.shape[0] * steps):
                return inner(z, *args, **kwargs)

        diffloss.sample = wrapped_sample

    def restore():
        for name, fn in originals.items():
            if name == "diffloss.sample":
                model.diffloss.sample = fn
            elif was_instance_attr.get(name, True):
                setattr(model, name, fn)
            else:
                delattr(model, name)

    return restore
