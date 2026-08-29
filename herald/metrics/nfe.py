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


def instrument_mar_sampler(model, ledger: NFELedger):
    """Patch a MAR/D-JEPA model in place so sampling populates ``ledger``.

    Returns a callable that restores the original methods. Only wraps methods that exist, so it
    works across the MAR and D-JEPA forks without branching on model class.
    """
    originals = {}

    def wrap(name: str, component: str, count_tokens):
        fn = getattr(model, name, None)
        if fn is None:
            return
        originals[name] = fn

        def wrapped(*args, **kwargs):
            with ledger.timed(component, count_tokens(args, kwargs)):
                return fn(*args, **kwargs)

        setattr(model, name, wrapped)

    def first_arg_rows(args, kwargs):
        t = args[0] if args else next(iter(kwargs.values()))
        return t.shape[0] if torch.is_tensor(t) else 1

    wrap("forward_encoder", "encoder", first_arg_rows)
    wrap("forward_decoder", "decoder", first_arg_rows)

    diffloss = getattr(model, "diffloss", None)
    if diffloss is not None and hasattr(diffloss, "sample"):
        originals["diffloss.sample"] = diffloss.sample
        inner = diffloss.sample

        def wrapped_sample(z, *args, **kwargs):
            with ledger.timed("diffusion_head", z.shape[0]):
                return inner(z, *args, **kwargs)

        diffloss.sample = wrapped_sample

    def restore():
        for name, fn in originals.items():
            if name == "diffloss.sample":
                model.diffloss.sample = fn
            else:
                setattr(model, name, fn)

    return restore
