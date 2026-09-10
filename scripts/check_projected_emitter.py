"""Which distribution does projected verification actually emit?

A reviewer proposed formalising the fidelity leakage of projected verification through the
idealised emitter

    p~_A(s, u) = p_A(s) q(u | s),          s = Ax,  u = A_perp x,

for which KL(p || p~_A) = E_{p_A}[ KL(p(u|s) || q(u|s)) ] exactly, so that the KL chain rule reads
KL(p||q) = KL(p_A||q_A) + KL(p||p~_A).  That is a clean statement, but it is not obviously the
scheme this project measures: ``herald.eval.fidelity._emit`` applies the *projected* acceptance
test and then, on rejection, draws from the *full-space* residual (p - q)_+ rather than from the
projected residual.  The two differ, and the paper must not attribute to its experiment a
decomposition that belongs to a different algorithm.

This script measures both emitters on a shared-covariance Gaussian pair, where every quantity the
theory predicts is available in closed form, and reports:

  * the excess NLL under p of each emitter (the paper's fidelity metric),
  * the acceptance rate of the projected test against 2 - 2 Phi(||A delta|| / 2 sigma),
  * the idealised leakage ||A_perp delta||^2 / 2 sigma^2 predicted by the conservation identity.

Run: python scripts/check_projected_emitter.py
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

RNG = np.random.default_rng(20260910)


def log_gauss(x: np.ndarray, mu: np.ndarray, sigma: float) -> np.ndarray:
    """Log density of N(mu, sigma^2 I), evaluated rowwise."""
    d = x.shape[-1]
    z = x - mu
    return -0.5 * (z * z).sum(-1) / sigma**2 - d * np.log(sigma) - 0.5 * d * np.log(2 * np.pi)


def emit_idealised(n: int, mu_p, mu_q, sigma: float, k: int) -> np.ndarray:
    """s ~ p_A, then u ~ q(u|s).

    For a shared isotropic covariance and A the first k coordinates, the complement is independent
    of s under both models, so q(u|s) = N(mu_q[k:], sigma^2 I) and the emitter is simply the target
    mean on the verified block and the draft mean on the rest.
    """
    d = mu_p.shape[0]
    x = np.empty((n, d))
    x[:, :k] = mu_p[:k] + sigma * RNG.standard_normal((n, k))
    x[:, k:] = mu_q[k:] + sigma * RNG.standard_normal((n, d - k))
    return x


def emit_implemented(n: int, mu_p, mu_q, sigma: float, k: int, max_tries: int = 200):
    """The scheme in herald.eval.fidelity: projected accept test, full-space residual on reject."""
    d = mu_p.shape[0]
    x = mu_q + sigma * RNG.standard_normal((n, d))

    # Projected acceptance test on s = Ax (A = first k coordinates).
    lr = log_gauss(x[:, :k], mu_p[:k], sigma) - log_gauss(x[:, :k], mu_q[:k], sigma)
    accepted = RNG.random(n) < np.minimum(1.0, np.exp(lr))

    # Full-space residual (p - q)_+ by rejection, proposing from p -- mirrors residual_sample.
    out = x.copy()
    need = ~accepted
    pending = need.copy()
    fallback = mu_p + sigma * RNG.standard_normal((n, d))
    out[need] = fallback[need]
    for _ in range(max_tries):
        if not pending.any():
            break
        idx = np.flatnonzero(pending)
        prop = mu_p + sigma * RNG.standard_normal((idx.size, d))
        alpha = np.clip(
            1.0 - np.exp(log_gauss(prop, mu_q, sigma) - log_gauss(prop, mu_p, sigma)), 0.0, None
        )
        take = RNG.random(idx.size) < alpha
        out[idx[take]] = prop[take]
        pending[idx[take]] = False
    return out, accepted.mean(), (~pending | ~need).mean()


def main() -> None:
    sigma = 1.0
    n = 400_000
    print(f"{'d':>4} {'k':>3} {'M':>6} {'A_meas':>8} {'A_pred':>8} "
          f"{'xNLL_ideal':>11} {'xNLL_pred':>10} {'xNLL_impl':>10}")
    for d in (16, 64, 128):
        for k in (1, 4, 16):
            if k > d:
                continue
            # Per-coordinate gap held at the value the trained sweep measures.
            delta_per = 0.578
            mu_q = np.zeros(d)
            mu_p = np.full(d, delta_per)
            delta = mu_p - mu_q
            M = np.linalg.norm(delta) / sigma
            a_norm = np.linalg.norm(delta[:k])
            perp_norm = np.linalg.norm(delta[k:])

            acc_pred = 2 - 2 * norm.cdf(a_norm / (2 * sigma))
            # Idealised leakage from the conservation identity, in nats.
            leak_pred = perp_norm**2 / (2 * sigma**2)

            ref = mu_p + sigma * RNG.standard_normal((n, d))
            base = (-log_gauss(ref, mu_p, sigma)).mean()

            xi = emit_idealised(n, mu_p, mu_q, sigma, k)
            xnll_ideal = (-log_gauss(xi, mu_p, sigma)).mean() - base

            xm, acc_meas, conv = emit_implemented(n, mu_p, mu_q, sigma, k)
            xnll_impl = (-log_gauss(xm, mu_p, sigma)).mean() - base

            print(f"{d:4d} {k:3d} {M:6.2f} {acc_meas:8.4f} {acc_pred:8.4f} "
                  f"{xnll_ideal:11.4f} {leak_pred:10.4f} {xnll_impl:10.4f}")

    print("\nxNLL_ideal should match xNLL_pred = ||A_perp delta||^2 / 2 sigma^2 (the conservation")
    print("identity's leakage term). xNLL_impl is what this project actually measures.")


if __name__ == "__main__":
    main()
