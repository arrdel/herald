"""Conformal risk control for acceptance thresholds.

Chooses the largest (most permissive) acceptance threshold whose expected loss on unseen data is
provably at most ``alpha``. Follows Angelopoulos et al., "Conformal Risk Control" (arXiv:2208.02814):
for a loss that is monotone non-increasing in the threshold parameter and bounded above by B, the
empirical-risk rule below controls E[L] <= alpha with finite-sample validity and no distributional
assumptions beyond exchangeability of calibration and test data.

Exchangeability is the assumption that can actually fail here -- see E2.3, which probes whether a
calibration set gathered at one CFG scale transfers to another.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RiskControlledThreshold:
    """Calibrated acceptance threshold with a finite-sample risk guarantee.

    Attributes:
        tau: Calibrated threshold. Accept when the nonconformity score is ``<= tau``.
        alpha: Target risk level.
        empirical_risk: Calibration-set risk at ``tau``.
        n_calib: Calibration set size.
    """

    tau: float
    alpha: float
    empirical_risk: float
    n_calib: int


def calibrate_threshold(
    scores: np.ndarray,
    losses: np.ndarray,
    alpha: float,
    loss_bound: float = 1.0,
    n_grid: int = 512,
) -> RiskControlledThreshold:
    """Calibrate an acceptance threshold to control expected loss at level ``alpha``.

    Args:
        scores: ``(n,)`` nonconformity scores; lower means the draft is more trustworthy.
        losses: ``(n,)`` loss incurred by accepting each draft, in ``[0, loss_bound]``.
            Typically the semantic deviation between the accepted output and the target
            model's output.
        alpha: Target expected loss. Must exceed ``loss_bound / (n + 1)``, the floor imposed by
            the finite-sample correction.
        loss_bound: Upper bound B on the loss.
        n_grid: Number of candidate thresholds swept over the observed score range.

    Returns:
        The calibrated threshold. Falls back to the most conservative candidate (rejecting
        everything) when no threshold satisfies the constraint.

    Raises:
        ValueError: If inputs are malformed or ``alpha`` is below the achievable floor.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    losses = np.asarray(losses, dtype=np.float64).ravel()
    if scores.shape != losses.shape:
        raise ValueError(f"scores {scores.shape} and losses {losses.shape} must have equal shape")
    n = scores.size
    if n == 0:
        raise ValueError("calibration set is empty")
    if np.any(losses < 0) or np.any(losses > loss_bound):
        raise ValueError(f"losses must lie in [0, {loss_bound}]")

    floor = loss_bound / (n + 1)
    if alpha <= floor:
        raise ValueError(
            f"alpha={alpha:g} unreachable with n={n} calibration points "
            f"(floor is B/(n+1)={floor:.4g}); lower alpha needs more calibration data"
        )

    # Rearranged risk-control condition: the empirical mean over n points plus the worst-case
    # contribution B of one unseen point must stay under (n + 1) * alpha.
    budget = ((n + 1) * alpha - loss_bound) / n

    candidates = np.quantile(scores, np.linspace(0.0, 1.0, n_grid))
    # Descending, so the first feasible candidate is the most permissive one.
    for tau in np.unique(candidates)[::-1]:
        accepted = scores <= tau
        # Rejected drafts fall back to the exact target sampler and incur zero loss.
        risk = float(losses[accepted].sum() / n)
        if risk <= budget:
            return RiskControlledThreshold(float(tau), alpha, risk, n)

    return RiskControlledThreshold(float(-np.inf), alpha, 0.0, n)


def empirical_coverage(scores: np.ndarray, losses: np.ndarray, tau: float) -> dict:
    """Evaluate a calibrated threshold on held-out data.

    E2.3 compares ``mean_risk`` against the nominal ``alpha``. A gap indicates the
    exchangeability assumption is not holding.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    losses = np.asarray(losses, dtype=np.float64).ravel()
    accepted = scores <= tau
    return {
        "acceptance_rate": float(accepted.mean()),
        "mean_risk": float(losses[accepted].sum() / scores.size),
        "mean_loss_given_accept": float(losses[accepted].mean()) if accepted.any() else 0.0,
        "n": int(scores.size),
    }
