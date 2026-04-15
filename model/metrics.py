"""
model.metrics
~~~~~~~~~~~~~
Mechanism quality metrics — the primary empirical validation targets.

Formal model reference: §6 (Mechanism Quality Metrics)

Metrics:
    signal_accuracy(signals, t)     → ρ(t): Spearman correlation of stake vs quality
    curator_roi(earnings, stake)    → ROI_i = (E_i - v_i) / v_i
    consumer_surplus(feed, t, k)    → CS(t): aggregate consumer utility
    participation_rate(results)     → fraction of curators with ROI > 0
"""
from __future__ import annotations

from typing import Callable, Optional
import numpy as np

from .agents import Signal, CuratorResult
from .staking import compute_payouts
from .ranking import feed_order
from .weight_functions import WeightFn, accumulation


# ---------------------------------------------------------------------------
# Signal accuracy  ρ(t)
# ---------------------------------------------------------------------------

def signal_accuracy(signals: list[Signal], t: float) -> float:
    """ρ(t): Spearman rank correlation between total stake and true quality.

    A mechanism with ρ(t) → 1 perfectly aligns curation with quality.
    ρ(t) ≈ 0 means staking is uninformative.

    Only signals published before t with at least one stake are included.

    Args:
        signals: List of Signal objects with known quality values.
        t:       Evaluation time.

    Returns:
        Spearman ρ ∈ [-1, 1], or NaN if fewer than 2 signals qualify.
    """
    eligible = [s for s in signals if s.publish_time <= t and s.n_curators > 0]
    if len(eligible) < 2:
        return float("nan")

    stakes = np.array([
        sum(sk.amount for sk in s.stakes if sk.time <= t)
        for s in eligible
    ])
    qualities = np.array([s.quality for s in eligible])

    # Spearman correlation via rank transform
    stake_ranks = _rank(stakes)
    quality_ranks = _rank(qualities)
    return float(np.corrcoef(stake_ranks, quality_ranks)[0, 1])


# ---------------------------------------------------------------------------
# Curator ROI
# ---------------------------------------------------------------------------

def curator_roi(earnings: float, stake: float) -> float:
    """ROI_i = (E_i - v_i) / v_i for a single curator.

    Args:
        earnings: E_i — total earnings.
        stake:    v_i — amount staked.

    Returns:
        ROI as a decimal (0.0 = break-even, 1.0 = doubled stake, -1.0 = total loss).

    Raises:
        ValueError: if stake <= 0.
    """
    if stake <= 0:
        raise ValueError(f"Stake must be > 0, got {stake}")
    return (earnings - stake) / stake


def participation_rate(results: list[CuratorResult]) -> float:
    """Fraction of curators who recouped their stake (ROI > 0).

    The participation constraint (§6.3) requires E[ROI_i] > 0 for rational
    agents to stake. This metric measures it empirically across a simulation run.

    Args:
        results: Output of compute_payouts().

    Returns:
        Float in [0, 1].
    """
    if not results:
        return 0.0
    return sum(1 for r in results if r.profitable) / len(results)


def roi_by_arrival(results: list[CuratorResult]) -> tuple[list[float], list[float]]:
    """Return (arrival_fraction, roi) arrays for plotting the ROI phase diagram.

    Arrival fraction = arrival_index / (n - 1), so first curator = 0.0,
    last = 1.0.

    Args:
        results: Output of compute_payouts().

    Returns:
        (arrival_fractions, rois): parallel lists.
    """
    n = len(results)
    if n == 0:
        return [], []
    fractions = [r.arrival_index / max(n - 1, 1) for r in results]
    rois = [r.roi for r in results]
    return fractions, rois


# ---------------------------------------------------------------------------
# Consumer surplus  CS(t)
# ---------------------------------------------------------------------------

def consumer_surplus(
    signals: list[Signal],
    t: float,
    weight_fn: WeightFn = accumulation,
    search_cost_fn: Optional[Callable[[int], float]] = None,
    n_consumers: int = 100,
    top_k: int = 20,
) -> float:
    """CS(t): aggregate consumer surplus at time t.

    Models n_consumers each consuming the top-k ranked signals.

    U_u(S, t) = q(S) - c(r(S, t))
    CS(t)     = Σ_u Σ_S 1[u consumes S] · U_u(S, t)

    Args:
        signals:         All signals available at time t.
        t:               Evaluation time.
        weight_fn:       Feed ranking weight function.
        search_cost_fn:  c(r) — search cost at rank r. Defaults to κ·log(r) with κ=0.05.
        n_consumers:     Number of consumers (scales CS linearly).
        top_k:           Each consumer reads the top-k signals.

    Returns:
        Aggregate consumer surplus (higher = better mechanism performance).
    """
    if search_cost_fn is None:
        import math
        search_cost_fn = lambda r: 0.05 * math.log(r)

    eligible = [s for s in signals if s.publish_time <= t]
    if not eligible:
        return 0.0

    ordered = feed_order(eligible, t, weight_fn)
    top_signals = ordered[:top_k]

    total_utility = 0.0
    for rank, signal in enumerate(top_signals, start=1):
        per_consumer = signal.quality - search_cost_fn(rank)
        total_utility += max(per_consumer, 0.0)  # consumers won't consume negative-utility items

    return total_utility * n_consumers


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rank(arr: np.ndarray) -> np.ndarray:
    """Return ordinal ranks (1-based) for a numpy array."""
    order = arr.argsort()
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(arr) + 1)
    return ranks.astype(float)
