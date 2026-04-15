"""
model.staking
~~~~~~~~~~~~~
Core payout mechanics for the staking game.

Formal model reference: §3 (The Staking Game)

Key equations implemented:
    π(i,j) = v_j · α · (v_i / V_j⁻)          [Individual payout, Def 3.3]
    E_i    = α · v_i · Σ_{j>i} v_j / V_j⁻    [Total curator earnings, Def 3.4]
    Π_P    = β · V_n                            [Platform revenue]
    Π_K    = v_1(1-β) + Σ_{j≥2} v_j · γ       [Creator revenue]
    Π_C    = α · (V_n - v_1)                   [Total curator revenue]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .agents import Signal, CuratorResult


def payout(i: int, j: int, signal: Signal, alpha: float) -> float:
    """π(i, j): payout to prior curator i when curator j stakes.

    Args:
        i:      0-based index of the receiving curator (must be < j).
        j:      0-based index of the staking curator (j > i).
        signal: Signal containing the ordered stake list.
        alpha:  Curator pool share ∈ (0, 1).

    Returns:
        Payout amount ≥ 0.

    Raises:
        ValueError: if i >= j or indices are out of range.
    """
    if i >= j:
        raise ValueError(f"Receiving curator index i={i} must be < staking index j={j}")
    stakes = signal.stakes
    v_j = stakes[j].amount
    V_j_minus = signal.cumulative_pool_before(j)
    if V_j_minus == 0:
        return 0.0
    v_i = stakes[i].amount
    return v_j * alpha * (v_i / V_j_minus)


def compute_payouts(signal: Signal, alpha: float, beta: float) -> list[CuratorResult]:
    """Compute per-curator earnings E_i for all curators on a signal.

    Implements the full payout distribution described in §3.3–3.4.

    Args:
        signal: Signal with an ordered stakes list.
        alpha:  Curator pool share ∈ (0, 1). Fraction of each new stake
                redistributed to prior curators.
        beta:   Platform share ∈ (0, 1). Must satisfy alpha + beta < 1.

    Returns:
        List of CuratorResult, one per curator, ordered by arrival (earliest first).

    Raises:
        ValueError: if alpha + beta >= 1 or parameters are out of range.
    """
    _validate_params(alpha, beta)

    stakes = signal.stakes
    n = len(stakes)
    if n == 0:
        return []

    # E_i = α · v_i · Σ_{j>i} (v_j / V_j⁻)
    # Precompute V_j⁻ for each j
    cumulative = [signal.cumulative_pool_before(j) for j in range(n)]

    results = []
    for i in range(n):
        v_i = stakes[i].amount
        earnings = 0.0
        for j in range(i + 1, n):
            V_j_minus = cumulative[j]
            if V_j_minus > 0:
                v_j = stakes[j].amount
                earnings += alpha * v_i * (v_j / V_j_minus)
        results.append(
            CuratorResult.from_earnings(
                curator_id=stakes[i].curator_id,
                arrival_index=i,
                stake=v_i,
                earnings=earnings,
            )
        )
    return results


def revenue_split(signal: Signal, alpha: float, beta: float) -> dict[str, float]:
    """Compute aggregate revenue for each agent class on a signal.

    Formal model §3.4 (Revenue by Role) + Proposition (Conservation).

    Args:
        signal: Signal with a populated stake list.
        alpha:  Curator pool share.
        beta:   Platform share.

    Returns:
        Dict with keys 'platform', 'creator', 'curators', 'total'.
        Invariant: platform + creator + curators == total == V_n.
    """
    _validate_params(alpha, beta)
    stakes = signal.stakes
    n = len(stakes)
    if n == 0:
        return {"platform": 0.0, "creator": 0.0, "curators": 0.0, "total": 0.0}

    gamma = 1.0 - alpha - beta
    V_n = signal.total_pool
    v_1 = stakes[0].amount

    platform = beta * V_n
    # First stake goes to creator minus platform cut; subsequent stakes split 3-way
    creator = v_1 * (1.0 - beta) + sum(s.amount for s in stakes[1:]) * gamma
    curators = alpha * (V_n - v_1)

    total = platform + creator + curators
    assert abs(total - V_n) < 1e-9, f"Conservation violated: {total} != {V_n}"

    return {
        "platform": platform,
        "creator": creator,
        "curators": curators,
        "total": V_n,
    }


def results_to_dataframe(results: list[CuratorResult]) -> pd.DataFrame:
    """Convert a list of CuratorResult to a tidy DataFrame.

    Useful for plotting and notebook exploration.
    """
    return pd.DataFrame([
        {
            "curator_id": r.curator_id,
            "arrival_index": r.arrival_index,
            "stake": r.stake,
            "earnings": r.earnings,
            "roi": r.roi,
            "profitable": r.profitable,
        }
        for r in results
    ])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_params(alpha: float, beta: float) -> None:
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0,1), got {alpha}")
    if not (0 < beta < 1):
        raise ValueError(f"beta must be in (0,1), got {beta}")
    if alpha + beta >= 1:
        raise ValueError(f"alpha + beta must be < 1, got {alpha + beta:.4f}")
