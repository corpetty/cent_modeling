"""
model.ranking
~~~~~~~~~~~~~
Feed mechanics: rank scores, feed ordering, and consumer discovery time.

Formal model reference: §4 (Feed Mechanics)

Key equations:
    R(S,t) = Σ_{i: t_i ≤ t} w(t - t_i) · v_i       [Rank score, Def 4.1]
    Δτ*(S,k) = inf{t : r(S,t) ≤ k} - τ(S)           [Discovery time, Def 6.2]

The staking–feed duality (§4.3): the same stake pool that determines
curator earnings also determines feed position. Incentive design and
feed quality are coupled.
"""
from __future__ import annotations

from typing import Callable, Optional

from .agents import Signal
from .weight_functions import WeightFn, accumulation


# ---------------------------------------------------------------------------
# Core ranking
# ---------------------------------------------------------------------------

def rank_score(signal: Signal, t: float, weight_fn: WeightFn = accumulation) -> float:
    """R(S,t): weighted sum of stakes placed on signal S up to time t.

    Args:
        signal:    Signal containing ordered stakes.
        t:         Evaluation time. Only stakes with t_i ≤ t are counted.
        weight_fn: w(Δt) callable. Defaults to accumulation (w=1).

    Returns:
        Non-negative rank score. Higher = better feed position.
    """
    score = 0.0
    for stake in signal.stakes:
        if stake.time <= t:
            delta_t = t - stake.time
            score += weight_fn(delta_t) * stake.amount
    return score


def feed_order(signals: list[Signal], t: float,
               weight_fn: WeightFn = accumulation) -> list[Signal]:
    """Order signals by rank score at time t, highest first.

    This is the consumer feed at time t.

    Args:
        signals:   All signals to rank.
        t:         Evaluation time.
        weight_fn: Temporal weight function.

    Returns:
        Signals sorted by R(S,t) descending (feed position 1 = index 0).
    """
    return sorted(signals, key=lambda s: rank_score(s, t, weight_fn), reverse=True)


def feed_position(signal: Signal, signals: list[Signal], t: float,
                  weight_fn: WeightFn = accumulation) -> int:
    """Return the 1-based feed position of `signal` among `signals` at time t.

    Position 1 = highest rank.

    Args:
        signal:   The signal whose position we want.
        signals:  Full set of signals to rank against.
        t:        Evaluation time.
        weight_fn: Temporal weight function.

    Returns:
        Integer position ≥ 1.
    """
    ordered = feed_order(signals, t, weight_fn)
    return ordered.index(signal) + 1


# ---------------------------------------------------------------------------
# Discovery time
# ---------------------------------------------------------------------------

def discovery_time(
    signal: Signal,
    all_signals: list[Signal],
    threshold_rank: int = 10,
    weight_fn: WeightFn = accumulation,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    resolution: int = 500,
) -> Optional[float]:
    """Δτ*(S, k): time at which signal first enters the top-k feed positions.

    Scans time from t_start to t_end at `resolution` steps. Returns the
    earliest time the signal's feed position is ≤ threshold_rank.

    Args:
        signal:          Target signal.
        all_signals:     Full set of signals on the platform.
        threshold_rank:  k — discovery occurs when position ≤ k. Default 10.
        weight_fn:       Temporal weight function.
        t_start:         Scan start time. Defaults to signal.publish_time.
        t_end:           Scan end time. Defaults to latest stake time + buffer.
        resolution:      Number of time steps in the scan. Default 500.

    Returns:
        Discovery time (absolute), or None if the signal never reaches
        the threshold within [t_start, t_end].
    """
    if t_start is None:
        t_start = signal.publish_time
    if t_end is None:
        all_stake_times = [
            s.time for sig in all_signals for s in sig.stakes
        ]
        t_end = max(all_stake_times) * 1.5 if all_stake_times else t_start + 100.0

    step = (t_end - t_start) / resolution
    t = t_start
    while t <= t_end:
        pos = feed_position(signal, all_signals, t, weight_fn)
        if pos <= threshold_rank:
            return t
        t += step
    return None


def rank_trajectory(
    signal: Signal,
    weight_fn: WeightFn = accumulation,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    n_points: int = 300,
) -> tuple[list[float], list[float]]:
    """Compute R(S,t) over a time range — useful for plotting.

    Args:
        signal:    Signal to trace.
        weight_fn: Temporal weight function.
        t_start:   Start time. Defaults to signal.publish_time.
        t_end:     End time. Defaults to last stake time * 3.
        n_points:  Number of evaluation points.

    Returns:
        (times, scores): two parallel lists of floats.
    """
    if t_start is None:
        t_start = signal.publish_time
    if t_end is None and signal.stakes:
        t_end = signal.stakes[-1].time * 3.0
    elif t_end is None:
        t_end = t_start + 100.0

    step = (t_end - t_start) / n_points
    times = [t_start + i * step for i in range(n_points + 1)]
    scores = [rank_score(signal, t, weight_fn) for t in times]
    return times, scores
