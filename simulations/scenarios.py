"""
simulations.scenarios
~~~~~~~~~~~~~~~~~~~~~
Named scenario constructors that produce Signal objects for simulation runs.

Each factory function returns a Signal with a populated stakes list, ready
to pass to model.staking.compute_payouts() or model.ranking.rank_score().

Available scenarios:
    uniform_stakes          — v_i ~ Uniform(lo, hi), t_i ~ Exponential(rate)
    mixed_normal_stakes     — multi-modal v_i distribution (original Cent model)
    power_law_stakes        — v_i ~ Pareto (realistic wealth distribution)
    coordinated_coalition   — honest curators + a colluding bloc (attack scenario)
"""
from __future__ import annotations

import numpy as np
from typing import Optional

from model.agents import Signal, Stake


def uniform_stakes(
    n: int = 60,
    lo: float = 0.01,
    hi: float = 1.0,
    arrival_rate: float = 1.0,
    quality: float = 0.8,
    signal_id: str = "sig_uniform",
    seed: Optional[int] = None,
) -> Signal:
    """Curators stake uniformly random amounts, arriving via Poisson process.

    Args:
        n:            Number of curators.
        lo:           Minimum stake amount.
        hi:           Maximum stake amount.
        arrival_rate: λ for exponential inter-arrival times (curators/time-unit).
        quality:      True signal quality q(S).
        signal_id:    Identifier for the returned Signal.
        seed:         Random seed for reproducibility.

    Returns:
        Signal with n stakes, sorted by arrival time.
    """
    rng = np.random.default_rng(seed)
    amounts = rng.uniform(lo, hi, n)
    inter_arrivals = rng.exponential(1.0 / arrival_rate, n)
    times = np.cumsum(inter_arrivals)

    stakes = [
        Stake(curator_id=f"c{i:03d}", amount=float(amounts[i]), time=float(times[i]))
        for i in range(n)
    ]
    return Signal(signal_id=signal_id, quality=quality, publish_time=0.0, stakes=stakes)


def mixed_normal_stakes(
    n: int = 80,
    centers: tuple[float, ...] = (0.2, 0.5, 0.8),
    sigmas: tuple[float, ...] = (0.08, 0.08, 0.35),
    weights: tuple[float, ...] = (0.45, 0.45, 0.10),
    arrival_rate: float = 1.0,
    quality: float = 0.75,
    signal_id: str = "sig_mixed_normal",
    seed: Optional[int] = None,
) -> Signal:
    """Stakes drawn from a mixture of normals (mirrors original Cent model).

    The last center is the outlier distribution — higher sigma, lower weight.
    Negative stakes are clipped to a small positive value.

    Args:
        n:            Total number of curators.
        centers:      Mean of each normal component.
        sigmas:       Std dev of each normal component.
        weights:      Mixture weights (must sum to 1).
        arrival_rate: λ for exponential inter-arrival times.
        quality:      True signal quality q(S).
        signal_id:    Identifier for the returned Signal.
        seed:         Random seed.

    Returns:
        Signal with n stakes drawn from the mixture, shuffled.
    """
    rng = np.random.default_rng(seed)
    weights_arr = np.array(weights)
    weights_arr /= weights_arr.sum()  # normalise
    sizes = (weights_arr * n).astype(int)
    sizes[-1] = n - sizes[:-1].sum()  # absorb rounding remainder

    amounts = np.concatenate([
        rng.normal(mu, sigma, size)
        for mu, sigma, size in zip(centers, sigmas, sizes)
    ])
    amounts = np.clip(amounts, 1e-4, None)
    rng.shuffle(amounts)

    inter_arrivals = rng.exponential(1.0 / arrival_rate, n)
    times = np.cumsum(inter_arrivals)

    stakes = [
        Stake(curator_id=f"c{i:03d}", amount=float(amounts[i]), time=float(times[i]))
        for i in range(n)
    ]
    return Signal(signal_id=signal_id, quality=quality, publish_time=0.0, stakes=stakes)


def power_law_stakes(
    n: int = 60,
    pareto_alpha: float = 2.0,
    x_min: float = 0.01,
    arrival_rate: float = 1.0,
    quality: float = 0.7,
    signal_id: str = "sig_power_law",
    seed: Optional[int] = None,
) -> Signal:
    """Stakes drawn from a Pareto distribution — heavy-tailed wealth model.

    A few curators stake large amounts; the majority stake small amounts.
    More realistic for token/crypto platforms.

    Args:
        n:             Number of curators.
        pareto_alpha:  Shape parameter α > 1. Lower = heavier tail.
        x_min:         Minimum stake (scale parameter).
        arrival_rate:  λ for exponential inter-arrival times.
        quality:       True signal quality.
        signal_id:     Identifier.
        seed:          Random seed.

    Returns:
        Signal with Pareto-distributed stakes.
    """
    rng = np.random.default_rng(seed)
    # Pareto: X = x_min / U^(1/α) where U ~ Uniform(0,1)
    u = rng.uniform(0, 1, n)
    amounts = x_min / (u ** (1.0 / pareto_alpha))

    inter_arrivals = rng.exponential(1.0 / arrival_rate, n)
    times = np.cumsum(inter_arrivals)

    stakes = [
        Stake(curator_id=f"c{i:03d}", amount=float(amounts[i]), time=float(times[i]))
        for i in range(n)
    ]
    return Signal(signal_id=signal_id, quality=quality, publish_time=0.0, stakes=stakes)


def coordinated_coalition(
    n_honest: int = 50,
    n_coalition: int = 10,
    honest_lo: float = 0.1,
    honest_hi: float = 1.0,
    coalition_amount: float = 0.5,
    arrival_rate: float = 1.0,
    quality: float = 0.2,   # low-quality signal being pumped
    signal_id: str = "sig_coalition_attack",
    seed: Optional[int] = None,
) -> Signal:
    """Simulate a coordinated coalition staking a low-quality signal.

    n_honest curators stake honestly with random amounts.
    n_coalition curators stake a fixed coordinated amount.
    All arrivals are Poisson-interleaved.

    Use this to test mechanism robustness (§7 open question 4).

    Args:
        n_honest:         Number of honest independent curators.
        n_coalition:      Size of the colluding bloc.
        honest_lo/hi:     Honest stake range.
        coalition_amount: Fixed stake per coalition member.
        arrival_rate:     λ for inter-arrival times (shared).
        quality:          True quality of the signal (typically low).
        signal_id:        Identifier.
        seed:             Random seed.

    Returns:
        Signal with honest + coalition stakes interleaved.
    """
    rng = np.random.default_rng(seed)
    n = n_honest + n_coalition

    honest_amounts = rng.uniform(honest_lo, honest_hi, n_honest)
    coalition_amounts = np.full(n_coalition, coalition_amount)
    all_amounts = np.concatenate([honest_amounts, coalition_amounts])

    # Labels for traceability
    labels = (
        [f"honest_{i:03d}" for i in range(n_honest)] +
        [f"coalition_{i:03d}" for i in range(n_coalition)]
    )

    # Shuffle arrival order
    order = rng.permutation(n)
    all_amounts = all_amounts[order]
    labels = [labels[i] for i in order]

    inter_arrivals = rng.exponential(1.0 / arrival_rate, n)
    times = np.cumsum(inter_arrivals)

    stakes = [
        Stake(curator_id=labels[i], amount=float(all_amounts[i]), time=float(times[i]))
        for i in range(n)
    ]
    return Signal(signal_id=signal_id, quality=quality, publish_time=0.0, stakes=stakes)
