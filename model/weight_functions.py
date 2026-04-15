"""
model.weight_functions
~~~~~~~~~~~~~~~~~~~~~~
Temporal weight functions w(Δt) for the feed ranking function R(S,t).

Formal model reference: §4.2 (Temporal Weight Functions)

Each function takes delta_t = t - t_i (time elapsed since stake i was placed)
and returns a non-negative weight. They are designed to be passed as callables
to ranking.rank_score().

Usage example:
    from model.weight_functions import exponential
    import functools

    w = functools.partial(exponential, lam=0.1)
    score = rank_score(signal, t=100.0, weight_fn=w)
"""
from __future__ import annotations

import math
from typing import Callable


# ---------------------------------------------------------------------------
# Weight function type alias
# ---------------------------------------------------------------------------
WeightFn = Callable[[float], float]


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------

def accumulation(delta_t: float) -> float:
    """w(Δt) = 1 — pure stake accumulation, no recency bias.

    The rank score equals the total staked pool V_n. Content never decays.
    Simplest baseline; equivalent to the original Cent model ranking.

    Args:
        delta_t: Time elapsed since this stake was placed (ignored).

    Returns:
        Always 1.0.
    """
    return 1.0


def exponential(delta_t: float, lam: float = 0.1) -> float:
    """w(Δt) = exp(-λ·Δt) — exponential decay.

    Recent stakes dominate. λ (lambda) controls the decay rate:
        half-life = ln(2) / λ ≈ 6.93 / λ (in the same time units as Δt).

    Args:
        delta_t: Time elapsed since this stake was placed. Must be ≥ 0.
        lam:     Decay rate λ > 0. Default 0.1.

    Returns:
        Weight ∈ (0, 1].
    """
    if lam <= 0:
        raise ValueError(f"lam must be > 0, got {lam}")
    return math.exp(-lam * delta_t)


def power_law(delta_t: float, theta: float = 1.5) -> float:
    """w(Δt) = (1 + Δt)^{-θ} — power-law decay.

    Heavy-tailed; similar to Hacker News gravity parameter. θ controls how
    aggressively old content is deranked. θ = 1.5 is a reasonable default
    (matches HN-style behaviour empirically).

    Args:
        delta_t: Time elapsed since this stake was placed. Must be ≥ 0.
        theta:   Decay exponent θ > 0. Default 1.5.

    Returns:
        Weight ∈ (0, 1].
    """
    if theta <= 0:
        raise ValueError(f"theta must be > 0, got {theta}")
    return (1.0 + delta_t) ** (-theta)


def step_window(delta_t: float, window: float = 24.0) -> float:
    """w(Δt) = 1[Δt ≤ W] — step window (hard cutoff).

    Only stakes placed within the last W time units contribute to rank.
    Produces the most aggressive freshness enforcement; content hard-resets
    to zero rank once all its stakes age past W.

    Args:
        delta_t: Time elapsed since this stake was placed. Must be ≥ 0.
        window:  Window width W > 0. Default 24.0 (e.g. 24 hours).

    Returns:
        1.0 if delta_t ≤ window, else 0.0.
    """
    if window <= 0:
        raise ValueError(f"window must be > 0, got {window}")
    return 1.0 if delta_t <= window else 0.0


# ---------------------------------------------------------------------------
# Registry — convenience for sweeps and plotting
# ---------------------------------------------------------------------------

WEIGHT_FUNCTIONS: dict[str, WeightFn] = {
    "accumulation": accumulation,
    "exponential":  exponential,
    "power_law":    power_law,
    "step_window":  step_window,
}
"""Named registry of all built-in weight functions with default parameters.

Use for parameter sweeps:
    for name, w in WEIGHT_FUNCTIONS.items():
        score = rank_score(signal, t, w)
"""
