"""
cent_modeling.model
~~~~~~~~~~~~~~~~~~~
Formal implementation of the Community Curation Pipeline model.

Modules:
    agents          — dataclasses: Signal, Stake, CuratorResult
    staking         — core payout math (§3 of formal model)
    weight_functions — temporal weight functions w(Δt) (§4.2)
    ranking         — rank score, feed ordering, discovery time (§4)
    metrics         — signal accuracy, curator ROI, consumer surplus (§6)
"""
from .agents import Signal, Stake, CuratorResult
from .staking import compute_payouts, revenue_split
from .weight_functions import accumulation, exponential, power_law, step_window
from .ranking import rank_score, feed_order, discovery_time
from .metrics import signal_accuracy, curator_roi, consumer_surplus

__all__ = [
    "Signal", "Stake", "CuratorResult",
    "compute_payouts", "revenue_split",
    "accumulation", "exponential", "power_law", "step_window",
    "rank_score", "feed_order", "discovery_time",
    "signal_accuracy", "curator_roi", "consumer_surplus",
]
