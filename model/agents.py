"""
model.agents
~~~~~~~~~~~~
Pure data structures — no computation.

Formal model reference: §2 (Primitives), §3.1 (Staking Game Setup)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Stake:
    """A single curator stake on a signal.

    Corresponds to the ordered pair (v_i, t_i) in the formal model.

    Attributes:
        curator_id: Unique identifier for the curator.
        amount:     Stake size v_i > 0. May be currency, tokens, reputation, etc.
        time:       Arrival time t_i ≥ 0 (seconds since signal publication, or
                    any consistent ordinal time unit).
    """
    curator_id: str
    amount: float       # v_i
    time: float         # t_i

    def __post_init__(self):
        if self.amount <= 0:
            raise ValueError(f"Stake amount must be > 0, got {self.amount}")
        if self.time < 0:
            raise ValueError(f"Stake time must be >= 0, got {self.time}")


@dataclass
class Signal:
    """A content item (post, claim, proposal, etc.) published to the platform.

    Corresponds to S ∈ 𝒮 in the formal model.

    Attributes:
        signal_id:    Unique identifier.
        quality:      True latent quality q(S) ∈ [0, 1]. Not observable by agents
                      at publication time; used only for evaluation/validation.
        publish_time: τ(S) ≥ 0. Time at which the signal was published.
        stakes:       Ordered list of stakes (v_i, t_i), sorted by arrival time.
                      Pass unsorted — they will be sorted on assignment.
    """
    signal_id: str
    quality: float              # q(S) ∈ [0,1], latent ground truth
    publish_time: float = 0.0   # τ(S)
    stakes: list[Stake] = field(default_factory=list)

    def __post_init__(self):
        if not (0.0 <= self.quality <= 1.0):
            raise ValueError(f"Signal quality must be in [0,1], got {self.quality}")
        # Enforce time ordering
        self.stakes = sorted(self.stakes, key=lambda s: s.time)

    def add_stake(self, stake: Stake) -> None:
        """Append a stake and keep the list sorted by arrival time."""
        self.stakes.append(stake)
        self.stakes.sort(key=lambda s: s.time)

    @property
    def total_pool(self) -> float:
        """V_n: total stake pool across all curators."""
        return sum(s.amount for s in self.stakes)

    @property
    def n_curators(self) -> int:
        """Number of curators who have staked."""
        return len(self.stakes)

    def cumulative_pool_before(self, index: int) -> float:
        """V_i^-: cumulative pool just before curator at position `index` stakes.

        Args:
            index: 0-based position in the sorted stakes list.

        Returns:
            Sum of all stakes with position < index.
        """
        return sum(s.amount for s in self.stakes[:index])


@dataclass
class CuratorResult:
    """Computed payout result for a single curator on a single signal.

    Produced by :func:`model.staking.compute_payouts`.

    Attributes:
        curator_id:    Identifier, matching the originating Stake.
        arrival_index: 0-based position in the staking sequence (0 = earliest).
        stake:         v_i — amount staked.
        earnings:      E_i — total earnings received from all subsequent stakers.
        roi:           (E_i - v_i) / v_i — return on investment.
        profitable:    True if E_i > v_i (curator recouped their stake).
    """
    curator_id: str
    arrival_index: int
    stake: float            # v_i
    earnings: float         # E_i
    roi: float              # (E_i - v_i) / v_i
    profitable: bool        # E_i > v_i

    @classmethod
    def from_earnings(cls, curator_id: str, arrival_index: int,
                      stake: float, earnings: float) -> "CuratorResult":
        roi = (earnings - stake) / stake if stake > 0 else 0.0
        return cls(
            curator_id=curator_id,
            arrival_index=arrival_index,
            stake=stake,
            earnings=earnings,
            roi=roi,
            profitable=earnings > stake,
        )
