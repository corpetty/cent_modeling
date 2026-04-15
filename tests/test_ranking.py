"""
tests.test_ranking
~~~~~~~~~~~~~~~~~~
Unit tests for model.ranking — feed mechanics.

Tests verify:
    - rank_score respects weight function (decayed < accumulation)
    - feed_order sorts by rank descending
    - discovery_time returns correct crossing point
    - rank_trajectory shapes are correct
"""
import pytest
import math

from model.agents import Signal, Stake
from model.ranking import rank_score, feed_order, feed_position, rank_trajectory
from model.weight_functions import accumulation, exponential, step_window


def make_signal(amounts, times, quality=0.8, signal_id="sig"):
    stakes = [Stake(f"c{i}", float(a), float(t)) for i, (a, t) in enumerate(zip(amounts, times))]
    return Signal(signal_id=signal_id, quality=quality, publish_time=0.0, stakes=stakes)


class TestRankScore:
    def test_accumulation_equals_pool(self):
        """R(S,t) with accumulation weight == total pool (for t ≥ last stake time)."""
        sig = make_signal([1.0, 2.0, 3.0], [0.0, 1.0, 2.0])
        score = rank_score(sig, t=100.0, weight_fn=accumulation)
        assert abs(score - 6.0) < 1e-10

    def test_only_stakes_before_t_counted(self):
        """Stakes at t_i > t are excluded."""
        sig = make_signal([1.0, 2.0, 3.0], [0.0, 5.0, 10.0])
        score = rank_score(sig, t=4.9, weight_fn=accumulation)
        assert abs(score - 1.0) < 1e-10  # only first stake counts

    def test_exponential_decay_less_than_accumulation(self):
        """Exponential decay score < accumulation score when Δt > 0."""
        sig = make_signal([1.0, 1.0, 1.0], [0.0, 1.0, 2.0])
        t = 100.0  # large Δt → heavy decay
        acc = rank_score(sig, t, accumulation)
        exp_score = rank_score(sig, t, lambda dt: math.exp(-0.1 * dt))
        assert exp_score < acc

    def test_step_window_excludes_old_stakes(self):
        """Step window ignores stakes older than W."""
        # stake at t=0 (Δt=12 at eval) excluded; stake at t=5 (Δt=7) included
        sig = make_signal([10.0, 1.0], [0.0, 5.0])
        score = rank_score(sig, t=12.0, weight_fn=lambda dt: step_window(dt, window=10.0))
        assert abs(score - 1.0) < 1e-10  # only the stake at t=5 (amount=1.0) remains

    def test_zero_score_before_any_stakes(self):
        """R(S, t) = 0 when t < first stake time."""
        sig = make_signal([5.0, 3.0], [10.0, 20.0])
        assert rank_score(sig, t=5.0) == 0.0


class TestFeedOrder:
    def test_highest_stake_ranks_first(self):
        """With accumulation, signal with highest total stake is first."""
        low = make_signal([1.0], [0.0], quality=0.2, signal_id="low")
        high = make_signal([5.0], [0.0], quality=0.9, signal_id="high")
        ordered = feed_order([low, high], t=10.0, weight_fn=accumulation)
        assert ordered[0].signal_id == "high"
        assert ordered[1].signal_id == "low"

    def test_feed_order_with_decay_can_flip(self):
        """A newer smaller stake can outrank an older larger stake under decay."""
        old_big = make_signal([100.0], [0.0], signal_id="old_big")
        new_small = make_signal([1.0], [90.0], signal_id="new_small")
        # At t=100, old_big Δt=100 with exp decay almost zero; new_small Δt=10
        w = lambda dt: math.exp(-0.5 * dt)  # fast decay
        ordered = feed_order([old_big, new_small], t=100.0, weight_fn=w)
        assert ordered[0].signal_id == "new_small"

    def test_feed_position_single_signal(self):
        """A single signal is always at position 1."""
        sig = make_signal([1.0], [0.0])
        assert feed_position(sig, [sig], t=10.0) == 1

    def test_feed_position_consistent_with_order(self):
        """feed_position() agrees with feed_order() ranking."""
        # Use distinct amounts and distinct ids to avoid list.index() finding a duplicate
        sigs = [
            make_signal([float(v)], [0.0], signal_id=f"s{i}")
            for i, v in enumerate([3.0, 1.0, 4.0, 2.0, 5.0, 9.0])
        ]
        ordered = feed_order(sigs, t=10.0)
        for rank, sig in enumerate(ordered, start=1):
            assert feed_position(sig, sigs, t=10.0) == rank


class TestRankTrajectory:
    def test_trajectory_length(self):
        """rank_trajectory returns lists of length n_points + 1."""
        sig = make_signal([1.0, 2.0], [0.0, 1.0])
        times, scores = rank_trajectory(sig, n_points=50)
        assert len(times) == 51
        assert len(scores) == 51

    def test_trajectory_monotone_with_accumulation(self):
        """Under accumulation, rank score is non-decreasing over time."""
        sig = make_signal([1.0, 1.0, 1.0], [1.0, 3.0, 5.0])
        times, scores = rank_trajectory(sig, weight_fn=accumulation, n_points=100)
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1] + 1e-12

    def test_trajectory_decays_with_exponential(self):
        """Under exponential decay, score eventually decreases after last stake."""
        sig = make_signal([1.0, 1.0], [1.0, 2.0])
        w = lambda dt: math.exp(-0.5 * dt)
        times, scores = rank_trajectory(sig, weight_fn=w, t_end=50.0, n_points=200)
        # Peak should be before the end
        peak_idx = scores.index(max(scores))
        assert peak_idx < len(scores) - 1
        assert scores[-1] < scores[peak_idx]
