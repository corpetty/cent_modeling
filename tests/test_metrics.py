"""
tests.test_metrics
~~~~~~~~~~~~~~~~~~
Unit tests for model.metrics — signal accuracy, ROI, participation rate,
and consumer surplus.
"""
import math
import pytest

from model.agents import Signal, Stake
from model.staking import compute_payouts
from model.metrics import (
    signal_accuracy,
    curator_roi,
    participation_rate,
    roi_by_arrival,
    consumer_surplus,
)
from model.weight_functions import accumulation


def make_signal(amounts, times, quality, signal_id="sig"):
    stakes = [Stake(f"c{i}", float(a), float(t)) for i, (a, t) in enumerate(zip(amounts, times))]
    return Signal(signal_id=signal_id, quality=quality, publish_time=0.0, stakes=stakes)


class TestSignalAccuracy:
    def test_perfect_correlation(self):
        """ρ = 1 when stake rank exactly matches quality rank."""
        sigs = [
            make_signal([1.0], [0.0], quality=0.2, signal_id="s1"),
            make_signal([2.0], [0.0], quality=0.5, signal_id="s2"),
            make_signal([3.0], [0.0], quality=0.9, signal_id="s3"),
        ]
        rho = signal_accuracy(sigs, t=10.0)
        assert abs(rho - 1.0) < 1e-9

    def test_zero_correlation(self):
        """ρ ≈ 0 when stakes are uncorrelated with quality (rough test)."""
        # Stake rank: [3,1,2], quality rank: [1,3,2] → negative correlation
        sigs = [
            make_signal([3.0], [0.0], quality=0.1, signal_id="s1"),
            make_signal([1.0], [0.0], quality=0.9, signal_id="s2"),
            make_signal([2.0], [0.0], quality=0.5, signal_id="s3"),
        ]
        rho = signal_accuracy(sigs, t=10.0)
        assert rho < 0  # anti-correlated

    def test_fewer_than_two_returns_nan(self):
        """With fewer than 2 eligible signals, returns NaN."""
        import math
        sigs = [make_signal([1.0], [0.0], quality=0.5, signal_id="s1")]
        rho = signal_accuracy(sigs, t=10.0)
        assert math.isnan(rho)

    def test_excludes_future_signals(self):
        """Signals published after t are excluded."""
        past = make_signal([3.0], [0.0], quality=0.9, signal_id="past")
        past.publish_time = 0.0
        future = make_signal([1.0], [100.0], quality=0.1, signal_id="future")
        future.publish_time = 50.0  # published after t=10
        rho = signal_accuracy([past, future], t=10.0)
        assert math.isnan(rho)  # only one eligible signal


class TestCuratorROI:
    def test_break_even(self):
        assert curator_roi(earnings=1.0, stake=1.0) == 0.0

    def test_positive_roi(self):
        assert abs(curator_roi(2.0, 1.0) - 1.0) < 1e-10

    def test_total_loss(self):
        assert abs(curator_roi(0.0, 1.0) - (-1.0)) < 1e-10

    def test_zero_stake_raises(self):
        with pytest.raises(ValueError):
            curator_roi(1.0, 0.0)


class TestParticipationRate:
    def test_all_profitable(self):
        sig = make_signal([1.0] * 20, list(range(20)), quality=0.8)
        results = compute_payouts(sig, alpha=0.5, beta=0.1)
        # First curator is always profitable with enough subsequent stakers
        rate = participation_rate(results)
        assert 0.0 < rate <= 1.0

    def test_empty_results(self):
        assert participation_rate([]) == 0.0

    def test_last_curator_not_profitable(self):
        """Last curator never earns anything → not profitable."""
        sig = make_signal([1.0] * 5, list(range(5)), quality=0.8)
        results = compute_payouts(sig, alpha=0.4, beta=0.1)
        assert not results[-1].profitable

    def test_roi_by_arrival_length(self):
        sig = make_signal([1.0] * 10, list(range(10)), quality=0.8)
        results = compute_payouts(sig, alpha=0.4, beta=0.1)
        fracs, rois = roi_by_arrival(results)
        assert len(fracs) == 10
        assert len(rois) == 10
        assert fracs[0] == 0.0
        assert fracs[-1] == 1.0


class TestConsumerSurplus:
    def test_higher_quality_signals_increase_surplus(self):
        """Adding a high-quality signal raises CS."""
        low = make_signal([5.0], [0.0], quality=0.1, signal_id="low")
        high = make_signal([5.0], [0.0], quality=0.9, signal_id="high")
        cs_one = consumer_surplus([low], t=10.0, weight_fn=accumulation, n_consumers=1)
        cs_two = consumer_surplus([low, high], t=10.0, weight_fn=accumulation, n_consumers=1)
        assert cs_two > cs_one

    def test_empty_feed_zero_surplus(self):
        cs = consumer_surplus([], t=10.0, weight_fn=accumulation)
        assert cs == 0.0

    def test_scales_with_consumers(self):
        sigs = [make_signal([1.0], [0.0], quality=0.9, signal_id="s")]
        cs1 = consumer_surplus(sigs, t=10.0, n_consumers=1)
        cs100 = consumer_surplus(sigs, t=10.0, n_consumers=100)
        assert abs(cs100 - 100 * cs1) < 1e-9
