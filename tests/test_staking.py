"""
tests.test_staking
~~~~~~~~~~~~~~~~~~
Unit tests for model.staking — the core payout math.

These tests verify formal model invariants:
    - Conservation: Π_P + Π_K + Π_C = V_n
    - First-mover advantage: E_0 > E_{n-1} (all else equal)
    - ROI monotonicity: earlier curators generally earn higher ROI
    - Payout formula: π(i,j) = v_j · α · v_i / V_j⁻
"""
import pytest
import numpy as np

from model.agents import Signal, Stake
from model.staking import compute_payouts, revenue_split, payout, _validate_params


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_signal(amounts, times=None, quality=0.8, signal_id="test_sig"):
    """Helper: build a Signal from lists of stake amounts."""
    if times is None:
        times = list(range(len(amounts)))
    stakes = [
        Stake(curator_id=f"c{i}", amount=float(a), time=float(t))
        for i, (a, t) in enumerate(zip(amounts, times))
    ]
    return Signal(signal_id=signal_id, quality=quality, publish_time=0.0, stakes=stakes)


# ---------------------------------------------------------------------------
# Revenue conservation
# ---------------------------------------------------------------------------

class TestConservation:
    def test_conservation_uniform(self):
        """Π_P + Π_K + Π_C == V_n for uniform stakes."""
        sig = make_signal([1.0] * 10)
        rev = revenue_split(sig, alpha=0.4, beta=0.1)
        assert abs(rev["platform"] + rev["creator"] + rev["curators"] - rev["total"]) < 1e-9

    def test_conservation_varied(self):
        """Conservation holds for varied stake sizes."""
        sig = make_signal([0.1, 0.5, 2.0, 0.3, 1.1])
        rev = revenue_split(sig, alpha=0.3, beta=0.2)
        assert abs(rev["platform"] + rev["creator"] + rev["curators"] - rev["total"]) < 1e-9

    def test_conservation_extreme_alpha(self):
        """Conservation holds near the boundary alpha + beta → 1."""
        sig = make_signal([1.0, 1.0, 1.0])
        rev = revenue_split(sig, alpha=0.8, beta=0.15)
        assert abs(rev["platform"] + rev["creator"] + rev["curators"] - rev["total"]) < 1e-9

    def test_total_equals_pool(self):
        """Total revenue == total staked (V_n)."""
        amounts = [0.5, 1.0, 0.25, 2.0]
        sig = make_signal(amounts)
        rev = revenue_split(sig, alpha=0.4, beta=0.1)
        assert abs(rev["total"] - sum(amounts)) < 1e-9


# ---------------------------------------------------------------------------
# First-mover advantage
# ---------------------------------------------------------------------------

class TestFirstMoverAdvantage:
    def test_first_curator_earns_more_than_last(self):
        """E_0 > E_{n-1} for equal uniform stakes."""
        sig = make_signal([1.0] * 20)
        results = compute_payouts(sig, alpha=0.4, beta=0.1)
        assert results[0].earnings > results[-1].earnings

    def test_earnings_monotonically_decreasing_uniform(self):
        """With equal stakes, earnings strictly decrease by arrival order."""
        sig = make_signal([1.0] * 10)
        results = compute_payouts(sig, alpha=0.4, beta=0.1)
        earnings = [r.earnings for r in results]
        for i in range(len(earnings) - 1):
            assert earnings[i] >= earnings[i + 1], (
                f"Earnings not monotone at index {i}: {earnings[i]:.4f} < {earnings[i+1]:.4f}"
            )

    def test_last_curator_earns_zero(self):
        """The last curator (no subsequent stakers) earns exactly 0."""
        sig = make_signal([1.0] * 5)
        results = compute_payouts(sig, alpha=0.4, beta=0.1)
        assert results[-1].earnings == 0.0

    def test_first_curator_roi_positive(self):
        """With enough subsequent stakers, the first curator is always profitable."""
        sig = make_signal([1.0] * 15)
        results = compute_payouts(sig, alpha=0.4, beta=0.1)
        assert results[0].roi > 0


# ---------------------------------------------------------------------------
# Payout formula
# ---------------------------------------------------------------------------

class TestPayoutFormula:
    def test_payout_manual(self):
        """π(0,1) = v_1 · α · v_0 / V_1⁻ when there are exactly 2 curators."""
        alpha = 0.4
        sig = make_signal([2.0, 3.0])
        # V_1⁻ = v_0 = 2.0; π(0,1) = 3.0 * 0.4 * (2.0 / 2.0) = 1.2
        expected = 3.0 * alpha * (2.0 / 2.0)
        actual = payout(0, 1, sig, alpha)
        assert abs(actual - expected) < 1e-10

    def test_payout_proportional_to_incoming_stake(self):
        """π(i,j) is proportional to v_j (the incoming stake) for fixed pool state."""
        alpha = 0.4
        # sig1 has v_2 = 3.0, sig2 has v_2 = 6.0; everything before j=2 is identical
        sig1 = make_signal([1.0, 2.0, 3.0])
        sig2 = make_signal([1.0, 2.0, 6.0])
        p1 = payout(0, 2, sig1, alpha)
        p2 = payout(0, 2, sig2, alpha)
        # V_2⁻ = 3.0 for both; only v_j differs → doubling v_j doubles payout
        assert abs(p2 - 2 * p1) < 1e-10

    def test_payout_i_ge_j_raises(self):
        """payout() raises ValueError when i >= j."""
        sig = make_signal([1.0, 1.0, 1.0])
        with pytest.raises(ValueError):
            payout(1, 1, sig, 0.4)
        with pytest.raises(ValueError):
            payout(2, 1, sig, 0.4)


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_alpha_out_of_range(self):
        with pytest.raises(ValueError):
            _validate_params(0.0, 0.1)
        with pytest.raises(ValueError):
            _validate_params(1.0, 0.1)

    def test_beta_out_of_range(self):
        with pytest.raises(ValueError):
            _validate_params(0.4, 0.0)

    def test_alpha_plus_beta_ge_1(self):
        with pytest.raises(ValueError):
            _validate_params(0.5, 0.5)
        with pytest.raises(ValueError):
            _validate_params(0.9, 0.2)

    def test_valid_params_no_raise(self):
        _validate_params(0.4, 0.1)   # should not raise
        _validate_params(0.01, 0.01)
        _validate_params(0.8, 0.15)

    def test_stake_must_be_positive(self):
        with pytest.raises(ValueError):
            Stake("c0", amount=0.0, time=1.0)
        with pytest.raises(ValueError):
            Stake("c0", amount=-1.0, time=1.0)

    def test_signal_quality_range(self):
        with pytest.raises(ValueError):
            Signal("s0", quality=-0.1, publish_time=0.0)
        with pytest.raises(ValueError):
            Signal("s0", quality=1.1, publish_time=0.0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_curator(self):
        """With one curator, earnings = 0, revenue goes to creator + platform."""
        sig = make_signal([5.0])
        results = compute_payouts(sig, alpha=0.4, beta=0.1)
        assert len(results) == 1
        assert results[0].earnings == 0.0

        rev = revenue_split(sig, alpha=0.4, beta=0.1)
        assert abs(rev["curators"]) < 1e-9
        assert abs(rev["total"] - 5.0) < 1e-9

    def test_empty_signal(self):
        """Empty signal returns empty results and zero revenue."""
        sig = Signal("empty", quality=0.5, publish_time=0.0, stakes=[])
        results = compute_payouts(sig, alpha=0.4, beta=0.1)
        assert results == []
        rev = revenue_split(sig, alpha=0.4, beta=0.1)
        assert rev["total"] == 0.0

    def test_two_curators_earnings_sum(self):
        """With 2 curators, total curator earnings == α · v_1 (only curator 0 earns)."""
        alpha, beta = 0.4, 0.1
        sig = make_signal([1.0, 2.0])
        results = compute_payouts(sig, alpha=alpha, beta=beta)
        # E_0 = α · v_0 · (v_1 / V_1⁻) = 0.4 · 1.0 · (2.0 / 1.0) = 0.8
        assert abs(results[0].earnings - 0.8) < 1e-10
        assert results[1].earnings == 0.0
