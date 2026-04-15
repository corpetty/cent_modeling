"""
simulations.run_scenario
~~~~~~~~~~~~~~~~~~~~~~~~
Entry point for running a single scenario and printing a summary.

Usage (from repo root):
    python -m simulations.run_scenario --scenario uniform --alpha 0.4 --beta 0.1 --n 60

Or from a notebook:
    from simulations.run_scenario import run
    results = run("mixed_normal", alpha=0.4, beta=0.1, n=80, seed=42)
"""
from __future__ import annotations

import argparse
from typing import Optional

from model.staking import compute_payouts, revenue_split, results_to_dataframe
from model.metrics import signal_accuracy, participation_rate, roi_by_arrival
from model.ranking import discovery_time
from model.weight_functions import exponential
from simulations.scenarios import (
    uniform_stakes,
    mixed_normal_stakes,
    power_law_stakes,
    coordinated_coalition,
)

SCENARIO_MAP = {
    "uniform":     uniform_stakes,
    "mixed_normal": mixed_normal_stakes,
    "power_law":   power_law_stakes,
    "coalition":   coordinated_coalition,
}


def run(
    scenario: str = "uniform",
    alpha: float = 0.4,
    beta: float = 0.1,
    n: int = 60,
    seed: Optional[int] = 42,
    verbose: bool = True,
) -> dict:
    """Run a named scenario and return a results dict.

    Args:
        scenario: One of 'uniform', 'mixed_normal', 'power_law', 'coalition'.
        alpha:    Curator pool share.
        beta:     Platform share.
        n:        Number of curators (passed to scenario factory as kwarg where supported).
        seed:     Random seed.
        verbose:  If True, print a formatted summary.

    Returns:
        Dict containing signal, results DataFrame, revenue split, and metrics.
    """
    if scenario not in SCENARIO_MAP:
        raise ValueError(f"Unknown scenario '{scenario}'. Choose from: {list(SCENARIO_MAP)}")

    factory = SCENARIO_MAP[scenario]
    # coalition scenario uses n_honest/n_coalition instead of n
    if scenario == "coalition":
        n_honest = max(1, int(n * 0.83))
        n_coalition = n - n_honest
        signal = factory(n_honest=n_honest, n_coalition=n_coalition, seed=seed)
    else:
        signal = factory(n=n, seed=seed)

    curator_results = compute_payouts(signal, alpha=alpha, beta=beta)
    df = results_to_dataframe(curator_results)
    rev = revenue_split(signal, alpha=alpha, beta=beta)
    p_rate = participation_rate(curator_results)

    out = {
        "signal":          signal,
        "results_df":      df,
        "revenue_split":   rev,
        "participation_rate": p_rate,
        "alpha":           alpha,
        "beta":            beta,
        "gamma":           1 - alpha - beta,
    }

    if verbose:
        _print_summary(scenario, out)

    return out


def _print_summary(scenario: str, out: dict) -> None:
    sig = out["signal"]
    rev = out["revenue_split"]
    df = out["results_df"]

    print(f"\n{'='*52}")
    print(f" Scenario: {scenario}  |  n={sig.n_curators}  |  q={sig.quality:.2f}")
    print(f" α={out['alpha']:.2f}  β={out['beta']:.2f}  γ={out['gamma']:.2f}")
    print(f"{'='*52}")
    print(f" Total staked:        {rev['total']:.4f}")
    print(f" Platform revenue:    {rev['platform']:.4f}  ({rev['platform']/rev['total']*100:.1f}%)")
    print(f" Creator revenue:     {rev['creator']:.4f}  ({rev['creator']/rev['total']*100:.1f}%)")
    print(f" Curator revenue:     {rev['curators']:.4f}  ({rev['curators']/rev['total']*100:.1f}%)")
    print(f" Participation rate:  {out['participation_rate']*100:.1f}% profitable curators")
    print(f" Median ROI:          {df['roi'].median()*100:.1f}%")
    print(f" First curator ROI:   {df['roi'].iloc[0]*100:.1f}%")
    print(f" Last curator ROI:    {df['roi'].iloc[-1]*100:.1f}%")
    print(f"{'='*52}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a curation model scenario.")
    parser.add_argument("--scenario", default="uniform", choices=list(SCENARIO_MAP))
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.scenario, alpha=args.alpha, beta=args.beta, n=args.n, seed=args.seed)
