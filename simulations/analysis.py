"""
Full analysis suite — curation market model.

Plots:
  01_roi_and_feed.png       — ROI curves & feed dynamics
  02_equilibrium.png        — Equilibrium / participation constraint
  03_weight_comparison.png  — Weight function comparison
  04_coalition.png          — Coalition robustness
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from model.agents import Signal, Stake
from model.staking import compute_payouts, revenue_split
from model.weight_functions import accumulation, exponential, power_law, step_window
from model.ranking import rank_score, feed_order, feed_position, discovery_time
from model.metrics import signal_accuracy, curator_roi, participation_rate, roi_by_arrival

OUT = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────

def make_signal(n=50, seed=42, low=1.0, high=10.0,
                quality=None, signal_id="s0"):
    rng = np.random.default_rng(seed)
    times = np.sort(rng.uniform(0, 100, n))
    amounts = rng.uniform(low, high, n)
    stakes = [Stake(f"c{i}", float(amounts[i]), float(times[i])) for i in range(n)]
    q = quality if quality is not None else float(rng.uniform(0.3, 0.95))
    return Signal(signal_id, q, float(times[0]), stakes)


def make_coalition_signal(n_honest=40, n_coalition=10, seed=42,
                          coalition_stake=20.0, signal_id="s_col"):
    rng = np.random.default_rng(seed)
    honest_times = np.sort(rng.uniform(0, 80, n_honest))
    honest_amounts = rng.uniform(1, 8, n_honest)
    ct = float(rng.uniform(5, 20))
    coalition_times = np.full(n_coalition, ct) + rng.uniform(0, 0.5, n_coalition)
    coalition_amounts = np.full(n_coalition, float(coalition_stake))
    all_times = np.concatenate([honest_times, coalition_times])
    all_amounts = np.concatenate([honest_amounts, coalition_amounts])
    idx = np.argsort(all_times)
    stakes = [Stake(f"c{i}", float(all_amounts[idx[i]]), float(all_times[idx[i]]))
              for i in range(len(idx))]
    return Signal(signal_id, 0.3, float(all_times[idx[0]]), stakes)


WEIGHT_FNS = {
    "accumulation": accumulation,
    "exp (λ=0.05)":  lambda dt: exponential(dt, lam=0.05),
    "power (θ=1.5)": lambda dt: power_law(dt, theta=1.5),
    "step (W=30)":   lambda dt: step_window(dt, window=30.0),
}
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


# ─────────────────────────────────────────────────────────────
# Plot 1 — ROI curves & feed dynamics
# ─────────────────────────────────────────────────────────────

def plot_roi_and_feed():
    alpha, beta = 0.4, 0.1
    sig = make_signal(n=60, seed=7)

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    # 1a ─ ROI by arrival order
    ax = fig.add_subplot(gs[0, 0])
    results = compute_payouts(sig, alpha=alpha, beta=beta)
    arrivals, rois = roi_by_arrival(results)
    bar_colors = ["#2ca02c" if r > 0 else "#d62728" for r in rois]
    ax.bar(arrivals, rois, color=bar_colors, alpha=0.75)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Curator arrival order")
    ax.set_ylabel("ROI")
    ax.set_title("ROI by arrival order\n(α=0.4, uniform stakes, n=60)")

    # 1b ─ Earnings vs stake amount, coloured by arrival
    ax2 = fig.add_subplot(gs[0, 1])
    earnings = [r.earnings for r in results]
    stakes_v = [r.stake for r in results]
    sc = ax2.scatter(stakes_v, earnings, c=range(len(results)),
                     cmap="plasma", alpha=0.75, s=40)
    ax2.set_xlabel("Stake amount")
    ax2.set_ylabel("Total earnings")
    ax2.set_title("Earnings vs stake\n(colour = arrival order, early=dark)")
    fig.colorbar(sc, ax=ax2, label="arrival order (0=first)")

    # 1c ─ Rank score trajectory (3 quality tiers)
    ax3 = fig.add_subplot(gs[1, 0])
    tiers = [(0.9, "High q=0.9", 8), (0.5, "Mid q=0.5", 9), (0.2, "Low q=0.2", 10)]
    t_range = np.linspace(0, 130, 200)
    for q, label, seed in tiers:
        s = make_signal(n=30, seed=seed, quality=q)
        scores = [rank_score(s, t=t) for t in t_range]
        ax3.plot(t_range, scores, label=label)
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Rank score (Σ stakes)")
    ax3.set_title("Rank score over time\n(accumulation weight, n=30 each)")
    ax3.legend(fontsize=8)

    # 1d ─ Feed position over time (1=top of feed)
    ax4 = fig.add_subplot(gs[1, 1])
    sigs = [make_signal(n=30, seed=s, quality=q, signal_id=f"sig{s}")
            for q, s in [(0.9, 8), (0.5, 9), (0.2, 10)]]
    t_range2 = np.linspace(10, 130, 60)
    labels2 = ["High q=0.9", "Mid q=0.5", "Low q=0.2"]
    for sig_i, label in zip(sigs, labels2):
        positions = [feed_position(sig_i, sigs, t=t) for t in t_range2]
        ax4.plot(t_range2, positions, label=label)
    ax4.invert_yaxis()
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Feed position (1 = top)")
    ax4.set_title("Feed position over time\n(staking pool IS the ranking function)")
    ax4.legend(fontsize=8)

    plt.suptitle("Plot 1 — ROI curves & feed dynamics", fontsize=13, y=1.01)
    path = os.path.join(OUT, "01_roi_and_feed.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────
# Plot 2 — Equilibrium / participation constraint
# ─────────────────────────────────────────────────────────────

def plot_equilibrium():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Plot 2 — Equilibrium & participation constraint", fontsize=13)

    beta = 0.1
    n = 40
    alpha_vals = np.linspace(0.05, 0.88, 50)

    # 2a ─ Participation rate vs α
    part_rates = []
    for alpha in alpha_vals:
        sig = make_signal(n=n, seed=42)
        results = compute_payouts(sig, alpha=alpha, beta=beta)
        part_rates.append(participation_rate(results))
    axes[0].plot(alpha_vals, part_rates, lw=2)
    axes[0].set_xlabel("α (curator pool share)")
    axes[0].set_ylabel("Fraction with ROI > 0")
    axes[0].set_title("Participation rate vs α\n(β=0.1, n=40, uniform stakes)")

    # 2b ─ ROI curve shape for several αs
    ax = axes[1]
    for alpha in [0.2, 0.4, 0.6, 0.8]:
        if alpha + beta >= 1.0:
            continue
        sig = make_signal(n=30, seed=42)
        results = compute_payouts(sig, alpha=alpha, beta=beta)
        _, rois = roi_by_arrival(results)
        ax.plot(range(len(rois)), rois, label=f"α={alpha}", alpha=0.85)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Arrival position")
    ax.set_ylabel("ROI")
    ax.set_title("ROI curve shape vs α\n(β=0.1, n=30)")
    ax.legend(fontsize=8)

    # 2c ─ Break-even position (last profitable curator) vs α
    breakeven = []
    for alpha in alpha_vals:
        sig = make_signal(n=n, seed=42)
        results = compute_payouts(sig, alpha=alpha, beta=beta)
        _, rois = roi_by_arrival(results)
        profitable = [i for i, r in enumerate(rois) if r > 0]
        breakeven.append(max(profitable) / n if profitable else 0.0)
    axes[2].plot(alpha_vals, breakeven, lw=2)
    axes[2].set_xlabel("α (curator pool share)")
    axes[2].set_ylabel("Break-even position (fraction)")
    axes[2].set_title("Last profitable arrival fraction vs α")

    plt.tight_layout()
    path = os.path.join(OUT, "02_equilibrium.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────
# Plot 3 — Weight function comparison
# ─────────────────────────────────────────────────────────────

def plot_weight_comparison():
    n_trials = 20
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Plot 3 — Weight function comparison", fontsize=13)
    axes = axes.flatten()

    # 3a ─ Rank score over time (same signal, all weight fns)
    ax = axes[0]
    sig = make_signal(n=40, seed=3)
    t_range = np.linspace(0, 140, 300)
    for (name, wfn), c in zip(WEIGHT_FNS.items(), COLORS):
        scores = [rank_score(sig, t=t, weight_fn=wfn) for t in t_range]
        ax.plot(t_range, scores, label=name, color=c)
    ax.set_xlabel("Time")
    ax.set_ylabel("Rank score")
    ax.set_title("Rank score over time\n(same signal, 40 curators)")
    ax.legend(fontsize=8)

    # 3b ─ Discovery time distribution across trials
    ax = axes[1]
    disc = {name: [] for name in WEIGHT_FNS}
    for seed in range(n_trials):
        sigs = [make_signal(n=20, seed=seed + s*100, quality=q, signal_id=f"s{s}")
                for s, q in enumerate([0.9, 0.6, 0.3])]
        for name, wfn in WEIGHT_FNS.items():
            dt = discovery_time(sigs[0], sigs, threshold_rank=1, weight_fn=wfn)
            disc[name].append(dt if dt is not None else float("nan"))
    bp = ax.boxplot([disc[n] for n in WEIGHT_FNS],
                    labels=list(WEIGHT_FNS.keys()),
                    patch_artist=True)
    for patch, c in zip(bp["boxes"], COLORS):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax.set_ylabel("Discovery time (reach pos 1)")
    ax.set_title("Discovery speed by weight fn\n(20 trials, 3-signal feed)")
    ax.tick_params(axis="x", labelsize=7)

    # 3c ─ Signal accuracy (Spearman ρ) — uses accumulation only (no weight_fn arg)
    ax = axes[2]
    acc_vals = []
    for seed in range(n_trials):
        sigs = [make_signal(n=20, seed=seed + s*100, quality=float(q), signal_id=f"s{s}")
                for s, q in enumerate(np.linspace(0.1, 0.9, 5))]
        val = signal_accuracy(sigs, t=80.0)
        if not np.isnan(val):
            acc_vals.append(val)
    ax.hist(acc_vals, bins=10, color="#1f77b4", alpha=0.75, edgecolor="white")
    ax.axvline(np.mean(acc_vals), color="red", ls="--", lw=1.5,
               label=f"mean={np.mean(acc_vals):.2f}")
    ax.set_xlabel("Signal accuracy (Spearman ρ)")
    ax.set_ylabel("Count")
    ax.set_title("Signal accuracy distribution\n(20 trials, 5-signal feed, t=80)")
    ax.legend(fontsize=8)

    # 3d ─ Mean discovery time vs weight function
    ax = axes[3]
    mean_disc = [np.nanmean(disc[n]) for n in WEIGHT_FNS]
    bars = ax.bar(list(WEIGHT_FNS.keys()), mean_disc, color=COLORS, alpha=0.75)
    ax.set_ylabel("Mean discovery time")
    ax.set_title("Mean discovery time by weight fn\n(lower = quality surfaces faster)")
    ax.tick_params(axis="x", labelsize=7)
    for bar, v in zip(bars, mean_disc):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.5,
                    f"{v:.1f}", ha="center", fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUT, "03_weight_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────
# Plot 4 — Coalition robustness
# ─────────────────────────────────────────────────────────────

def plot_coalition():
    alpha, beta = 0.4, 0.1
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Plot 4 — Coalition robustness", fontsize=13)
    axes = axes.flatten()

    # 4a ─ Feed position of honest signal vs coalition size
    ax = axes[0]
    honest_sig = make_signal(n=40, seed=5, quality=0.85, signal_id="honest")
    nc_range = list(range(0, 22))
    positions = []
    for nc in nc_range:
        if nc == 0:
            positions.append(feed_position(honest_sig, [honest_sig], t=50.0))
        else:
            col_sig = make_coalition_signal(n_honest=40, n_coalition=nc,
                                            seed=5, signal_id="col")
            positions.append(feed_position(honest_sig, [honest_sig, col_sig], t=50.0))
    ax.plot(nc_range, positions, "o-", color="#d62728")
    ax.axhline(1, color="#2ca02c", ls="--", alpha=0.6, label="top of feed")
    ax.invert_yaxis()
    ax.set_xlabel("Coalition curators (stake=20 each)")
    ax.set_ylabel("Feed position of honest signal")
    ax.set_title("Coalition displacement vs coalition size\n(honest n=40, q=0.85)")
    ax.legend(fontsize=8)

    # 4b ─ Coalition vs honest ROI as honest pool grows
    ax = axes[1]
    n_honest_vals = list(range(5, 50, 5))
    col_roi_means, hon_roi_means = [], []
    for n_h in n_honest_vals:
        c_rois, h_rois = [], []
        for seed in range(12):
            col_sig = make_coalition_signal(n_honest=n_h, n_coalition=8,
                                            seed=seed, coalition_stake=15.0)
            results = compute_payouts(col_sig, alpha=alpha, beta=beta)
            for r in results:
                roi = curator_roi(r.earnings, r.stake)
                if r.stake > 12.0:
                    c_rois.append(roi)
                else:
                    h_rois.append(roi)
        col_roi_means.append(np.mean(c_rois) if c_rois else 0)
        hon_roi_means.append(np.mean(h_rois) if h_rois else 0)
    ax.plot(n_honest_vals, col_roi_means, "o-", color="#d62728", label="Coalition (n=8, stake=15)")
    ax.plot(n_honest_vals, hon_roi_means, "s-", color="#1f77b4", label="Honest curators")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Number of honest curators")
    ax.set_ylabel("Mean ROI")
    ax.set_title("Coalition vs honest ROI\n(α=0.4, β=0.1)")
    ax.legend(fontsize=8)

    # 4c ─ Stake threshold for displacement (step function)
    ax = axes[2]
    col_stakes = np.linspace(1, 50, 50)
    displaced = []
    for cs in col_stakes:
        honest2 = make_signal(n=40, seed=5, quality=0.85, signal_id="honest")
        col_sig = make_coalition_signal(n_honest=40, n_coalition=10,
                                        seed=5, coalition_stake=cs, signal_id="col")
        pos = feed_position(honest2, [honest2, col_sig], t=50.0)
        displaced.append(int(pos > 1))
    ax.fill_between(col_stakes, displaced, alpha=0.3, color="#d62728")
    ax.plot(col_stakes, displaced, color="#d62728", lw=2)
    # mark threshold
    threshold_stake = next((col_stakes[i] for i, d in enumerate(displaced) if d == 1), None)
    if threshold_stake is not None:
        ax.axvline(threshold_stake, color="black", ls="--", lw=1.2,
                   label=f"threshold ≈ {threshold_stake:.1f}")
        ax.legend(fontsize=8)
    ax.set_xlabel("Coalition stake per curator (10 curators)")
    ax.set_ylabel("High-quality signal displaced (1=yes)")
    ax.set_title("Displacement threshold\n(10 coalition curators vs n=40 honest)")

    # 4d ─ Signal accuracy degradation vs coalition strength
    ax = axes[3]
    nc_range2 = list(range(0, 18))
    acc_means, acc_stds = [], []
    for nc in nc_range2:
        trial_accs = []
        for seed in range(15):
            honest_sigs = [make_signal(n=25, seed=seed + s*50, quality=float(q),
                                       signal_id=f"h{s}")
                           for s, q in enumerate(np.linspace(0.15, 0.85, 5))]
            if nc > 0:
                col_sigs = [make_coalition_signal(n_honest=25, n_coalition=nc,
                                                  seed=seed + s*50,
                                                  coalition_stake=15.0,
                                                  signal_id=f"c{s}")
                            for s in range(3)]
                all_sigs = honest_sigs + col_sigs
            else:
                all_sigs = honest_sigs
            val = signal_accuracy(all_sigs, t=60.0)
            if not np.isnan(val):
                trial_accs.append(val)
        acc_means.append(np.mean(trial_accs) if trial_accs else float("nan"))
        acc_stds.append(np.std(trial_accs) if trial_accs else 0.0)
    acc_means = np.array(acc_means)
    acc_stds = np.array(acc_stds)
    ax.plot(nc_range2, acc_means, "o-", color="#ff7f0e")
    ax.fill_between(nc_range2, acc_means - acc_stds, acc_means + acc_stds,
                    alpha=0.2, color="#ff7f0e")
    ax.set_xlabel("Coalition curators per injected signal")
    ax.set_ylabel("Signal accuracy (Spearman ρ)")
    ax.set_title("Signal accuracy vs coalition strength\n(15 trials ± 1σ)")

    plt.tight_layout()
    path = os.path.join(OUT, "04_coalition.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running full analysis suite...")
    paths = []
    paths.append(plot_roi_and_feed())
    paths.append(plot_equilibrium())
    paths.append(plot_weight_comparison())
    paths.append(plot_coalition())
    print(f"\nDone — {len(paths)} plots in {OUT}/")
    for p in paths:
        print(f"  {p}")
