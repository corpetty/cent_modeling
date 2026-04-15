"""
Curation Market Model — Visualizations
Formal model v1 (4-sided: Creator, Curator, Consumer, Platform)

Plots:
  1. Curator ROI bubble chart (echoes original Yours Network plot)
  2. Cumulative earnings per curator vs. arrival order
  3. Ranking dynamics under three weight functions
  4. Phase diagram: early vs. late staking under varying alpha
  5. Revenue split across agents (stacked bar, parametric sweep)
  6. Consumer discovery curve (rank threshold crossing)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import expon

plt.rcParams.update({
    'figure.facecolor': '#0f0f0f',
    'axes.facecolor': '#1a1a1a',
    'axes.edgecolor': '#444',
    'axes.labelcolor': '#ccc',
    'xtick.color': '#999',
    'ytick.color': '#999',
    'text.color': '#ccc',
    'grid.color': '#2a2a2a',
    'grid.linestyle': '--',
    'font.family': 'monospace',
    'axes.titlecolor': '#fff',
})

RNG = np.random.default_rng(42)

# ─── Model core ──────────────────────────────────────────────────────────────

def simulate(n=80, alpha=0.5, beta=0.1, stake_dist='uniform',
             max_stake=5.0, arrival_dist='uniform', T=100):
    """
    Simulate n curators staking on a single content item.
    Returns a DataFrame with per-curator results.
    """
    gamma = 1 - alpha - beta

    # Stake amounts
    if stake_dist == 'uniform':
        stakes = RNG.uniform(0.1, max_stake, n)
    elif stake_dist == 'mixed_normal':
        s1 = RNG.normal(1.0, 0.2, int(0.6*n))
        s2 = RNG.normal(3.0, 0.4, int(0.3*n))
        s3 = RNG.normal(5.0, 1.0, n - int(0.6*n) - int(0.3*n))
        stakes = np.concatenate([s1, s2, s3])
        stakes = np.clip(stakes, 0.05, max_stake*1.5)
    elif stake_dist == 'power':
        stakes = RNG.pareto(1.5, n) * 0.5 + 0.1

    # Arrival times
    if arrival_dist == 'uniform':
        times = np.sort(RNG.uniform(0, T, n))
    elif arrival_dist == 'exponential':
        gaps = RNG.exponential(T/n, n)
        times = np.cumsum(gaps)

    # Payout: π(i,j) = v_j · α · (v_i / V_j⁻)
    # E_i = α · v_i · Σ_{j>i} v_j / V_j⁻
    earnings = np.zeros(n)
    cum_pool = np.cumsum(stakes)  # V_j = Σ_{k≤j} v_k (1-indexed via shift)

    for i in range(n):
        for j in range(i+1, n):
            V_j_minus = cum_pool[j-1] if j > 0 else 0
            if V_j_minus > 0:
                earnings[i] += stakes[j] * alpha * (stakes[i] / V_j_minus)

    total_pool = stakes.sum()
    platform_revenue = total_pool * beta
    creator_revenue = total_pool * gamma  # simplified: creator gets γ of total

    df = pd.DataFrame({
        'curator_idx': np.arange(n),
        'arrival_time': times,
        'stake': stakes,
        'earnings': earnings,
        'net': earnings - stakes,
        'roi': (earnings - stakes) / stakes,
        'platform_revenue': platform_revenue / n,
        'creator_revenue': creator_revenue / n,
    })
    return df, total_pool, platform_revenue, creator_revenue


def ranking(stakes, times, T_now, mode='exponential', half_life=20):
    """R(S,t) = Σ w(t - t_i) · v_i"""
    dt = T_now - times
    if mode == 'accumulation':
        w = np.ones_like(dt)
    elif mode == 'exponential':
        w = np.exp(-np.log(2) * dt / half_life)
    elif mode == 'power':
        w = 1.0 / (1 + dt)**0.5
    elif mode == 'step':
        w = (dt <= half_life).astype(float)
    return (w * stakes).sum()


# ─── Plot 1: Bubble chart — curator stake vs. earnings (profit colored) ──────

def plot_bubble(ax, df):
    profit = df['net'] > 0
    size = np.abs(df['earnings']) * 30 + 5

    ax.scatter(df.loc[~profit, 'curator_idx'], df.loc[~profit, 'stake'],
               s=size[~profit], c='#e05252', alpha=0.6, label="Didn't recoup", edgecolors='none')
    ax.scatter(df.loc[profit, 'curator_idx'], df.loc[profit, 'stake'],
               s=size[profit], c='#52c0e0', alpha=0.7, label='Profited', edgecolors='none')

    ax.set_xlabel('Arrival order (curator index)')
    ax.set_ylabel('Stake amount')
    ax.set_title('Fig 1 — Curator stake vs. earnings\n(bubble size = total earned)')
    ax.legend(framealpha=0.2)
    ax.grid(True)

    # Annotation
    total = df['stake'].sum()
    ax.text(0.98, 0.97,
            f"Total staked: {total:.2f}\nCurators profitable: {profit.sum()}/{len(df)}",
            transform=ax.transAxes, ha='right', va='top',
            fontsize=8, color='#aaa',
            bbox=dict(boxstyle='round,pad=0.3', fc='#111', alpha=0.6))


# ─── Plot 2: Cumulative earnings trajectory for selected curators ─────────────

def plot_earnings_trajectory(ax, df):
    """Show how E_i accumulates as later curators arrive."""
    n = len(df)
    stakes = df['stake'].values
    cum_pool = np.cumsum(stakes)
    alpha = 0.5

    # Track 5 curators: very early, early, mid, late, very late
    highlight = [0, int(n*0.1), int(n*0.3), int(n*0.6), int(n*0.85)]
    colors = ['#f5a623', '#52c0e0', '#7ed321', '#bd10e0', '#e05252']

    for idx, color in zip(highlight, colors):
        traj = []
        cumulative = 0
        for j in range(idx+1, n):
            V_j_minus = cum_pool[j-1] if j > 0 else 0
            if V_j_minus > 0:
                cumulative += stakes[j] * alpha * (stakes[idx] / V_j_minus)
            traj.append(cumulative)

        xs = np.arange(idx+1, n)
        ax.plot(xs, traj, color=color, linewidth=1.5,
                label=f'Curator #{idx} (staked {stakes[idx]:.2f})')
        ax.axhline(stakes[idx], color=color, linewidth=0.5, linestyle=':')

    ax.set_xlabel('Curator arrivals (j)')
    ax.set_ylabel('Cumulative earnings E_i')
    ax.set_title('Fig 2 — Earnings accumulation by arrival order\n(dotted = break-even)')
    ax.legend(framealpha=0.2, fontsize=7)
    ax.grid(True)


# ─── Plot 3: Ranking dynamics under different weight functions ────────────────

def plot_ranking_dynamics(ax, df):
    T_vals = np.linspace(df['arrival_time'].max(), df['arrival_time'].max() * 3, 200)
    stakes = df['stake'].values
    times = df['arrival_time'].values

    modes = {
        'Accumulation (flat)': ('accumulation', '#f5a623'),
        'Exponential decay': ('exponential', '#52c0e0'),
        'Power law decay': ('power', '#7ed321'),
        'Step window (20t)': ('step', '#e05252'),
    }

    for label, (mode, color) in modes.items():
        ranks = [ranking(stakes, times, t, mode=mode, half_life=20) for t in T_vals]
        ax.plot(T_vals, ranks, label=label, color=color, linewidth=1.8)

    ax.axvline(times.max(), color='#555', linestyle='--', linewidth=1)
    ax.text(times.max() + 1, ax.get_ylim()[1]*0.9 if ax.get_ylim()[1] > 0 else 1,
            'last stake', color='#555', fontsize=7)
    ax.set_xlabel('Time t')
    ax.set_ylabel('Rank score R(S,t)')
    ax.set_title('Fig 3 — Ranking decay after last stake\nby weight function w(·)')
    ax.legend(framealpha=0.2, fontsize=8)
    ax.grid(True)


# ─── Plot 4: Phase diagram — ROI vs. arrival fraction, varying alpha ──────────

def plot_phase_diagram(ax):
    n = 60
    alphas = [0.3, 0.5, 0.7, 0.9]
    colors = ['#f5a623', '#52c0e0', '#7ed321', '#e05252']

    for alpha, color in zip(alphas, colors):
        df, *_ = simulate(n=n, alpha=alpha, stake_dist='uniform')
        arrival_frac = df['curator_idx'] / n
        # smooth ROI
        from scipy.ndimage import uniform_filter1d
        roi_smooth = uniform_filter1d(df['roi'].values, size=7)
        ax.plot(arrival_frac, roi_smooth, label=f'α={alpha}', color=color, linewidth=2)

    ax.axhline(0, color='#666', linewidth=1, linestyle='--')
    ax.set_xlabel('Arrival fraction (0=first, 1=last)')
    ax.set_ylabel('ROI  (earnings − stake) / stake')
    ax.set_title('Fig 4 — Curator ROI by arrival order\nacross curator pool share α')
    ax.legend(framealpha=0.2)
    ax.grid(True)


# ─── Plot 5: Revenue split parametric sweep ───────────────────────────────────

def plot_revenue_split(ax):
    alphas = np.linspace(0.05, 0.85, 30)
    beta = 0.1  # fixed platform cut

    curator_rev, creator_rev, platform_rev = [], [], []
    for alpha in alphas:
        df, total, plat, creator = simulate(n=60, alpha=alpha, beta=beta)
        curator_rev.append(df['earnings'].sum())
        platform_rev.append(plat)
        creator_rev.append(creator)

    curator_rev = np.array(curator_rev)
    creator_rev = np.array(creator_rev)
    platform_rev = np.array(platform_rev)
    total = curator_rev + creator_rev + platform_rev

    ax.stackplot(alphas,
                 curator_rev / total,
                 creator_rev / total,
                 platform_rev / total,
                 labels=['Curators (α)', 'Creator (γ=1−α−β)', 'Platform (β=0.1)'],
                 colors=['#52c0e0', '#7ed321', '#f5a623'],
                 alpha=0.85)

    ax.set_xlabel('Curator pool share α')
    ax.set_ylabel('Revenue fraction')
    ax.set_title('Fig 5 — Revenue split across agents\n(β fixed at 0.10)')
    ax.legend(loc='upper right', framealpha=0.3, fontsize=8)
    ax.set_xlim(alphas[0], alphas[-1])
    ax.set_ylim(0, 1)
    ax.grid(True)


# ─── Plot 6: Consumer discovery curve ─────────────────────────────────────────

def plot_discovery(ax):
    """
    Simulate rank rising as curators stake over time.
    Show when rank crosses consumer discovery threshold.
    Consumer surplus ~ quality - cost(rank).
    """
    n = 80
    df, *_ = simulate(n=n, alpha=0.5, stake_dist='mixed_normal', arrival_dist='exponential', T=50)
    stakes = df['stake'].values
    times = df['arrival_time'].values
    t_end = times.max() * 2
    T_vals = np.linspace(0, t_end, 300)

    rank_exp = [ranking(stakes[:sum(times <= t)], times[times <= t], t, mode='exponential') if sum(times <= t) > 0 else 0 for t in T_vals]
    rank_acc = [ranking(stakes[:sum(times <= t)], times[times <= t], t, mode='accumulation') if sum(times <= t) > 0 else 0 for t in T_vals]

    threshold = max(rank_exp) * 0.35  # discovery threshold

    ax.plot(T_vals, rank_acc, color='#f5a623', linewidth=1.5, label='Accumulation rank', alpha=0.6)
    ax.plot(T_vals, rank_exp, color='#52c0e0', linewidth=2, label='Exponential rank')
    ax.axhline(threshold, color='#e05252', linestyle='--', linewidth=1.2, label=f'Discovery threshold (35% peak)')

    # Mark first crossing
    crossings = np.where(np.array(rank_exp) >= threshold)[0]
    if len(crossings):
        t_cross = T_vals[crossings[0]]
        ax.axvline(t_cross, color='#e05252', linestyle=':', linewidth=1)
        ax.text(t_cross + 0.5, threshold * 1.05, f'Δτ* = {t_cross:.1f}', color='#e05252', fontsize=8)

    # Shade consumer discovery zone
    ax.fill_between(T_vals, 0, threshold, alpha=0.05, color='#e05252')
    ax.fill_between(T_vals, threshold, max(rank_exp)*1.1, alpha=0.05, color='#52c0e0')
    ax.text(t_end * 0.7, threshold * 0.5, 'undiscovered', color='#e05252', fontsize=7, alpha=0.7)
    ax.text(t_end * 0.7, threshold * 1.3, 'consumer-visible', color='#52c0e0', fontsize=7, alpha=0.7)

    ax.set_xlabel('Time')
    ax.set_ylabel('Rank score R(S,t)')
    ax.set_title('Fig 6 — Consumer discovery dynamics\n(rank crossing threshold Δτ*)')
    ax.legend(framealpha=0.2, fontsize=8)
    ax.grid(True)


# ─── Compose ──────────────────────────────────────────────────────────────────

def main():
    fig = plt.figure(figsize=(18, 22))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.32)

    df_uniform, *_ = simulate(n=80, alpha=0.5, stake_dist='uniform')
    df_mixed, *_ = simulate(n=80, alpha=0.5, stake_dist='mixed_normal')

    ax1 = fig.add_subplot(gs[0, 0]); plot_bubble(ax1, df_mixed)
    ax2 = fig.add_subplot(gs[0, 1]); plot_earnings_trajectory(ax2, df_uniform)
    ax3 = fig.add_subplot(gs[1, 0]); plot_ranking_dynamics(ax3, df_uniform)
    ax4 = fig.add_subplot(gs[1, 1]); plot_phase_diagram(ax4)
    ax5 = fig.add_subplot(gs[2, 0]); plot_revenue_split(ax5)
    ax6 = fig.add_subplot(gs[2, 1]); plot_discovery(ax6)

    fig.suptitle(
        'Curation Market Model — Formal Model v1 Visualizations\n'
        'π(i,j) = v_j · α · (v_i / V_j⁻)   |   R(S,t) = Σ w(t−tᵢ)·vᵢ',
        fontsize=13, color='#fff', y=0.995
    )

    out = 'research/curation-market-model/model_visualizations.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0f0f0f')
    print(f'Saved: {out}')
    return out


if __name__ == '__main__':
    main()
