"""
analyze_sweep.py
================
Reads sweep_results.csv produced by sweep_eval.py and generates all
dissertation-quality plots and statistical tables.

Outputs (all in baseline_results/sweep/):
  sweep_summary.csv             — mean ± std ± 95% CI per (agent, config)
  sweep_ndr_ci.png              — NDR vs grid config, shaded 95% CI
  sweep_jfi_ci.png              — JFI vs grid config, shaded 95% CI
  sweep_aoi.png                 — Mean & peak AoI vs grid config
  sweep_alpha_fairness.png      — α-fairness (α=2) vs grid config
  sweep_boxplot_ndr.png         — Box-and-whisker NDR per agent × config
  sweep_boxplot_jfi.png         — Box-and-whisker JFI per agent × config
  sweep_iqm.png                 — IQM + bootstrap CI bar chart (NDR)
  sweep_performance_profiles.png — Fraction of runs above threshold τ
  sweep_prob_improvement.png    — P(agent A > agent B) heatmap
  sweep_heatmap_ndr.png         — Grid × sensors → NDR per agent (cross sweep)
  sweep_heatmap_gini.png        — Grid × sensors → Gini per agent
  sweep_heatmap_aoi.png         — Grid × sensors → mean AoI per agent
  sweep_efficiency.png          — Bytes/Wh vs grid config

Usage:
    PYTHONIOENCODING=utf-8 uv run python \\
        src/agents/dqn/dqn_evaluation_results/analyze_sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scipy.stats as stats

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    import ieee_style
    ieee_style.apply()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# ── CONFIGURATION ────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
OUTPUT_DIR  = _HERE / "baseline_results" / "sweep"
RESULTS_CSV = OUTPUT_DIR / "sweep_results.csv"

# Ordered agent list — determines plot legend order
AGENT_ORDER = [
    "DQN",
    "Relational RL",
    "Smart Greedy V2",
    "Nearest Greedy",
    "TSP Oracle",
    "GTrXL",
]

AGENT_COLORS = {
    "DQN":            "#1f78b4",
    "Relational RL":  "#33a02c",
    "Smart Greedy V2": "#ff7f00",
    "Nearest Greedy": "#6a3d9a",
    "TSP Oracle":     "#e31a1c",
    "GTrXL":          "#b15928",
}
AGENT_MARKERS = {
    "DQN":            "o",
    "Relational RL":  "s",
    "Smart Greedy V2": "^",
    "Nearest Greedy": "v",
    "TSP Oracle":     "D",
    "GTrXL":          "P",
}

# Curriculum-aligned config labels (for main sweep plots)
MAIN_LABELS = ["100×100\nN=10", "200×200\nN=20", "300×300\nN=30",
               "400×400\nN=40", "500×500\nN=50"]
MAIN_GRIDS  = [(100, 100), (200, 200), (300, 300), (400, 400), (500, 500)]
MAIN_N      = [10, 20, 30, 40, 50]

N_BOOTSTRAP = 2_000
CI_LEVEL    = 0.95


# ---------------------------------------------------------------------------
# ── STATISTICAL HELPERS ───────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def iqm(x: np.ndarray) -> float:
    """Interquartile Mean — drops bottom and top 25% before averaging."""
    x = np.sort(x)
    n  = len(x)
    lo = int(np.floor(0.25 * n))
    hi = int(np.ceil(0.75 * n))
    return float(np.mean(x[lo:hi]))


def bootstrap_ci(x: np.ndarray, stat_fn=np.mean,
                 n_boot: int = N_BOOTSTRAP, ci: float = CI_LEVEL):
    """Return (point_estimate, lower_bound, upper_bound) via stratified bootstrap."""
    rng = np.random.default_rng(0)
    boots = [stat_fn(rng.choice(x, size=len(x), replace=True)) for _ in range(n_boot)]
    alpha = (1.0 - ci) / 2.0
    return float(stat_fn(x)), float(np.percentile(boots, alpha * 100)), float(np.percentile(boots, (1 - alpha) * 100))


def prob_improvement(a: np.ndarray, b: np.ndarray) -> float:
    """P(A > B) estimated across all cross-seed pairs."""
    total = len(a) * len(b)
    if total == 0:
        return 0.5
    wins = sum(1 for x in a for y in b if x > y)
    return wins / total


def welch_t(a: np.ndarray, b: np.ndarray):
    """Return (t_stat, p_value) for Welch's t-test (unequal variance)."""
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(t), float(p)


def mean_ci(x: np.ndarray, ci: float = CI_LEVEL):
    """Mean + CI half-width using t-distribution."""
    n = len(x)
    if n < 2:
        return float(np.mean(x)), 0.0
    se = stats.sem(x)
    h  = se * stats.t.ppf((1 + ci) / 2, df=n - 1)
    return float(np.mean(x)), float(h)


# ---------------------------------------------------------------------------
# ── DATA LOADING & FILTERING ──────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"Results CSV not found: {RESULTS_CSV}")
    df = pd.read_csv(RESULTS_CSV)
    main  = df[df["sweep_type"] == "main"].copy()
    cross = df[df["sweep_type"] == "cross"].copy()
    main["config_label"] = main.apply(
        lambda r: f"{int(r.grid_w)}×{int(r.grid_h)}\nN={int(r.n_sensors)}", axis=1
    )
    agents_present = [a for a in AGENT_ORDER if a in df["agent"].unique()]
    return main, cross, agents_present


# ---------------------------------------------------------------------------
# ── SUMMARY TABLE ────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def make_summary(main: pd.DataFrame, agents: list[str]) -> pd.DataFrame:
    rows = []
    for agent in agents:
        for (gw, gh), n, label in zip(MAIN_GRIDS, MAIN_N, MAIN_LABELS):
            sub = main[
                (main["agent"] == agent) &
                (main["grid_w"] == gw) &
                (main["n_sensors"] == n)
            ]
            if sub.empty:
                continue
            for metric in ["ndr_pct", "jfi", "bytes_per_wh", "mean_aoi", "alpha_fairness"]:
                vals = sub[metric].dropna().values
                if len(vals) == 0:
                    continue
                mu, ci_h = mean_ci(vals)
                iqm_val, iqm_lo, iqm_hi = bootstrap_ci(vals, iqm)
                rows.append({
                    "agent": agent, "grid": f"{gw}x{gh}", "n_sensors": n,
                    "metric": metric,
                    "mean": round(mu, 3),
                    "std":  round(float(np.std(vals)), 3),
                    "ci_half": round(ci_h, 3),
                    "ci_lo":  round(mu - ci_h, 3),
                    "ci_hi":  round(mu + ci_h, 3),
                    "iqm":    round(iqm_val, 3),
                    "iqm_lo": round(iqm_lo, 3),
                    "iqm_hi": round(iqm_hi, 3),
                    "n_samples": len(vals),
                })
    df = pd.DataFrame(rows)
    path = OUTPUT_DIR / "sweep_summary.csv"
    df.to_csv(path, index=False)
    print(f"Summary → {path}")
    return df


# ---------------------------------------------------------------------------
# ── PLOT HELPERS ──────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def _agent_sorted(agents):
    return [a for a in AGENT_ORDER if a in agents]


def _legend_handles(agents):
    return [
        Line2D([0], [0], color=AGENT_COLORS.get(a, "grey"),
               marker=AGENT_MARKERS.get(a, "o"), linewidth=1.8,
               markersize=6, label=a)
        for a in agents
    ]


def _save(fig, name: str) -> None:
    path = OUTPUT_DIR / name
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


# ---------------------------------------------------------------------------
# ── PLOT 1 & 2: CI band line plots (NDR, JFI) ────────────────────────────────
# ---------------------------------------------------------------------------

def plot_ci_line(main: pd.DataFrame, agents: list[str],
                 metric: str, ylabel: str, filename: str) -> None:
    x = np.arange(len(MAIN_LABELS))
    fig, ax = plt.subplots(figsize=(7, 4))
    for agent in _agent_sorted(agents):
        means, lo, hi = [], [], []
        for (gw, gh), n in zip(MAIN_GRIDS, MAIN_N):
            vals = main[
                (main["agent"] == agent) & (main["grid_w"] == gw) & (main["n_sensors"] == n)
            ][metric].dropna().values
            if len(vals) == 0:
                means.append(np.nan); lo.append(np.nan); hi.append(np.nan)
                continue
            mu, h = mean_ci(vals)
            means.append(mu); lo.append(mu - h); hi.append(mu + h)

        c = AGENT_COLORS.get(agent, "grey")
        m = AGENT_MARKERS.get(agent, "o")
        ax.plot(x, means, color=c, marker=m, linewidth=1.8, markersize=6)
        ax.fill_between(x, lo, hi, color=c, alpha=0.15)

    ax.set_xticks(x); ax.set_xticklabels(MAIN_LABELS, fontsize=8)
    ax.set_xlabel("Environment Configuration")
    ax.set_ylabel(ylabel)
    ax.legend(handles=_legend_handles(_agent_sorted(agents)), framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    _save(fig, filename)


# ---------------------------------------------------------------------------
# ── PLOT 3: AoI ───────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def plot_aoi(main: pd.DataFrame, agents: list[str]) -> None:
    x  = np.arange(len(MAIN_LABELS))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, metric, title in zip(axes, ["mean_aoi", "peak_aoi"],
                                 ["Mean AoI Proxy", "Peak AoI Proxy"]):
        for agent in _agent_sorted(agents):
            means = []
            for (gw, gh), n in zip(MAIN_GRIDS, MAIN_N):
                vals = main[
                    (main["agent"] == agent) & (main["grid_w"] == gw) & (main["n_sensors"] == n)
                ][metric].dropna().values
                means.append(np.mean(vals) if len(vals) > 0 else np.nan)
            c = AGENT_COLORS.get(agent, "grey")
            ax.plot(x, means, color=c, marker=AGENT_MARKERS.get(agent, "o"),
                    linewidth=1.8, markersize=6, label=agent)
        ax.set_xticks(x); ax.set_xticklabels(MAIN_LABELS, fontsize=8)
        ax.set_title(title); ax.set_ylabel("Buffer Occupancy (AoI proxy)")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
    axes[0].legend(framealpha=0.9)
    fig.tight_layout()
    _save(fig, "sweep_aoi.png")


# ---------------------------------------------------------------------------
# ── PLOT 4: α-fairness ───────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def plot_alpha_fairness(main: pd.DataFrame, agents: list[str]) -> None:
    plot_ci_line(main, agents, "alpha_fairness",
                 "α-Fairness (α=2, ↑ better)", "sweep_alpha_fairness.png")


# ---------------------------------------------------------------------------
# ── PLOT 5 & 6: Box plots ────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def plot_boxplot(main: pd.DataFrame, agents: list[str],
                 metric: str, ylabel: str, filename: str) -> None:
    fig, axes = plt.subplots(1, len(MAIN_LABELS), figsize=(14, 4),
                             sharey=True, gridspec_kw={"wspace": 0.05})
    for ax, (gw, gh), n, label in zip(axes, MAIN_GRIDS, MAIN_N, MAIN_LABELS):
        data, colors, labels = [], [], []
        for agent in _agent_sorted(agents):
            vals = main[
                (main["agent"] == agent) & (main["grid_w"] == gw) & (main["n_sensors"] == n)
            ][metric].dropna().values
            if len(vals) > 0:
                data.append(vals)
                colors.append(AGENT_COLORS.get(agent, "grey"))
                labels.append(agent[:6])
        if not data:
            continue
        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops={"color": "black", "linewidth": 1.5},
                        whiskerprops={"linewidth": 1}, capprops={"linewidth": 1},
                        flierprops={"marker": ".", "markersize": 4})
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.7)
        ax.set_xticklabels(labels, rotation=30, fontsize=7)
        ax.set_title(label, fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].set_ylabel(ylabel)
    fig.tight_layout()
    _save(fig, filename)


# ---------------------------------------------------------------------------
# ── PLOT 7: IQM bar chart ────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def plot_iqm(main: pd.DataFrame, agents: list[str]) -> None:
    n_configs = len(MAIN_LABELS)
    n_agents  = len(_agent_sorted(agents))
    width     = 0.8 / n_agents
    x         = np.arange(n_configs)
    fig, ax   = plt.subplots(figsize=(10, 4))

    for i, agent in enumerate(_agent_sorted(agents)):
        iqm_vals, iqm_los, iqm_his = [], [], []
        for (gw, gh), n in zip(MAIN_GRIDS, MAIN_N):
            vals = main[
                (main["agent"] == agent) & (main["grid_w"] == gw) & (main["n_sensors"] == n)
            ]["ndr_pct"].dropna().values
            if len(vals) >= 4:
                v, lo, hi = bootstrap_ci(vals, iqm)
            else:
                v, lo, hi = (np.mean(vals) if len(vals) else np.nan,) * 3
            iqm_vals.append(v); iqm_los.append(v - lo); iqm_his.append(hi - v)

        offset = (i - n_agents / 2 + 0.5) * width
        ax.bar(x + offset, iqm_vals, width=width * 0.9,
               color=AGENT_COLORS.get(agent, "grey"), alpha=0.8, label=agent)
        ax.errorbar(x + offset, iqm_vals,
                    yerr=[iqm_los, iqm_his],
                    fmt="none", color="black", capsize=3, linewidth=1)

    ax.set_xticks(x); ax.set_xticklabels(MAIN_LABELS, fontsize=8)
    ax.set_ylabel("IQM NDR (%) + 95% bootstrap CI")
    ax.set_xlabel("Environment Configuration")
    ax.legend(framealpha=0.9, fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    _save(fig, "sweep_iqm.png")


# ---------------------------------------------------------------------------
# ── PLOT 8: Performance profiles ─────────────────────────────────────────────
# ---------------------------------------------------------------------------

def plot_performance_profiles(main: pd.DataFrame, agents: list[str]) -> None:
    thresholds = np.linspace(0, 100, 200)
    fig, ax = plt.subplots(figsize=(7, 4))
    for agent in _agent_sorted(agents):
        scores = main[main["agent"] == agent]["ndr_pct"].dropna().values
        if len(scores) == 0:
            continue
        profile = [float(np.mean(scores >= tau)) for tau in thresholds]
        ax.plot(thresholds, profile,
                color=AGENT_COLORS.get(agent, "grey"),
                linewidth=1.8, label=agent)
    ax.set_xlabel("NDR threshold τ (%)")
    ax.set_ylabel("Fraction of runs ≥ τ")
    ax.legend(framealpha=0.9)
    ax.grid(linestyle="--", alpha=0.4)
    ax.set_xlim(0, 100); ax.set_ylim(0, 1.05)
    fig.tight_layout()
    _save(fig, "sweep_performance_profiles.png")


# ---------------------------------------------------------------------------
# ── PLOT 9: Probability of improvement heatmap ────────────────────────────────
# ---------------------------------------------------------------------------

def plot_prob_improvement(main: pd.DataFrame, agents: list[str]) -> None:
    sorted_agents = _agent_sorted(agents)
    n = len(sorted_agents)
    # Aggregate across all configs
    scores = {a: main[main["agent"] == a]["ndr_pct"].dropna().values
              for a in sorted_agents}

    matrix = np.full((n, n), 0.5)
    for i, a in enumerate(sorted_agents):
        for j, b in enumerate(sorted_agents):
            if i != j and len(scores[a]) > 0 and len(scores[b]) > 0:
                matrix[i, j] = prob_improvement(scores[a], scores[b])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="P(row agent > col agent)")
    ax.set_xticks(range(n)); ax.set_xticklabels(sorted_agents, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(sorted_agents, fontsize=8)
    ax.set_title("Probability of Improvement (NDR, all configs)")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if 0.25 < matrix[i,j] < 0.75 else "white")
    fig.tight_layout()
    _save(fig, "sweep_prob_improvement.png")


# ---------------------------------------------------------------------------
# ── PLOTS 10–12: Cross-sweep heatmaps ────────────────────────────────────────
# ---------------------------------------------------------------------------

def plot_heatmap(cross: pd.DataFrame, agents: list[str],
                 metric: str, title: str, filename: str) -> None:
    sorted_agents = _agent_sorted(agents)
    n_agents = len(sorted_agents)
    grids   = sorted(cross["grid_w"].unique())
    sensors = sorted(cross["n_sensors"].unique())

    fig, axes = plt.subplots(1, n_agents,
                             figsize=(3.5 * n_agents, 3.5),
                             gridspec_kw={"wspace": 0.3})
    if n_agents == 1:
        axes = [axes]

    vmin = cross[metric].quantile(0.05)
    vmax = cross[metric].quantile(0.95)

    for ax, agent in zip(axes, sorted_agents):
        mat = np.full((len(sensors), len(grids)), np.nan)
        for ci, g in enumerate(grids):
            for ri, n in enumerate(sensors):
                vals = cross[
                    (cross["agent"] == agent) &
                    (cross["grid_w"] == g) &
                    (cross["n_sensors"] == n)
                ][metric].dropna().values
                if len(vals) > 0:
                    mat[ri, ci] = np.mean(vals)
        im = ax.imshow(mat, origin="lower", aspect="auto",
                       vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_xticks(range(len(grids)));    ax.set_xticklabels(grids, fontsize=7)
        ax.set_yticks(range(len(sensors)));  ax.set_yticklabels(sensors, fontsize=7)
        ax.set_xlabel("Grid size (one side)")
        ax.set_ylabel("# Sensors")
        ax.set_title(agent, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=10, y=1.02)
    fig.tight_layout()
    _save(fig, filename)


# ---------------------------------------------------------------------------
# ── PLOT 13: Efficiency ────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def plot_efficiency(main: pd.DataFrame, agents: list[str]) -> None:
    plot_ci_line(main, agents, "bytes_per_wh",
                 "Efficiency (bytes / Wh)", "sweep_efficiency.png")


# ---------------------------------------------------------------------------
# ── WELCH'S T-TEST TABLE ──────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def make_significance_table(main: pd.DataFrame, agents: list[str]) -> None:
    baseline = "DQN" if "DQN" in agents else agents[0]
    rows = []
    for agent in agents:
        if agent == baseline:
            continue
        for (gw, gh), n, label in zip(MAIN_GRIDS, MAIN_N, MAIN_LABELS):
            a_vals = main[
                (main["agent"] == baseline) & (main["grid_w"] == gw) & (main["n_sensors"] == n)
            ]["ndr_pct"].dropna().values
            b_vals = main[
                (main["agent"] == agent) & (main["grid_w"] == gw) & (main["n_sensors"] == n)
            ]["ndr_pct"].dropna().values
            t, p = welch_t(a_vals, b_vals)
            rows.append({
                "comparison": f"{baseline} vs {agent}",
                "config": label.replace("\n", " "),
                "mean_A": round(np.mean(a_vals), 2) if len(a_vals) else np.nan,
                "mean_B": round(np.mean(b_vals), 2) if len(b_vals) else np.nan,
                "t_stat": round(t, 3),
                "p_value": round(p, 4),
                "significant": "yes" if p < 0.05 else "no",
            })
    df = pd.DataFrame(rows)
    path = OUTPUT_DIR / "sweep_significance.csv"
    df.to_csv(path, index=False)
    print(f"Significance table → {path}")
    print(df.to_string(index=False))


# ---------------------------------------------------------------------------
# ── ENTRY POINT ───────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def main() -> None:
    main_df, cross_df, agents = load_data()
    print(f"Loaded {len(main_df)} main-sweep rows, {len(cross_df)} cross-sweep rows")
    print(f"Agents present: {agents}")

    make_summary(main_df, agents)
    make_significance_table(main_df, agents)

    # CI line plots
    plot_ci_line(main_df, agents, "ndr_pct", "NDR (%)", "sweep_ndr_ci.png")
    plot_ci_line(main_df, agents, "jfi",     "Jain's Fairness Index", "sweep_jfi_ci.png")

    # AoI and α-fairness
    plot_aoi(main_df, agents)
    plot_alpha_fairness(main_df, agents)

    # Box plots
    plot_boxplot(main_df, agents, "ndr_pct", "NDR (%)",            "sweep_boxplot_ndr.png")
    plot_boxplot(main_df, agents, "jfi",     "Jain's Fairness Index", "sweep_boxplot_jfi.png")

    # Statistical Precipice plots
    plot_iqm(main_df, agents)
    plot_performance_profiles(main_df, agents)
    plot_prob_improvement(main_df, agents)

    # Heatmaps (cross sweep)
    if not cross_df.empty:
        plot_heatmap(cross_df, agents, "ndr_pct",   "NDR (%) — Grid × Sensors",       "sweep_heatmap_ndr.png")
        plot_heatmap(cross_df, agents, "gini",      "Gini — Grid × Sensors",           "sweep_heatmap_gini.png")
        plot_heatmap(cross_df, agents, "mean_aoi",  "Mean AoI — Grid × Sensors",       "sweep_heatmap_aoi.png")
    else:
        print("No cross-sweep data — heatmaps skipped")

    # Efficiency
    plot_efficiency(main_df, agents)

    print(f"\nAll outputs → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
