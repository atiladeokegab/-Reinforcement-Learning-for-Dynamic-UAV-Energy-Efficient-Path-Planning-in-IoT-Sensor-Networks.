"""
Cross-Layout Generalisation Analysis
=====================================
Tests whether the trained DQN policy (trained on uniformly random sensor
layouts via domain randomisation) generalises to structurally different
sensor deployment patterns it was NOT explicitly trained on.

Layout types (all N=20, 500×500, 5 seeds each):
  random      -- uniform random (training distribution, baseline)
  clustered   -- 4 tight clusters in the 4 quadrant corners
  one_quadrant-- all sensors in the bottom-left quadrant (worst-case asymmetry)
  grid_aligned-- sensors on a regular 4×5 grid
  ring        -- sensors arranged on a circle at mid-radius
  two_clusters-- 2 dense clusters (one near centre, one at periphery)

Outputs (to cross_layout_results/):
  fig_cross_layout_reward.png
  fig_cross_layout_fairness.png
  fig_cross_layout_trajectories.png
  cross_layout_results.csv

Author: ATILADE GABRIEL OKE
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gymnasium
from pathlib import Path
from scipy import stats
import ieee_style
ieee_style.apply()
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

script_dir = Path(__file__).resolve().parent
src_dir    = script_dir.parent.parent.parent
sys.path.insert(0, str(src_dir))

from environment.uav_env import UAVEnvironment
from greedy_agents import MaxThroughputGreedyV2, NearestSensorGreedy

# ==================== CONFIG ====================

SEEDS         = [42, 123, 256, 789, 1337, 2024, 999, 314, 555, 2048]
NUM_SENSORS   = 20
GRID_SIZE     = (500, 500)
MAX_STEPS     = 2100
N_STACK       = 4
MAX_SEN_LIMIT = 50
EVAL_BATTERY  = 274.0
W, H          = GRID_SIZE

BASE_ENV_KWARGS = {
    "grid_size":          GRID_SIZE,
    "num_sensors":        NUM_SENSORS,
    "max_steps":          MAX_STEPS,
    "path_loss_exponent": 3.8,
    "rssi_threshold":     -120.0,
    "sensor_duty_cycle":  10.0,
    "max_battery":        EVAL_BATTERY,
}

OUTPUT_DIR = script_dir / "cross_layout_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_MODEL_CANDIDATES = [
    script_dir.parent / "models" / "dqn_domain_rand"  / "dqn_clustered.zip",   # fine-tuned (layout-diversified)
    script_dir.parent / "models" / "dqn_domain_rand"  / "dqn_final.zip",       # original fallback
    script_dir.parent / "models" / "dqn_fairness_framestack" / "dqn_final.zip",
    script_dir.parent / "models" / "dqn_fairness"            / "dqn_final.zip",
]

AGENTS = {
    "DQN":             None,           # filled in main
    "SF-Aware Greedy": MaxThroughputGreedyV2,
    "Nearest Greedy":  NearestSensorGreedy,
}

AGENT_COLORS = {
    "DQN":             "#1b9e77",
    "SF-Aware Greedy": "#d95f02",
    "Nearest Greedy":  "#7570b3",
}

AGENT_LINESTYLES = {
    "DQN":             "-",
    "SF-Aware Greedy": "--",
    "Nearest Greedy":  ":",
}

LAYOUT_LABELS = {
    "random":       "Uniform Random\n(training dist.)",
    "clustered":    "4-Corner Clusters",
    "one_quadrant": "Single Quadrant",
    "grid_aligned": "Grid-Aligned",
    "ring":         "Ring Layout",
    "two_clusters": "Two Clusters",
}


# ==================== LAYOUT GENERATORS ====================

def _jitter(pos, seed, sigma=8.0):
    rng = np.random.default_rng(seed)
    return [(float(np.clip(x + rng.normal(0, sigma), 5, W-5)),
             float(np.clip(y + rng.normal(0, sigma), 5, H-5)))
            for x, y in pos]


def layout_random(seed):
    rng = np.random.default_rng(seed)
    return [(float(rng.uniform(10, W-10)), float(rng.uniform(10, H-10)))
            for _ in range(NUM_SENSORS)]


def layout_clustered(seed):
    """5 sensors near each of the 4 corners."""
    centres = [(100, 100), (400, 100), (100, 400), (400, 400)]
    base = []
    for cx, cy in centres:
        for _ in range(NUM_SENSORS // 4):
            base.append((float(cx), float(cy)))
    # top up to NUM_SENSORS
    while len(base) < NUM_SENSORS:
        base.append(base[len(base) % len(centres)])
    return _jitter(base[:NUM_SENSORS], seed, sigma=20)


def layout_one_quadrant(seed):
    """All sensors in the bottom-left quadrant."""
    rng = np.random.default_rng(seed)
    return [(float(rng.uniform(10, W//2 - 10)), float(rng.uniform(10, H//2 - 10)))
            for _ in range(NUM_SENSORS)]


def layout_grid_aligned(seed):
    """Regular 4×5 grid with small jitter."""
    cols, rows = 5, 4
    xs = np.linspace(60, W - 60, cols)
    ys = np.linspace(60, H - 60, rows)
    base = [(float(x), float(y)) for y in ys for x in xs]
    return _jitter(base[:NUM_SENSORS], seed, sigma=10)


def layout_ring(seed):
    """Sensors arranged on a circle at radius 200 centred on the grid."""
    cx, cy = W / 2, H / 2
    angles = np.linspace(0, 2 * np.pi, NUM_SENSORS, endpoint=False)
    base   = [(float(cx + 200 * np.cos(a)), float(cy + 200 * np.sin(a)))
              for a in angles]
    return _jitter(base, seed, sigma=8)


def layout_two_clusters(seed):
    """One dense cluster near the centre, one at the periphery."""
    centre = [(float(W/2), float(H/2))] * 10
    corner = [(float(420), float(420))] * 10
    return _jitter(centre + corner, seed, sigma=25)


LAYOUT_GENERATORS = {
    "random":       layout_random,
    "clustered":    layout_clustered,
    "one_quadrant": layout_one_quadrant,
    "grid_aligned": layout_grid_aligned,
    "ring":         layout_ring,
    "two_clusters": layout_two_clusters,
}


# ==================== ENVIRONMENT WRAPPERS ====================

class FixedLayoutEnv(UAVEnvironment):
    def __init__(self, fixed_positions, **kwargs):
        self._fixed_positions = fixed_positions
        super().__init__(**kwargs)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        for s, p in zip(self.sensors, self._fixed_positions):
            s.position = np.array(p, dtype=np.float32)
        return obs, info


class PaddedFixedEnv(FixedLayoutEnv):
    def __init__(self, fixed_positions, **kwargs):
        super().__init__(fixed_positions, **kwargs)
        raw = self.observation_space.shape[0]
        self._fps = 0
        for uav_f in range(raw + 1):
            rem = raw - uav_f
            if rem > 0 and rem % self.num_sensors == 0:
                self._fps = rem // self.num_sensors
                break
        padded = raw + (MAX_SEN_LIMIT - self.num_sensors) * self._fps
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(padded,), dtype=np.float32
        )

    def _pad(self, raw):
        return np.concatenate(
            [raw, np.zeros((MAX_SEN_LIMIT - self.num_sensors) * self._fps, dtype=np.float32)]
        ).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        return self._pad(obs), info

    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        return self._pad(obs), r, term, trunc, info


def _unwrap_base_env(vec):
    inner = vec
    while hasattr(inner, "venv"):
        inner = inner.venv
    e = inner.envs[0]
    while hasattr(e, "env"):
        e = e.env
    return e


def compute_jains(env):
    rates = []
    for s in env.sensors:
        gen = float(s.total_data_generated)
        rates.append((float(s.total_data_transmitted) / gen * 100) if gen > 0 else 0.0)
    n  = len(rates); s1 = sum(rates); s2 = sum(x**2 for x in rates)
    return (s1**2) / (n * s2) if s2 > 0 else 1.0


# ==================== EPISODE RUNNERS ====================

def run_dqn_episode(model, positions):
    env     = PaddedFixedEnv(positions, **BASE_ENV_KWARGS)
    vec_env = VecFrameStack(DummyVecEnv([lambda: env]), n_stack=N_STACK)
    base    = _unwrap_base_env(vec_env)
    obs     = vec_env.reset()
    cum_r   = 0.0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done_arr, _ = vec_env.step(action)
        cum_r += float(r[0])
        if bool(done_arr[0]):
            break
    energy = EVAL_BATTERY - base.uav.battery
    return {
        "reward":     cum_r,
        "jains":      compute_jains(base),
        "coverage":   len(base.sensors_visited) / base.num_sensors * 100,
        "efficiency": float(base.total_data_collected / energy) if energy > 0 else 0.0,
        "trajectory": list(zip(
            [float(base.uav.position[0])], [float(base.uav.position[1])]
        )),
    }


def run_greedy_episode(agent_class, positions):
    env    = FixedLayoutEnv(positions, **BASE_ENV_KWARGS)
    obs, _ = env.reset()
    agent  = agent_class(env)
    cum_r  = 0.0
    while True:
        action = agent.select_action(obs)
        obs, r, done, trunc, _ = env.step(action)
        cum_r += r
        if done or trunc:
            break
    energy = EVAL_BATTERY - env.uav.battery
    return {
        "reward":     cum_r,
        "jains":      compute_jains(env),
        "coverage":   len(env.sensors_visited) / env.num_sensors * 100,
        "efficiency": float(env.total_data_collected / energy) if energy > 0 else 0.0,
    }


# ==================== MAIN SWEEP ====================

def run_all_layouts(model):
    records = []
    for layout_name, gen_fn in LAYOUT_GENERATORS.items():
        print(f"\n  Layout: {layout_name}")
        for seed in SEEDS:
            positions = gen_fn(seed)

            # DQN
            r = run_dqn_episode(model, positions)
            records.append({
                "layout": layout_name, "agent": "DQN",
                "seed": seed, **{k: v for k, v in r.items() if k != "trajectory"}
            })

            # Greedy baselines
            for name, cls in [("SF-Aware Greedy", MaxThroughputGreedyV2),
                               ("Nearest Greedy",  NearestSensorGreedy)]:
                gr = run_greedy_episode(cls, positions)
                records.append({"layout": layout_name, "agent": name, "seed": seed, **gr})

        # Print per-layout summary
        sub = pd.DataFrame(records)
        sub = sub[sub["layout"] == layout_name]
        for agent in ["DQN", "SF-Aware Greedy", "Nearest Greedy"]:
            ag = sub[sub["agent"] == agent]
            print(f"    {agent:20s}: reward={ag['reward'].mean():>10.0f}±{ag['reward'].std():.0f}"
                  f"  J={ag['jains'].mean():.4f}")

    return pd.DataFrame(records)


# ==================== PLOTTING ====================

def plot_reward_comparison(df):
    layouts = list(LAYOUT_GENERATORS.keys())
    agents  = ["DQN", "SF-Aware Greedy", "Nearest Greedy"]
    x       = np.arange(len(layouts))
    width   = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, agent in enumerate(agents):
        means, stds = [], []
        for layout in layouts:
            sub   = df[(df["layout"] == layout) & (df["agent"] == agent)]["reward"]
            means.append(sub.mean())
            stds.append(sub.std())
        ax.bar(x + i * width, means, width, yerr=stds, label=agent,
               color=AGENT_COLORS[agent], alpha=0.85, capsize=4)

    ax.set_xticks(x + width)
    ax.set_xticklabels([LAYOUT_LABELS[l] for l in layouts], fontsize=9)
    ax.set_ylabel("Mean Cumulative Reward (×10⁶)", fontsize=11)
    ax.set_title("DQN Cross-Layout Generalisation — Cumulative Reward\n"
                 "(N=20, 500×500, 5 seeds per layout)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}"))
    fig.tight_layout()
    ieee_style.clean_figure(fig)
    ieee_style.save(fig, str(OUTPUT_DIR / "fig_cross_layout_reward"))
    plt.close(fig)
    print("  Saved fig_cross_layout_reward.png")


def plot_fairness_comparison(df):
    layouts = list(LAYOUT_GENERATORS.keys())
    agents  = ["DQN", "SF-Aware Greedy", "Nearest Greedy"]
    x       = np.arange(len(layouts))
    width   = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, agent in enumerate(agents):
        means, stds = [], []
        for layout in layouts:
            sub   = df[(df["layout"] == layout) & (df["agent"] == agent)]["jains"]
            means.append(sub.mean())
            stds.append(sub.std())
        ax.bar(x + i * width, means, width, yerr=stds, label=agent,
               color=AGENT_COLORS[agent], alpha=0.85, capsize=4)

    ax.set_xticks(x + width)
    ax.set_xticklabels([LAYOUT_LABELS[l] for l in layouts], fontsize=9)
    ax.set_ylabel("Mean Jain's Fairness Index", fontsize=11)
    ax.set_title("DQN Cross-Layout Generalisation — Jain's Fairness Index\n"
                 "(N=20, 500×500, 5 seeds per layout)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0.5, 1.05)
    fig.tight_layout()
    ieee_style.clean_figure(fig)
    ieee_style.save(fig, str(OUTPUT_DIR / "fig_cross_layout_fairness"))
    plt.close(fig)
    print("  Saved fig_cross_layout_fairness.png")


def plot_dqn_advantage(df):
    """DQN % advantage over SF-Aware Greedy per layout."""
    layouts = list(LAYOUT_GENERATORS.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, metric, ylabel, scale in [
        (axes[0], "reward", "DQN Reward Advantage over SF-Aware Greedy (%)", 1e6),
        (axes[1], "jains",  "DQN Jain's Index Advantage over SF-Aware Greedy (pp)", 1.0),
    ]:
        adv_means, adv_stds = [], []
        for layout in layouts:
            dqn_r = df[(df["layout"] == layout) & (df["agent"] == "DQN")][metric]
            sf_r  = df[(df["layout"] == layout) & (df["agent"] == "SF-Aware Greedy")][metric]
            if metric == "reward":
                adv = ((dqn_r.values - sf_r.values) / sf_r.values * 100)
            else:
                adv = (dqn_r.values - sf_r.values) * 100
            adv_means.append(adv.mean())
            adv_stds.append(adv.std())

        colors = ["#1b9e77" if v >= 0 else "#d95f02" for v in adv_means]
        bars = ax.bar(range(len(layouts)), adv_means, yerr=adv_stds,
                      color=colors, alpha=0.85, capsize=4)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(len(layouts)))
        ax.set_xticklabels([LAYOUT_LABELS[l] for l in layouts], fontsize=8, rotation=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("DQN Generalisation Advantage Across Layout Types\n"
                 "(blue = DQN leads, red = DQN trails)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    ieee_style.clean_figure(fig)
    ieee_style.save(fig, str(OUTPUT_DIR / "fig_cross_layout_advantage"))
    plt.close(fig)
    print("  Saved fig_cross_layout_advantage.png")


def plot_layout_previews():
    """Show one example of each layout type (seed=42)."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    for ax, (name, gen_fn) in zip(axes, LAYOUT_GENERATORS.items()):
        pos = gen_fn(42)
        xs, ys = zip(*pos)
        ax.scatter(xs, ys, s=40, c="#d95f02", alpha=0.8, zorder=3)
        ax.scatter([0], [0], s=60, marker="^", c="#1b9e77", zorder=4, label="UAV start")
        ax.set_xlim(0, W); ax.set_ylim(0, H)
        ax.set_title(LAYOUT_LABELS[name], fontsize=10, fontweight="bold")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
    fig.suptitle("Sensor Layout Types Used in Cross-Layout Generalisation Test\n"
                 "(seed=42, N=20, 500×500)", fontsize=12)
    fig.tight_layout()
    ieee_style.clean_figure(fig)
    ieee_style.save(fig, str(OUTPUT_DIR / "fig_layout_previews"))
    plt.close(fig)
    print("  Saved fig_layout_previews.png")


# ==================== STATS ====================

def print_stats(df):
    layouts = list(LAYOUT_GENERATORS.keys())
    print("\n=== CROSS-LAYOUT SUMMARY ===")
    print(f"{'Layout':<20} {'DQN Reward':>12} {'SF-Aware':>12} {'Adv %':>8} "
          f"{'DQN J':>8} {'SF J':>8}")
    print("-" * 72)
    for layout in layouts:
        dqn_r  = df[(df["layout"] == layout) & (df["agent"] == "DQN")]["reward"]
        sf_r   = df[(df["layout"] == layout) & (df["agent"] == "SF-Aware Greedy")]["reward"]
        dqn_j  = df[(df["layout"] == layout) & (df["agent"] == "DQN")]["jains"]
        sf_j   = df[(df["layout"] == layout) & (df["agent"] == "SF-Aware Greedy")]["jains"]
        adv    = (dqn_r.mean() / sf_r.mean() - 1) * 100
        t, p   = stats.ttest_ind(dqn_r, sf_r, equal_var=False)  # Welch's t-test
        # Cohen's d
        pooled_std = np.sqrt((dqn_r.std()**2 + sf_r.std()**2) / 2)
        d = abs(dqn_r.mean() - sf_r.mean()) / pooled_std if pooled_std > 0 else 0.0
        print(f"{layout:<20} {dqn_r.mean():>12.0f} {sf_r.mean():>12.0f} "
              f"{adv:>+7.1f}% {dqn_j.mean():>8.4f} {sf_j.mean():>8.4f}  p={p:.3f}  d={d:.2f}")


# ==================== MAIN ====================

def main():
    print("Loading DQN model...")
    model = load_model()

    print("\nGenerating layout previews...")
    plot_layout_previews()

    print("\nRunning cross-layout evaluation...")
    df = run_all_layouts(model)
    df.to_csv(OUTPUT_DIR / "cross_layout_results.csv", index=False)
    print(f"\nResults saved: {OUTPUT_DIR / 'cross_layout_results.csv'}")

    print_stats(df)

    print("\nGenerating plots...")
    plot_reward_comparison(df)
    plot_fairness_comparison(df)
    plot_dqn_advantage(df)
    print("\nDone.")


def load_model():
    for p in _MODEL_CANDIDATES:
        if p.exists():
            print(f"  Model: {p}")
            return DQN.load(str(p))
    raise FileNotFoundError("No DQN model found.")


if __name__ == "__main__":
    main()
