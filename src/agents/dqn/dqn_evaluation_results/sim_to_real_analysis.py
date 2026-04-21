"""
Sim-to-Real Robustness Analysis
================================
Tests the trained DQN policy under synthetic domain shifts that model
real-world deployment imperfections:

  Perturbation A -- GPS position noise
    Adds N(0, sigma) noise to the UAV position observation at every step.
    Sigma values: 0 (baseline), 1, 5, 10 grid units (≡ 10, 50, 100 m at 10 m/unit)

  Perturbation B -- Path-loss model mismatch
    Evaluates with path_loss_exponent shifted by +delta from the training value of 3.8.
    Delta values: 0 (baseline), +0.2, +0.5, +1.0
    Models real-world channel deviations (urban clutter, foliage, multipath).

  Perturbation C -- Duty-cycle timing uncertainty
    Evaluates with sensor_duty_cycle in {10, 8, 6, 4} % (default = 10).
    Models devices with stricter duty-cycle enforcement than assumed.

For each condition: 5 seeds, N=20, 500x500 grid.

Outputs (to sim_to_real_results/):
  fig_sim_to_real_gps.png
  fig_sim_to_real_pathloss.png
  fig_sim_to_real_dutycycle.png
  sim_to_real_results.csv

Author: ATILADE GABRIEL OKE
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium
from pathlib import Path
from stable_baselines3 import DQN
import ieee_style
ieee_style.apply()
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

script_dir = Path(__file__).resolve().parent
src_dir    = script_dir.parent.parent.parent
sys.path.insert(0, str(src_dir))

from environment.uav_env import UAVEnvironment
from greedy_agents import MaxThroughputGreedyV2, NearestSensorGreedy

# ==================== CONFIG ====================

SEEDS         = [42, 123, 256, 789, 1337]
NUM_SENSORS   = 20
GRID_SIZE     = (500, 500)
MAX_STEPS     = 2100
N_STACK       = 4
MAX_SEN_LIMIT = 50
EVAL_BATTERY  = 274.0

BASE_ENV_KWARGS = {
    "grid_size":          GRID_SIZE,
    "num_sensors":        NUM_SENSORS,
    "max_steps":          MAX_STEPS,
    "path_loss_exponent": 3.8,
    "rssi_threshold":     -85.0,
    "sensor_duty_cycle":  10.0,
    "max_battery":        EVAL_BATTERY,
}

GPS_SIGMAS       = [0, 1, 5, 10]          # grid units
PATHLOSS_DELTAS  = [0.0, 0.2, 0.5, 1.0]  # added to exponent
DUTY_CYCLES      = [10, 8, 6, 4]          # percent
WIND_DRIFTS      = [0.0, 0.5, 1.0, 2.0]  # grid units/step systematic drift magnitude
LATENCY_STEPS    = [0, 1, 2, 4]           # observation delay in timesteps

OUTPUT_DIR = script_dir / "sim_to_real_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_MODEL_CANDIDATES = [
    script_dir.parent / "models" / "dqn_domain_rand"  / "dqn_final.zip",
    script_dir.parent / "models" / "dqn_fairness_framestack" / "dqn_final.zip",
    script_dir.parent / "models" / "dqn_fairness"            / "dqn_final.zip",
]


# ==================== HELPERS ====================

def load_model():
    for p in _MODEL_CANDIDATES:
        if p.exists():
            print(f"  Model: {p}")
            return DQN.load(str(p))
    raise FileNotFoundError("No DQN model found.")


def get_positions(seed, env_kwargs):
    env = UAVEnvironment(**env_kwargs)
    env.reset(seed=seed)
    pos = [tuple(float(v) for v in s.position) for s in env.sensors]
    env.close()
    return pos


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
    n  = len(rates)
    s1 = sum(rates);  s2 = sum(x**2 for x in rates)
    return (s1**2) / (n * s2) if s2 > 0 else 1.0


# ==================== ENVIRONMENT WRAPPERS ====================

class FixedLayoutEnv(UAVEnvironment):
    def __init__(self, fixed_positions, **kwargs):
        self._fixed_positions = fixed_positions
        super().__init__(**kwargs)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        for sensor, pos in zip(self.sensors, self._fixed_positions):
            sensor.position = np.array(pos, dtype=np.float32)
        return obs, info


class PaddedFixedEnv(FixedLayoutEnv):
    """Zero-padded fixed-layout env for DQN inference."""
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


class GPSNoisyEnv(PaddedFixedEnv):
    """Adds Gaussian noise to the UAV position in the observation."""
    def __init__(self, fixed_positions, gps_sigma=0.0, **kwargs):
        self.gps_sigma = gps_sigma
        super().__init__(fixed_positions, **kwargs)

    def _add_gps_noise(self, obs):
        if self.gps_sigma > 0:
            obs = obs.copy()
            obs[0] += np.random.normal(0, self.gps_sigma)
            obs[1] += np.random.normal(0, self.gps_sigma)
        return obs

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        return self._add_gps_noise(obs), info

    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        return self._add_gps_noise(obs), r, term, trunc, info


class WindDriftEnv(PaddedFixedEnv):
    """Models systematic wind drift: UAV position is displaced each step by a fixed
    bias vector whose direction is fixed at episode start (random angle) and whose
    magnitude is wind_drift grid-units/step.  The observation reports the drifted
    position, and the underlying environment state is also shifted so the physics
    (RSSI, path-loss) reflect the true displaced location."""
    def __init__(self, fixed_positions, wind_drift=0.0, **kwargs):
        self.wind_drift = wind_drift
        self._drift_dx = 0.0
        self._drift_dy = 0.0
        super().__init__(fixed_positions, **kwargs)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        if self.wind_drift > 0:
            angle = np.random.uniform(0, 2 * np.pi)
            self._drift_dx = self.wind_drift * np.cos(angle)
            self._drift_dy = self.wind_drift * np.sin(angle)
        return obs, info

    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        if self.wind_drift > 0:
            # shift UAV position in underlying env state
            self.uav.position[0] = float(np.clip(
                self.uav.position[0] + self._drift_dx, 0, self.grid_size[0] - 1))
            self.uav.position[1] = float(np.clip(
                self.uav.position[1] + self._drift_dy, 0, self.grid_size[1] - 1))
            # reflect drift in observation (indices 0,1 = UAV x,y)
            obs = obs.copy()
            obs[0] = self.uav.position[0]
            obs[1] = self.uav.position[1]
        return obs, r, term, trunc, info


class LatencyEnv(PaddedFixedEnv):
    """Models communication/processing latency: the agent receives an observation
    delayed by `latency_steps` timesteps.  Stale observations are zero-padded
    for the first latency_steps steps of each episode."""
    def __init__(self, fixed_positions, latency_steps=0, **kwargs):
        self.latency_steps = latency_steps
        self._obs_buffer = []
        super().__init__(fixed_positions, **kwargs)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._obs_buffer = [obs.copy() for _ in range(self.latency_steps)]
        return obs, info

    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        if self.latency_steps == 0:
            return obs, r, term, trunc, info
        self._obs_buffer.append(obs.copy())
        stale_obs = self._obs_buffer.pop(0)
        return stale_obs, r, term, trunc, info


# ==================== EPISODE RUNNERS ====================

def run_dqn_episode(model, env_class, env_kwargs, fixed_positions, seed, **extra_kwargs):
    env     = env_class(fixed_positions, **env_kwargs, **extra_kwargs)
    vec_env = VecFrameStack(DummyVecEnv([lambda: env]), n_stack=N_STACK)
    base    = _unwrap_base_env(vec_env)

    obs        = vec_env.reset()
    cum_reward = 0.0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done_arr, _ = vec_env.step(action)
        cum_reward += float(reward[0])
        if bool(done_arr[0]):
            break

    energy_used = EVAL_BATTERY - base.uav.battery
    return {
        "reward":     cum_reward,
        "jains":      compute_jains(base),
        "efficiency": float(base.total_data_collected / energy_used) if energy_used > 0 else 0.0,
        "coverage":   len(base.sensors_visited) / base.num_sensors * 100,
    }


def run_greedy_episode(agent_class, env_kwargs, fixed_positions, seed):
    env    = FixedLayoutEnv(fixed_positions, **env_kwargs)
    obs, _ = env.reset(seed=seed)
    agent  = agent_class(env)
    cum_reward = 0.0
    while True:
        action         = agent.select_action(obs)
        obs, r, done, trunc, _ = env.step(action)
        cum_reward    += r
        if done or trunc:
            break
    energy_used = EVAL_BATTERY - env.uav.battery
    return {
        "reward":     cum_reward,
        "jains":      compute_jains(env),
        "efficiency": float(env.total_data_collected / energy_used) if energy_used > 0 else 0.0,
        "coverage":   len(env.sensors_visited) / env.num_sensors * 100,
    }


# ==================== SWEEP RUNNERS ====================

def sweep_gps_noise(model):
    print("\n=== GPS Noise Sweep ===")
    records = []
    for sigma in GPS_SIGMAS:
        print(f"  sigma={sigma}")
        for seed in SEEDS:
            pos = get_positions(seed, BASE_ENV_KWARGS)
            r   = run_dqn_episode(model, GPSNoisyEnv, BASE_ENV_KWARGS, pos, seed,
                                  gps_sigma=sigma)
            records.append({"perturbation": "gps", "sigma": sigma, "seed": seed, **r})
            # baseline greedy (no GPS noise — greedy uses true position)
            for name, cls in [("SF-Aware Greedy", MaxThroughputGreedyV2),
                               ("Nearest Greedy",  NearestSensorGreedy)]:
                gr = run_greedy_episode(cls, BASE_ENV_KWARGS, pos, seed)
                records.append({"perturbation": "gps_baseline", "sigma": sigma,
                                 "seed": seed, "agent": name, **gr})
    return records


def sweep_pathloss(model):
    print("\n=== Path-Loss Mismatch Sweep ===")
    records = []
    for delta in PATHLOSS_DELTAS:
        print(f"  delta={delta:+.1f}")
        kw = {**BASE_ENV_KWARGS, "path_loss_exponent": 3.8 + delta}
        for seed in SEEDS:
            pos = get_positions(seed, kw)
            r   = run_dqn_episode(model, PaddedFixedEnv, kw, pos, seed)
            records.append({"perturbation": "pathloss", "delta": delta, "seed": seed, **r})
    return records


def sweep_dutycycle(model):
    print("\n=== Duty-Cycle Variation Sweep ===")
    records = []
    for dc in DUTY_CYCLES:
        print(f"  duty_cycle={dc}%")
        kw = {**BASE_ENV_KWARGS, "sensor_duty_cycle": float(dc)}
        for seed in SEEDS:
            pos = get_positions(seed, kw)
            r   = run_dqn_episode(model, PaddedFixedEnv, kw, pos, seed)
            records.append({"perturbation": "dutycycle", "duty_cycle": dc,
                             "seed": seed, **r})
    return records


def sweep_wind_drift(model):
    print("\n=== Wind Drift Sweep ===")
    records = []
    for drift in WIND_DRIFTS:
        print(f"  wind_drift={drift} grid-units/step")
        for seed in SEEDS:
            pos = get_positions(seed, BASE_ENV_KWARGS)
            r   = run_dqn_episode(model, WindDriftEnv, BASE_ENV_KWARGS, pos, seed,
                                  wind_drift=drift)
            records.append({"perturbation": "wind", "wind_drift": drift,
                             "seed": seed, **r})
    return records


def sweep_latency(model):
    print("\n=== Observation Latency Sweep ===")
    records = []
    for lat in LATENCY_STEPS:
        print(f"  latency={lat} steps")
        for seed in SEEDS:
            pos = get_positions(seed, BASE_ENV_KWARGS)
            r   = run_dqn_episode(model, LatencyEnv, BASE_ENV_KWARGS, pos, seed,
                                  latency_steps=lat)
            records.append({"perturbation": "latency", "latency_steps": lat,
                             "seed": seed, **r})
    return records


# ==================== PLOTTING ====================

COLORS = {"DQN": "#1b9e77", "SF-Aware Greedy": "#d95f02", "Nearest Greedy": "#7570b3"}


def _mean_std(df, x_col):
    g = df.groupby(x_col)["reward"]
    return g.mean(), g.std()


def plot_gps(df):
    dqn = df[df["perturbation"] == "gps"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, metric, label in [
        (axes[0], "reward", "Mean Cumulative Reward"),
        (axes[1], "jains",  "Mean Jain's Fairness Index"),
    ]:
        g      = dqn.groupby("sigma")[metric]
        means  = g.mean()
        stds   = g.std()
        base   = means.iloc[0]
        pct    = (means / base - 1) * 100
        ax.errorbar(GPS_SIGMAS, means.values, yerr=stds.values,
                    marker="o", color=COLORS["DQN"], linewidth=2, capsize=4,
                    label="DQN")
        ax.set_xlabel("GPS Position Noise σ (grid units)", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_xticks(GPS_SIGMAS)
        ax.set_xticklabels([f"σ={s}\n({s*10} m)" for s in GPS_SIGMAS])
        ax.grid(True, alpha=0.3)

    # secondary y-axis showing % degradation
    ax2 = axes[0].twinx()
    g   = dqn.groupby("sigma")["reward"]
    pct = (g.mean() / g.mean().iloc[0] - 1) * 100
    ax2.plot(GPS_SIGMAS, pct.values, "--", color="grey", alpha=0.6, label="% change")
    ax2.set_ylabel("% change from baseline", fontsize=9, color="grey")
    ax2.axhline(0, color="grey", linestyle=":", alpha=0.4)

    fig.suptitle("DQN Robustness to GPS Position Noise", fontsize=13, fontweight="bold")
    fig.tight_layout()
    ieee_style.clean_figure(fig)
    ieee_style.save(fig, str(OUTPUT_DIR / "fig_sim_to_real_gps"))
    plt.close(fig)
    print("  Saved fig_sim_to_real_gps.png")


def plot_pathloss(df):
    dqn = df[df["perturbation"] == "pathloss"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x_labels = [f"+{d:.1f}" for d in PATHLOSS_DELTAS]

    for ax, metric, label in [
        (axes[0], "reward", "Mean Cumulative Reward"),
        (axes[1], "jains",  "Mean Jain's Fairness Index"),
    ]:
        g     = dqn.groupby("delta")[metric]
        means = g.mean()
        stds  = g.std()
        ax.errorbar(PATHLOSS_DELTAS, means.values, yerr=stds.values,
                    marker="s", color="#7570b3", linewidth=2, capsize=4)
        ax.set_xlabel("Path-Loss Exponent Shift Δn (training: n=3.8)", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_xticks(PATHLOSS_DELTAS)
        ax.set_xticklabels(x_labels)
        ax.grid(True, alpha=0.3)

    fig.suptitle("DQN Robustness to Channel Model Mismatch (Path-Loss Exponent Shift)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    ieee_style.clean_figure(fig)
    ieee_style.save(fig, str(OUTPUT_DIR / "fig_sim_to_real_pathloss"))
    plt.close(fig)
    print("  Saved fig_sim_to_real_pathloss.png")


def plot_dutycycle(df):
    dqn = df[df["perturbation"] == "dutycycle"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, metric, label in [
        (axes[0], "reward", "Mean Cumulative Reward"),
        (axes[1], "jains",  "Mean Jain's Fairness Index"),
    ]:
        g     = dqn.groupby("duty_cycle")[metric]
        means = g.mean()
        stds  = g.std()
        xs    = DUTY_CYCLES[::-1]  # ascending: 4,6,8,10
        ax.errorbar(DUTY_CYCLES, means.reindex(DUTY_CYCLES).values,
                    yerr=stds.reindex(DUTY_CYCLES).values,
                    marker="^", color="#66a61e", linewidth=2, capsize=4)
        ax.set_xlabel("Sensor Duty Cycle (%)", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_xticks(DUTY_CYCLES)
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3)

    fig.suptitle("DQN Robustness to Duty-Cycle Tightening", fontsize=13, fontweight="bold")
    fig.tight_layout()
    ieee_style.clean_figure(fig)
    ieee_style.save(fig, str(OUTPUT_DIR / "fig_sim_to_real_dutycycle"))
    plt.close(fig)
    print("  Saved fig_sim_to_real_dutycycle.png")


def plot_summary(df_all):
    """Single 3-panel summary figure for the report."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    panels = [
        ("gps",       "sigma",      GPS_SIGMAS,
         [f"{s}\n({s*10} m)" for s in GPS_SIGMAS],
         "GPS σ (grid units / m)", "#1b9e77"),
        ("pathloss",  "delta",      PATHLOSS_DELTAS,
         [f"+{d:.1f}" for d in PATHLOSS_DELTAS],
         "Path-loss shift Δn",     "#7570b3"),
        ("dutycycle", "duty_cycle", DUTY_CYCLES,
         [f"{d}%" for d in DUTY_CYCLES],
         "Duty cycle (%)",         "#66a61e"),
    ]

    for ax, (pert, xcol, xvals, xlabs, xlabel, color) in zip(axes, panels):
        sub   = df_all[df_all["perturbation"] == pert]
        g     = sub.groupby(xcol)["reward"]
        means = g.mean().reindex(xvals)
        stds  = g.std().reindex(xvals)
        base  = means.iloc[0]
        pct   = (means / base - 1) * 100

        ax.errorbar(range(len(xvals)), means.values, yerr=stds.values,
                    marker="o", color=color, linewidth=2, capsize=4)
        ax.set_xticks(range(len(xvals)))
        ax.set_xticklabels(xlabs, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Mean Cumulative Reward", fontsize=10)
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(range(len(xvals)), pct.values, "--", color="grey", alpha=0.5)
        ax2.set_ylabel("% vs. baseline", fontsize=8, color="grey")
        ax2.axhline(0, color="grey", linestyle=":", alpha=0.3)

    titles = ["(a) GPS Noise", "(b) Path-Loss Mismatch", "(c) Duty-Cycle Tightening"]
    for ax, t in zip(axes, titles):
        ax.set_title(t, fontsize=11, fontweight="bold")

    fig.suptitle("Sim-to-Real Robustness: DQN Reward Degradation Under Domain Shift\n"
                 "(N=20, 500×500, 5 seeds per condition)", fontsize=12)
    fig.tight_layout()
    ieee_style.clean_figure(fig)
    ieee_style.save(fig, str(OUTPUT_DIR / "fig_sim_to_real_summary"))
    plt.close(fig)
    print("  Saved fig_sim_to_real_summary.png")


def plot_hil_summary(df_all):
    """Two-panel summary figure for the HIL preparation results (wind + latency)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    panels = [
        ("wind",    "wind_drift",    WIND_DRIFTS,
         [f"{d}\n({d*10:.0f} m/s)" for d in WIND_DRIFTS],
         "Wind Drift Magnitude (grid units/step)", "#d95f02"),
        ("latency", "latency_steps", LATENCY_STEPS,
         [f"{l} step{'s' if l != 1 else ''}\n({l*0.1:.1f} s)" for l in LATENCY_STEPS],
         "Observation Latency (timesteps / seconds)", "#e6ab02"),
    ]

    for ax, (pert, xcol, xvals, xlabs, xlabel, color) in zip(axes, panels):
        sub   = df_all[df_all["perturbation"] == pert]
        if sub.empty:
            ax.set_title(f"No data for {pert}")
            continue
        g     = sub.groupby(xcol)["reward"]
        means = g.mean().reindex(xvals)
        stds  = g.std().reindex(xvals)
        base  = means.iloc[0]
        pct   = (means / base - 1) * 100

        ax.errorbar(range(len(xvals)), means.values, yerr=stds.values,
                    marker="D", color=color, linewidth=2, capsize=5, markersize=7)
        ax.set_xticks(range(len(xvals)))
        ax.set_xticklabels(xlabs, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Mean Cumulative Reward", fontsize=11)
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(range(len(xvals)), pct.values, "--", color="grey", alpha=0.6,
                 linewidth=1.5)
        ax2.set_ylabel("% vs. baseline", fontsize=9, color="grey")
        ax2.axhline(0, color="grey", linestyle=":", alpha=0.4)
        ax2.tick_params(labelcolor="grey")

    axes[0].set_title("(d) Systematic Wind Drift", fontsize=12, fontweight="bold")
    axes[1].set_title("(e) Observation Latency", fontsize=12, fontweight="bold")

    fig.suptitle("HIL Preparation Analysis: Additional Domain-Shift Robustness\n"
                 "(N=20, 500×500, 5 seeds per condition)", fontsize=12)
    fig.tight_layout()
    ieee_style.clean_figure(fig)
    ieee_style.save(fig, str(OUTPUT_DIR / "fig_hil_preparation"))
    plt.close(fig)
    print("  Saved fig_hil_preparation.png")


# ==================== MAIN ====================

def main():
    print("Loading DQN model...")
    model = load_model()

    all_records = []
    all_records.extend(sweep_gps_noise(model))
    all_records.extend(sweep_pathloss(model))
    all_records.extend(sweep_dutycycle(model))
    all_records.extend(sweep_wind_drift(model))
    all_records.extend(sweep_latency(model))

    df = pd.DataFrame(all_records)
    df.to_csv(OUTPUT_DIR / "sim_to_real_results.csv", index=False)
    print(f"\nResults saved: {OUTPUT_DIR / 'sim_to_real_results.csv'}")

    # --- summary statistics ---
    print("\n=== GPS NOISE SUMMARY ===")
    gps = df[df["perturbation"] == "gps"].groupby("sigma")[["reward", "jains"]].agg(["mean", "std"])
    print(gps.to_string())
    base_r = gps["reward"]["mean"].iloc[0]
    for sigma, row in gps.iterrows():
        print(f"  sigma={sigma}: reward {row['reward']['mean']:.0f} "
              f"({(row['reward']['mean']/base_r-1)*100:+.1f}%), "
              f"Jain's {row['jains']['mean']:.4f}")

    print("\n=== PATH-LOSS MISMATCH SUMMARY ===")
    pl = df[df["perturbation"] == "pathloss"].groupby("delta")[["reward", "jains"]].agg(["mean", "std"])
    base_r = pl["reward"]["mean"].iloc[0]
    for delta, row in pl.iterrows():
        print(f"  delta={delta:+.1f}: reward {row['reward']['mean']:.0f} "
              f"({(row['reward']['mean']/base_r-1)*100:+.1f}%), "
              f"Jain's {row['jains']['mean']:.4f}")

    print("\n=== DUTY-CYCLE SUMMARY ===")
    dc = df[df["perturbation"] == "dutycycle"].groupby("duty_cycle")[["reward", "jains"]].agg(["mean", "std"])
    base_r = dc["reward"]["mean"].reindex([10]).iloc[0]
    for duty, row in dc.iterrows():
        print(f"  duty_cycle={duty}%: reward {row['reward']['mean']:.0f} "
              f"({(row['reward']['mean']/base_r-1)*100:+.1f}%), "
              f"Jain's {row['jains']['mean']:.4f}")

    # --- summary statistics for new perturbations ---
    if not df[df["perturbation"] == "wind"].empty:
        print("\n=== WIND DRIFT SUMMARY ===")
        wd = df[df["perturbation"] == "wind"].groupby("wind_drift")[["reward", "jains"]].agg(["mean", "std"])
        base_r = wd["reward"]["mean"].iloc[0]
        for drift, row in wd.iterrows():
            print(f"  wind_drift={drift}: reward {row['reward']['mean']:.0f} "
                  f"({(row['reward']['mean']/base_r-1)*100:+.1f}%), "
                  f"Jain's {row['jains']['mean']:.4f}")

    if not df[df["perturbation"] == "latency"].empty:
        print("\n=== OBSERVATION LATENCY SUMMARY ===")
        lt = df[df["perturbation"] == "latency"].groupby("latency_steps")[["reward", "jains"]].agg(["mean", "std"])
        base_r = lt["reward"]["mean"].iloc[0]
        for lat, row in lt.iterrows():
            print(f"  latency={lat} steps: reward {row['reward']['mean']:.0f} "
                  f"({(row['reward']['mean']/base_r-1)*100:+.1f}%), "
                  f"Jain's {row['jains']['mean']:.4f}")

    # --- plots ---
    print("\nGenerating plots...")
    plot_gps(df)
    plot_pathloss(df)
    plot_dutycycle(df)
    plot_summary(df)
    plot_hil_summary(df)
    print("\nDone.")


if __name__ == "__main__":
    main()
