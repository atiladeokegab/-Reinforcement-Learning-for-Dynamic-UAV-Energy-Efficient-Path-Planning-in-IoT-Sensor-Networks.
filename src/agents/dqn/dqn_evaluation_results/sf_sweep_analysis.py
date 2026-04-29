"""
sf_sweep_analysis.py
====================
Analyse how DQN, Relational RL, and TSP Oracle affect the LoRaWAN
Spreading Factor (SF) distribution across the five sweep conditions.

For each (agent, condition, seed) we record:
  - fraction of sensors at each SF at episode end
  - SF monoculture index  (Herfindahl-Hirschman: sum of fractions^2; 1.0 = total monoculture)
  - SF entropy (diversity; 0 = monoculture, log2(4) ≈ 2 = perfectly spread)
  - SF changes per sensor (how many times ADR fired during the episode)

The analysis answers: does the Relational RL agent's interior exploration
produce a more diverse SF distribution than the DQN perimeter patrol?

Usage:
    PYTHONIOENCODING=utf-8 uv run python \\
        src/agents/dqn/dqn_evaluation_results/sf_sweep_analysis.py
"""

from __future__ import annotations

import sys
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import gymnasium

# ---------------------------------------------------------------------------
# Path setup (mirrors sweep_eval.py)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_DQN  = _HERE.parent
_SRC  = _HERE.parents[2]
_ROOT = _HERE.parents[3]

for _p in (str(_SRC), str(_DQN), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

for _sub in ("relational_policy",):
    _sp = str(_SRC / "experiments" / _sub)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

for _gnn in _DQN.rglob("gnn_extractor.py"):
    _gd = str(_gnn.parent)
    if _gd not in sys.path:
        sys.path.insert(0, _gd)

from environment.uav_env import UAVEnvironment
from greedy_agents import TSPOracleAgent
from relational_rl_runner import InferenceRelationalUAVEnv, load_relational_rl_module

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIGS = [
    {"grid_size": (100, 100), "num_sensors": 10},
    {"grid_size": (200, 200), "num_sensors": 20},
    {"grid_size": (300, 300), "num_sensors": 30},
    {"grid_size": (400, 400), "num_sensors": 40},
    {"grid_size": (500, 500), "num_sensors": 50},
]
N_SEEDS = 10
SEEDS   = list(range(200, 200 + N_SEEDS))

ENV_KWARGS: dict[str, Any] = {
    "max_steps":          2100,
    "path_loss_exponent": 3.8,
    "rssi_threshold":     -85.0,
    "sensor_duty_cycle":  10.0,
}

DQN_MODEL_PATH  = _DQN / "models" / "dqn_v3_retrain" / "dqn_final.zip"
DQN_MAX_SENSORS = 50

REL_CKPT_DIR = (
    _DQN / "models" / "relational_rl" /
    "results" / "checkpoints" / "stage_4" / "final"
)

OUTPUT_DIR = _HERE / "baseline_results" / "sf_sweep"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUTPUT_DIR / "sf_sweep_results.csv"

SFS = [7, 9, 11, 12]


# ---------------------------------------------------------------------------
# SF metric helpers
# ---------------------------------------------------------------------------

def _sf_metrics(sensors: list) -> dict:
    """Compute SF distribution metrics from a list of IoTSensor objects."""
    final_sfs = [s.spreading_factor for s in sensors]
    n = len(final_sfs)
    counts = {sf: final_sfs.count(sf) for sf in SFS}
    fracs  = {sf: counts[sf] / n for sf in SFS}

    # Monoculture index: HHI (1 = all same SF, 0.25 = perfectly spread over 4 SFs)
    hhi = sum(f**2 for f in fracs.values())

    # Entropy (bits, 0 = monoculture, log2(4)=2 = perfectly spread)
    entropy = -sum(f * np.log2(f) if f > 0 else 0.0 for f in fracs.values())

    # Mean SF changes per sensor (ADR fires)
    sf_changes = [s.get_sf_changes() for s in sensors]

    return {
        "frac_sf7":  fracs[7],
        "frac_sf9":  fracs[9],
        "frac_sf11": fracs[11],
        "frac_sf12": fracs[12],
        "hhi":       hhi,
        "entropy":   entropy,
        "mean_sf_changes": float(np.mean(sf_changes)),
        "mean_final_sf":   float(np.mean(final_sfs)),
    }


# ---------------------------------------------------------------------------
# DQN env wrapper (zero-padded observations)
# ---------------------------------------------------------------------------

class _DQNEnv(UAVEnvironment):
    def __init__(self, max_sensors_limit: int = DQN_MAX_SENSORS, **kw):
        self.max_sensors_limit = max_sensors_limit
        self._snap: dict | None = None
        super().__init__(**kw)
        raw = self.observation_space.shape[0]
        self._fps = 0
        for uav_f in range(raw + 1):
            rem = raw - uav_f
            if rem > 0 and rem % self.num_sensors == 0:
                self._fps = rem // self.num_sensors
                break
        padded = raw + (self.max_sensors_limit - self.num_sensors) * self._fps
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(padded,), dtype=np.float32)

    def _pad(self, obs):
        pad = np.zeros(
            (self.max_sensors_limit - self.num_sensors) * self._fps, dtype=np.float32)
        return np.concatenate([obs, pad]).astype(np.float32)

    def reset(self, **kw):
        if hasattr(self, "sensors") and self.current_step > 0:
            self._snap = {
                "sensors":        list(self.sensors),
                "sensors_visited_count": len(self.sensors_visited),
            }
        obs, info = super().reset(**kw)
        return self._pad(obs), info

    def step(self, action):
        obs, r, te, tr, info = super().step(action)
        return self._pad(obs), r, te, tr, info


# ---------------------------------------------------------------------------
# Episode runners
# ---------------------------------------------------------------------------

def _run_dqn(model, env: _DQNEnv, seed: int) -> dict:
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
    def make_e():
        e2 = _DQNEnv(
            max_sensors_limit=DQN_MAX_SENSORS,
            grid_size=env.grid_size,
            num_sensors=env.num_sensors,
            **ENV_KWARGS,
        )
        return e2
    venv  = DummyVecEnv([make_e])
    fenv  = VecFrameStack(venv, n_stack=4)
    obs   = fenv.reset()
    done  = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = fenv.step(action)
        done = dones[0]
    # DummyVecEnv auto-resets on done; read state from pre-reset snapshot
    raw_env = fenv.venv.envs[0]
    snap = raw_env._snap
    fenv.close()
    if snap is None:
        return {}
    m = _sf_metrics(snap["sensors"])
    m["ndr"] = snap["sensors_visited_count"] / raw_env.num_sensors * 100.0
    return m


def _run_relational(module, seed: int, grid_size, num_sensors) -> dict:
    import torch
    from ray.rllib.utils.typing import TensorStructType
    try:
        from ray.rllib.core.columns import Columns
    except ImportError:
        from ray.rllib.policy.sample_batch import SampleBatch as Columns

    env = InferenceRelationalUAVEnv(
        n_max=DQN_MAX_SENSORS,
        grid_size=grid_size,
        num_sensors=num_sensors,
        **ENV_KWARGS,
    )
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        batch = {Columns.OBS: {
            k: torch.as_tensor(np.asarray(v), dtype=torch.float32).unsqueeze(0)
            for k, v in obs.items()
        }}
        with torch.no_grad():
            out = module._forward_inference(batch)
        action = int(torch.argmax(out[Columns.ACTION_DIST_INPUTS], dim=-1).item())
        obs, _, te, tr, _ = env.step(action)
        done = te or tr
    m = _sf_metrics(env.sensors)
    m["ndr"] = len(env.sensors_visited) / env.num_sensors * 100.0
    return m


def _run_tsp(seed: int, grid_size, num_sensors) -> dict:
    env = UAVEnvironment(
        grid_size=grid_size,
        num_sensors=num_sensors,
        **ENV_KWARGS,
    )
    obs, info = env.reset(seed=seed)
    agent = TSPOracleAgent(env)
    done = False
    while not done:
        action = agent.select_action(obs)
        obs, _, te, tr, info = env.step(action)
        done = te or tr
    m = _sf_metrics(env.sensors)
    m["ndr"] = len(env.sensors_visited) / env.num_sensors * 100.0
    return m


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep() -> pd.DataFrame:
    # Load DQN
    from stable_baselines3 import DQN as SB3DQN
    log.info("Loading DQN from %s", DQN_MODEL_PATH)
    dqn_model = SB3DQN.load(str(DQN_MODEL_PATH), device="cpu")

    # Load Relational RL
    log.info("Loading Relational RL from %s", REL_CKPT_DIR)
    rel_module = load_relational_rl_module(str(REL_CKPT_DIR))

    rows = []
    for cfg in CONFIGS:
        grid_size  = cfg["grid_size"]
        n_sensors  = cfg["num_sensors"]
        label      = f"{grid_size[0]}x{grid_size[1]}/N={n_sensors}"
        log.info("=== %s ===", label)

        for seed in SEEDS:
            log.info("  seed=%d", seed)

            # DQN
            dqn_env = _DQNEnv(
                max_sensors_limit=DQN_MAX_SENSORS,
                grid_size=grid_size,
                num_sensors=n_sensors,
                **ENV_KWARGS,
            )
            dqn_env.reset(seed=seed)
            m_dqn = _run_dqn(dqn_model, dqn_env, seed)
            rows.append({"agent": "DQN", "condition": label, "seed": seed,
                         "grid": grid_size[0], "n_sensors": n_sensors, **m_dqn})

            # Relational RL
            m_rel = _run_relational(rel_module, seed, grid_size, n_sensors)
            rows.append({"agent": "Relational RL", "condition": label, "seed": seed,
                         "grid": grid_size[0], "n_sensors": n_sensors, **m_rel})

            # TSP Oracle
            m_tsp = _run_tsp(seed, grid_size, n_sensors)
            rows.append({"agent": "TSP Oracle", "condition": label, "seed": seed,
                         "grid": grid_size[0], "n_sensors": n_sensors, **m_tsp})

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    log.info("Results saved to %s", CSV_PATH)
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

AGENT_COLORS = {
    "DQN":          "#C00000",
    "Relational RL": "#7030A0",
    "TSP Oracle":    "#2E75B6",
}
AGENT_MARKERS = {"DQN": "o", "Relational RL": "s", "TSP Oracle": "^"}
CONDITIONS = [f"{c['grid_size'][0]}x{c['grid_size'][0]}/N={c['num_sensors']}"
              for c in CONFIGS]


def plot_sf_analysis(df: pd.DataFrame) -> None:

    summary = (
        df.groupby(["agent", "condition"])[
            ["frac_sf7", "frac_sf9", "frac_sf11", "frac_sf12",
             "hhi", "entropy", "mean_sf_changes", "ndr"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )

    # ── Figure 1: SF distribution stacked bars per condition ────────────────
    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(14, 4), sharey=True)
    sf_cols   = ["frac_sf7", "frac_sf9", "frac_sf11", "frac_sf12"]
    sf_labels = ["SF7", "SF9", "SF11", "SF12"]
    sf_colors = ["#2E75B6", "#70AD47", "#FFC000", "#C00000"]
    agents    = ["DQN", "Relational RL", "TSP Oracle"]
    x_pos     = np.arange(len(agents))
    bar_w     = 0.7

    for ax, cond in zip(axes, CONDITIONS):
        sub = df[df["condition"] == cond]
        means = {a: sub[sub["agent"] == a][sf_cols].mean() for a in agents}
        bottoms = np.zeros(len(agents))
        for sf_col, sf_lbl, sf_col_hex in zip(sf_cols, sf_labels, sf_colors):
            heights = np.array([means[a][sf_col] for a in agents])
            ax.bar(x_pos, heights, bar_w, bottom=bottoms,
                   color=sf_col_hex, label=sf_lbl if cond == CONDITIONS[0] else "")
            bottoms += heights
        ax.set_title(cond, fontsize=7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(["DQN", "Rel.\nRL", "TSP"], fontsize=7)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.set_ylim(0, 1.05)

    axes[0].set_ylabel("Fraction of sensors")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in sf_colors]
    fig.legend(handles, sf_labels, loc="upper right", fontsize=8,
               title="SF at episode end", frameon=True)
    fig.suptitle("Final SF Distribution by Agent and Condition", fontsize=10)
    fig.tight_layout()
    out = OUTPUT_DIR / "sf_distribution_stacked.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out)

    # ── Figure 2: HHI monoculture index + entropy across conditions ─────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(CONDITIONS))

    for ax, metric, ylabel, title in [
        (ax1, "hhi",     "HHI monoculture index (1 = total monoculture)", "SF Monoculture Index (HHI)"),
        (ax2, "entropy", "SF entropy (bits; 2 = max diversity)",           "SF Diversity (Entropy)"),
    ]:
        for agent in agents:
            sub = df[df["agent"] == agent].groupby("condition")[metric]
            means = [sub.get_group(c).mean() if c in sub.groups else np.nan for c in CONDITIONS]
            stds  = [sub.get_group(c).std()  if c in sub.groups else 0.0    for c in CONDITIONS]
            ax.errorbar(x, means, yerr=stds, label=agent,
                        color=AGENT_COLORS[agent],
                        marker=AGENT_MARKERS[agent],
                        linewidth=1.6, markersize=6, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(CONDITIONS, rotation=20, ha="right", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = OUTPUT_DIR / "sf_monoculture_entropy.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out)

    # ── Figure 3: SF12 fraction vs NDR scatter (per-seed) ───────────────────
    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(14, 3.5), sharey=True)
    for ax, cond in zip(axes, CONDITIONS):
        sub = df[df["condition"] == cond]
        for agent in agents:
            a_sub = sub[sub["agent"] == agent]
            ax.scatter(a_sub["frac_sf12"], a_sub["ndr"],
                       color=AGENT_COLORS[agent], marker=AGENT_MARKERS[agent],
                       alpha=0.65, s=22, label=agent if cond == CONDITIONS[0] else "")
        ax.set_title(cond, fontsize=7)
        ax.set_xlabel("SF12 fraction", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    axes[0].set_ylabel("NDR (%)", fontsize=8)
    handles = [plt.Line2D([0], [0], marker=AGENT_MARKERS[a], color=AGENT_COLORS[a],
                           linestyle="None", markersize=7) for a in agents]
    fig.legend(handles, agents, loc="upper right", fontsize=8, frameon=True)
    fig.suptitle("SF12 Monoculture vs Sensor Coverage (NDR) — per seed", fontsize=10)
    fig.tight_layout()
    out = OUTPUT_DIR / "sf12_vs_ndr_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out)

    # ── Figure 4: mean ADR change rate per condition ─────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    for agent in agents:
        sub = df[df["agent"] == agent].groupby("condition")["mean_sf_changes"]
        means = [sub.get_group(c).mean() if c in sub.groups else np.nan for c in CONDITIONS]
        stds  = [sub.get_group(c).std()  if c in sub.groups else 0.0    for c in CONDITIONS]
        ax.errorbar(x, means, yerr=stds, label=agent,
                    color=AGENT_COLORS[agent],
                    marker=AGENT_MARKERS[agent],
                    linewidth=1.6, markersize=6, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(CONDITIONS, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Mean SF changes per sensor (ADR fires)")
    ax.set_title("ADR Activity: SF Changes per Sensor per Episode")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = OUTPUT_DIR / "sf_adr_changes.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out)

    # ── Print summary table ──────────────────────────────────────────────────
    print("\n=== SF Sweep Summary ===")
    for cond in CONDITIONS:
        print(f"\n{cond}:")
        for agent in agents:
            sub = df[(df["agent"] == agent) & (df["condition"] == cond)]
            if sub.empty:
                continue
            print(f"  {agent:15s}  "
                  f"SF7={sub['frac_sf7'].mean():.2f}  "
                  f"SF9={sub['frac_sf9'].mean():.2f}  "
                  f"SF11={sub['frac_sf11'].mean():.2f}  "
                  f"SF12={sub['frac_sf12'].mean():.2f}  "
                  f"HHI={sub['hhi'].mean():.3f}  "
                  f"Entropy={sub['entropy'].mean():.2f}  "
                  f"NDR={sub['ndr'].mean():.1f}%")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if CSV_PATH.exists():
        log.info("Loading existing results from %s", CSV_PATH)
        df = pd.read_csv(CSV_PATH)
    else:
        df = run_sweep()
    plot_sf_analysis(df)
    log.info("Done. Figures in %s", OUTPUT_DIR)
