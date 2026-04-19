"""
Ablation Study — DQN Protocol-Aware UAV Path Planning
======================================================
Evaluates the trained DQN policy under four ablated conditions to isolate
the contribution of each design component:

  A1 — No Capture Effect      : All intra-SF collisions -> mutual destruction
                                 (6 dB SIR threshold raised to infinity)
  A2 — Instant ADR            : EMA smoothing disabled (λ=1.0), ADR reacts
                                 immediately to each RSSI measurement
  A3 — No AoI Observation     : Urgency features zeroed in the observation
                                 vector — the agent cannot see sensor staleness
                                 (inference-time test of whether the policy
                                 relies on AoI signals to achieve fairness)
  A4 — No Domain Randomisation: Policy trained on fixed (500×500, N=20) only,
                                 without curriculum or grid-size randomisation.
                                 Requires running train_ablation_a4.py first.

A1–A3 use the main trained model (dqn_fairness_framestack/dqn_final.zip).
A4 uses the separately trained no-DR model (models/dqn_no_dr/dqn_final.zip).

Each condition is evaluated over 5 seeds on the 500×500, N=20 configuration
(the primary evaluation condition from Chapter 5).

Outputs (in ablation_results/):
  ablation_results.csv       — per-seed raw metrics
  ablation_summary.csv       — mean ± std per condition
  fig_ablation_bars.png      — bar chart (reward, Jain's, efficiency)
  fig_ablation_fairness.png  — Jain's Index per-seed scatter

Author: ATILADE GABRIEL OKE
"""

import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium
from pathlib import Path
import ieee_style
ieee_style.apply()
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# ==================== PATH SETUP ====================
script_dir = Path(__file__).resolve().parent
src_dir    = script_dir.parent.parent.parent
sys.path.insert(0, str(src_dir))

from environment.uav_env import UAVEnvironment

# ==================== CONFIGURATION ====================
SEEDS     = [42, 123, 256, 789, 1337, 2024, 999, 314, 555, 2048,
             100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
GRID_SIZE = (500, 500)
N_SENSORS = 20
MAX_STEPS         = 2100
N_STACK           = 4
MAX_SENSORS_LIMIT = 50   # must match training — observation is zero-padded to this

# Primary model (A1, A2, A3) — DQN-v3 domain-randomised model
_MAIN_MODEL_CANDIDATES = [
    script_dir.parent / "models" / "dqn_v4a" / "dqn_final.zip",
    script_dir.parent / "models" / "dqn_v3" / "dqn_final.zip",
    src_dir.parent / "models" / "dqn_domain_rand" / "dqn_final.zip",
]

# A4 model — trained by train_ablation_a4.py
A4_MODEL_PATH = script_dir.parent / "models" / "dqn_no_dr" / "dqn_final.zip"

OUTPUT_DIR = script_dir / "ablation_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EVAL_CONFIG = {
    "grid_size":          GRID_SIZE,
    "num_sensors":        N_SENSORS,
    "max_steps":          MAX_STEPS,
    "path_loss_exponent": 3.8,
    "rssi_threshold":     -85.0,
    "sensor_duty_cycle":  10.0,
}

# ==================== BASE ENVIRONMENT ====================

class FixedLayoutEnv(UAVEnvironment):
    """
    Forces identical sensor layout for a given seed across all conditions
    so per-condition differences reflect only the ablated component.

    Observation is zero-padded to MAX_SENSORS_LIMIT (same layout as the
    DomainRandEnv used during training) so the model's input size is matched.

    Also snapshots per-sensor data in reset() before the auto-reset wipes
    the state, enabling accurate Jain's index computation.
    """

    def __init__(self, seed: int, **kwargs):
        self._layout_seed     = seed
        self.last_sensor_data = None
        super().__init__(**kwargs)
        self._build_padded_obs_space()

    def _build_padded_obs_space(self):
        raw = self.observation_space.shape[0]
        self._fps = 0
        for uav_f in range(raw + 1):
            rem = raw - uav_f
            if rem > 0 and rem % self.num_sensors == 0:
                self._fps = rem // self.num_sensors
                break
        if self._fps == 0:
            raise ValueError(
                f"Cannot infer features_per_sensor: raw={raw}, "
                f"num_sensors={self.num_sensors}"
            )
        self._raw_obs_size = raw
        padded = raw + (MAX_SENSORS_LIMIT - self.num_sensors) * self._fps
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(padded,), dtype=np.float32
        )

    def _pad(self, obs: np.ndarray) -> np.ndarray:
        n_pad = (MAX_SENSORS_LIMIT - self.num_sensors) * self._fps
        return np.concatenate(
            [obs, np.zeros(n_pad, dtype=np.float32)]
        ).astype(np.float32)

    def reset(self, **kwargs):
        # Snapshot per-sensor data before parent resets it
        if hasattr(self, "sensors") and getattr(self, "current_step", 0) > 0:
            self.last_sensor_data = [
                {
                    "sensor_id":              int(s.sensor_id),
                    "total_data_generated":   float(s.total_data_generated),
                    "total_data_transmitted": float(s.total_data_transmitted),
                    "buffer_occupancy":       float(s.data_buffer / s.max_buffer_size),
                }
                for s in self.sensors
            ]
        obs, info = super().reset(**kwargs)
        return self._pad(obs), info

    def step(self, action):
        obs, reward, term, trunc, info = super().step(action)
        return self._pad(obs), reward, term, trunc, info

    def _generate_uniform_sensor_positions(self, num_sensors):
        rng = np.random.default_rng(self._layout_seed)
        x   = rng.uniform(0, self.grid_size[0], num_sensors)
        y   = rng.uniform(0, self.grid_size[1], num_sensors)
        return [(float(xi), float(yi)) for xi, yi in zip(x, y)]


# ==================== A1: NO CAPTURE EFFECT ====================

class NoCaptureEffectEnv(FixedLayoutEnv):
    """A1: All intra-SF collisions -> mutual destruction (no 6 dB SIR check)."""
    pass


_original_execute = UAVEnvironment._execute_collect_action


def _no_ce_execute(self, step_data_loss: float) -> float:
    """Replaces _execute_collect_action: capture threshold raised to infinity."""
    urgencies_before = self._get_sensor_urgencies()
    self.uav.hover(duration=self.collection_duration)
    battery_used = self.uav.battery_drain_hover * self.collection_duration

    transmission_attempts = {}
    for sensor in self.sensors:
        if sensor.data_buffer <= 0:
            continue
        sensor.update_spreading_factor(
            tuple(self.uav.position), current_step=self.current_step
        )
        P_link    = sensor.get_success_probability(
            tuple(self.uav.position), use_advanced_model=True
        )
        P_overall = P_link * sensor.duty_cycle_probability
        if P_overall > random.random():
            sf = sensor.spreading_factor
            transmission_attempts.setdefault(sf, []).append(sensor)

    # No capture effect: only solo transmissions succeed
    successful_sf_slots = {}
    collision_count     = 0
    for sf, slist in transmission_attempts.items():
        if len(slist) == 1:
            successful_sf_slots[sf] = slist[0]
        else:
            collision_count += len(slist) - 1   # all packets destroyed

    total_bytes_collected = 0.0
    new_sensors_collected = []
    attempted_empty       = False
    self.last_successful_collections = []

    for sf, winning_sensor in successful_sf_slots.items():
        bytes_collected, success = winning_sensor.collect_data(
            uav_position       = tuple(self.uav.position),
            collection_duration= self.collection_duration,
        )
        if success and bytes_collected > 0:
            total_bytes_collected      += bytes_collected
            self.total_data_collected  += bytes_collected
            if winning_sensor.sensor_id not in self.sensors_visited:
                new_sensors_collected.append(winning_sensor.sensor_id)
                self.sensors_visited.add(winning_sensor.sensor_id)
            self.last_successful_collections.append((winning_sensor, sf))

    in_range = [s for s in self.sensors if s.is_in_range(tuple(self.uav.position))]
    if in_range and total_bytes_collected == 0 and not transmission_attempts:
        attempted_empty = True

    self.last_step_bytes_collected = total_bytes_collected

    urgencies_after = self._get_sensor_urgencies()
    urgency_reduced = float(
        np.sum(np.maximum(urgencies_before - urgencies_after, 0))
    )

    sensor_buffers = [s.data_buffer for s in self.sensors]
    all_collected  = len(self.sensors_visited) == self.num_sensors
    unvisited = (
        (self.num_sensors - len(self.sensors_visited))
        if (self.current_step >= self.max_steps or self.uav.get_battery_percentage() <= 0)
        else 0
    )

    return self.reward_fn.calculate_collection_reward(
        bytes_collected       = total_bytes_collected,
        was_new_sensor        = len(new_sensors_collected) > 0,
        was_empty             = attempted_empty,
        all_sensors_collected = all_collected,
        battery_used          = battery_used,
        num_sensors_collected = len(successful_sf_slots),
        collision_count       = collision_count,
        data_loss             = step_data_loss,
        urgency_reduced       = urgency_reduced,
        sensor_buffers        = sensor_buffers,
        unvisited_count       = unvisited,
    )


NoCaptureEffectEnv._execute_collect_action = _no_ce_execute


# ==================== A2: INSTANT ADR ====================

class InstantADREnv(FixedLayoutEnv):
    """A2: EMA smoothing disabled — ADR updates SF on every step instantly."""

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        for sensor in self.sensors:
            sensor.use_ema_adr = False
            sensor.avg_rssi    = None     # discard any stale EMA state
        return obs, info


# ==================== A3: NO AoI OBSERVATION ====================

class NoAoIObsEnv(FixedLayoutEnv):
    """
    A3: Urgency (AoI) features zeroed in the raw observation vector before
    padding.

    Observation layout (from UAVEnvironment._get_observation):
        [uav_x, uav_y, battery,
         buf_0, urgency_0, lq_0,
         buf_1, urgency_1, lq_1, ...]

    Urgency sits at positions 3 + 3*i + 1  for i in 0..N-1.
    Setting these to 0.0 makes the policy blind to sensor AoI/starvation,
    testing whether the learned policy relies on urgency signals to achieve
    fair coverage of distant/stale sensors.
    """

    def _get_observation(self) -> np.ndarray:
        # Call UAVEnvironment's _get_observation (not FixedLayoutEnv's step/reset
        # which calls _pad on top) to get the raw unpadded obs.
        obs = UAVEnvironment._get_observation(self)   # shape (3 + 3*N,)
        for i in range(self.num_sensors):
            obs[3 + 3 * i + 1] = 0.0   # zero urgency feature for sensor i
        return obs


# ==================== A4: NO DOMAIN RANDOMISATION ====================
# Evaluated with the separately trained dqn_no_dr/dqn_final.zip model.
# The environment class is plain FixedLayoutEnv — the ablation is in the
# model weights, not the environment.
# Run train_ablation_a4.py first to produce that model.

NoDomainRandEnv = FixedLayoutEnv   # alias — same env, different model


# ==================== METRICS ====================

def jains_fairness(collection_rates: np.ndarray) -> float:
    """Jain's Fairness Index over per-sensor collection rates."""
    if np.sum(collection_rates) == 0:
        return 0.0
    n = len(collection_rates)
    return float(
        (np.sum(collection_rates) ** 2) / (n * np.sum(collection_rates ** 2))
    )


# ==================== EPISODE RUNNER ====================

def _unwrap_base_env(vec):
    """Traverse VecFrameStack -> DummyVecEnv -> TimeLimit -> base env."""
    inner = vec
    while hasattr(inner, "venv"):
        inner = inner.venv
    base = inner.envs[0]
    while hasattr(base, "env"):
        base = base.env
    return base


def run_episode(env_class, seed: int, model, env_kwargs: dict) -> dict:
    """Run one deterministic episode and return a metrics dict."""
    np.random.seed(seed)
    random.seed(seed)

    env      = env_class(seed=seed, **env_kwargs)
    vec_env  = VecFrameStack(DummyVecEnv([lambda: env]), n_stack=N_STACK)
    base_env = _unwrap_base_env(vec_env)

    obs               = vec_env.reset()
    cumulative_reward = 0.0
    pre_data          = 0.0
    pre_battery       = base_env.uav.max_battery
    pre_coverage      = 0.0

    while True:
        # Capture state *before* the terminal step because DummyVecEnv
        # auto-resets the environment the moment done=True is returned.
        pre_data     = base_env.total_data_collected
        pre_battery  = base_env.uav.battery
        pre_coverage = len(base_env.sensors_visited) / base_env.num_sensors

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done_arr, _ = vec_env.step(action)
        cumulative_reward += float(reward[0])

        if bool(done_arr[0]):
            break

    # Per-sensor rates from the snapshot saved during auto-reset
    if base_env.last_sensor_data:
        rates = np.array([
            s["total_data_transmitted"] / max(s["total_data_generated"], 1e-9)
            for s in base_env.last_sensor_data
        ], dtype=np.float64)
    else:
        rates = np.zeros(env_kwargs.get("num_sensors", N_SENSORS), dtype=np.float64)

    battery_used = base_env.uav.max_battery - pre_battery
    energy_eff   = pre_data / max(battery_used, 1e-9)

    vec_env.close()

    min_rate  = float(np.min(rates)) if len(rates) > 0 else 0.0
    peak_aoi  = float(max(
        s["buffer_occupancy"] for s in base_env.last_sensor_data
    )) if base_env.last_sensor_data else 1.0

    return {
        "cumulative_reward": cumulative_reward,
        "jains_index":       jains_fairness(rates),
        "min_collection_rate": min_rate,
        "peak_aoi_proxy":    peak_aoi,
        "energy_efficiency": energy_eff,
        "coverage":          pre_coverage,
        "bytes_collected":   pre_data,
    }


# ==================== MODEL LOADER ====================

def _load_main_model() -> DQN:
    for path in _MAIN_MODEL_CANDIDATES:
        if path.exists():
            model = DQN.load(str(path))
            print(f"  Main model loaded from: {path}")
            return model
    raise FileNotFoundError(
        "Main DQN model not found. Checked:\n"
        + "\n".join(f"  {p}" for p in _MAIN_MODEL_CANDIDATES)
    )


def _load_a4_model():
    if A4_MODEL_PATH.exists():
        model = DQN.load(str(A4_MODEL_PATH))
        print(f"  A4 model loaded from: {A4_MODEL_PATH}")
        return model
    print(
        f"\n[A4 SKIPPED] Model not found at {A4_MODEL_PATH}\n"
        "  Run train_ablation_a4.py first, then re-run this script.\n"
    )
    return None


# ==================== CONDITIONS ====================
#
# Each entry: (label, env_class, model_key)
# model_key: "main" -> main DQN model   |   "a4" -> no-DR model
#
CONDITIONS = [
    ("Full Model",            FixedLayoutEnv,      "main"),
    ("No Capture Effect",     NoCaptureEffectEnv,  "main"),
    ("Instant ADR",           InstantADREnv,       "main"),
    ("No AoI Observation",    NoAoIObsEnv,         "main"),
    ("No Domain Rand. (A4)",  NoDomainRandEnv,     "a4"),
]

CONDITION_XLABELS = [
    "Full\nModel",
    "No\nCapture\nEffect",
    "Instant\nADR",
    "No AoI\nObs.",
    "No Domain\nRand.",
]

COLORS = ieee_style.ABLATION_COLORS


# ==================== MAIN ====================

def main():
    print("=" * 65)
    print("Ablation Study — DQN Protocol-Aware UAV")
    print("=" * 65)

    main_model = _load_main_model()
    a4_model   = _load_a4_model()

    models = {"main": main_model, "a4": a4_model}

    records      = []
    active_conds = []   # conditions that were actually evaluated

    for label, env_class, model_key in CONDITIONS:
        model = models[model_key]
        if model is None:
            continue   # A4 model not trained yet — skip gracefully

        print(f"\n{'='*65}")
        print(f"Condition: {label}")
        print(f"{'='*65}")
        active_conds.append(label)

        for seed in SEEDS:
            print(f"  Seed {seed}...", end=" ", flush=True)
            try:
                metrics             = run_episode(env_class, seed, model, EVAL_CONFIG)
                metrics["condition"] = label
                metrics["seed"]      = seed
                records.append(metrics)
                print(
                    f"reward={metrics['cumulative_reward']:.0f}  "
                    f"J={metrics['jains_index']:.4f}  "
                    f"min_rate={metrics['min_collection_rate']*100:.1f}%  "
                    f"peak_aoi={metrics['peak_aoi_proxy']:.3f}  "
                    f"eff={metrics['energy_efficiency']:.1f} B/Wh  "
                    f"NDR={metrics['coverage']*100:.0f}%"
                )
            except Exception as exc:
                import traceback
                print(f"ERROR: {exc}")
                traceback.print_exc()

    if not records:
        print("No results collected — exiting.")
        return

    # ---- Save raw CSV ----
    df       = pd.DataFrame(records)
    raw_path = OUTPUT_DIR / "ablation_results.csv"
    df.to_csv(raw_path, index=False)
    print(f"\nRaw results  -> {raw_path}")

    # ---- Summary ----
    summary = (
        df.groupby("condition")[
            ["cumulative_reward", "jains_index", "min_collection_rate",
             "peak_aoi_proxy", "energy_efficiency", "coverage"]
        ]
        .agg(["mean", "std"])
        .round(4)
    )
    summary_path = OUTPUT_DIR / "ablation_summary.csv"
    summary.to_csv(summary_path)
    print(f"Summary      -> {summary_path}")
    print("\n", summary.to_string())

    # ---- Plots ----
    plot_ablation(df, active_conds, OUTPUT_DIR)


# ==================== PLOTTING ====================

def plot_ablation(df: pd.DataFrame, active_conds: list, out_dir: Path):
    n     = len(active_conds)
    cols  = COLORS[:n]

    # Map condition label -> x-label
    xlabel_map = dict(zip(
        [c[0] for c in CONDITIONS],
        CONDITION_XLABELS
    ))
    xlabels = [xlabel_map.get(c, c) for c in active_conds]

    # ---- Bar chart (3 metrics) with stripplot overlay ----
    metrics = ["cumulative_reward", "jains_index", "energy_efficiency"]
    ylabels = [r"Cumulative Reward ($\times 10^6$)", "Jain's Fairness Index",
               "Energy Efficiency (B/Wh)"]
    scale   = [1e6, 1.0, 1.0]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle(
        "Ablation Study — Impact of Each Design Component\n"
        f"($N={N_SENSORS}$ sensors, ${GRID_SIZE[0]}\\times{GRID_SIZE[1]}$ grid,"
        f" $n={len(SEEDS)}$ seeds)",
        fontweight="bold",
    )

    baseline_means = {}

    for ax, metric, ylabel, sc in zip(axes, metrics, ylabels, scale):
        means       = [df[df["condition"] == c][metric].mean() / sc for c in active_conds]
        stds        = [df[df["condition"] == c][metric].std()  / sc for c in active_conds]
        seed_vals   = [df[df["condition"] == c][metric].values / sc for c in active_conds]

        if means:
            baseline_means[metric] = means[0]

        bars = ax.bar(
            range(n), means, yerr=stds, color=cols,
            capsize=4, edgecolor="black", linewidth=0.6,
            error_kw={"elinewidth": 1.2, "ecolor": "black"},
            alpha=0.75, zorder=2,
        )

        # Stripplot overlay — shows individual seed variance
        ieee_style.add_stripplot(ax, seed_vals, list(range(n)), cols,
                                 jitter=0.10, s=28, alpha=0.80)

        baseline = means[0] if means else 1.0
        for i, (m, s, bar) in enumerate(zip(means, stds, bars)):
            top = m + s + 0.02 * (max(means) - min(means) + 1e-9)
            if i == 0:
                ax.text(bar.get_x() + bar.get_width() / 2, top,
                        "baseline", ha="center", va="bottom",
                        fontsize=8, color="gray")
            else:
                pct  = (m - baseline) / max(abs(baseline), 1e-9) * 100
                sign = "+" if pct >= 0 else ""
                c    = "#66a61e" if pct >= 0 else "#d95f02"
                ax.text(bar.get_x() + bar.get_width() / 2, top,
                        f"{sign}{pct:.1f}%", ha="center", va="bottom",
                        fontsize=8, fontweight="bold", color=c)

        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_xticks(range(n))
        ax.set_xticklabels(xlabels)
        ieee_style.clean_axes(ax)

    plt.tight_layout()
    ieee_style.save(fig, out_dir / "fig_ablation_bars")
    print(f"Bar chart    -> {out_dir / 'fig_ablation_bars.pdf'}")
    plt.close()

    # ---- Fairness distribution per seed — bars + stripplot ----
    fig, ax = plt.subplots(figsize=(max(8, n * 1.6), 5))

    seed_vals_fair = [df[df["condition"] == c]["jains_index"].values for c in active_conds]
    means_fair     = [v.mean() for v in seed_vals_fair]
    stds_fair      = [v.std()  for v in seed_vals_fair]

    ax.bar(range(n), means_fair, yerr=stds_fair, color=cols,
           capsize=4, edgecolor="black", linewidth=0.6,
           error_kw={"elinewidth": 1.2, "ecolor": "black"},
           alpha=0.70, zorder=2)

    # Mean tick marks
    for i, (m, col) in enumerate(zip(means_fair, cols)):
        ax.hlines(m, i - 0.28, i + 0.28, colors=col, linewidths=2.5, zorder=4)

    # Individual seed points
    ieee_style.add_stripplot(ax, seed_vals_fair, list(range(n)), cols,
                             jitter=0.10, s=35, alpha=0.85)

    ax.set_xticks(range(n))
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("Jain's Fairness Index", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"Fairness Index per Seed — Ablation Study\n"
        f"($N={N_SENSORS}$, ${GRID_SIZE[0]}\\times{GRID_SIZE[1]}$, $n={len(SEEDS)}$ seeds)",
        fontweight="bold",
    )
    ieee_style.clean_axes(ax)

    ieee_style.save(fig, out_dir / "fig_ablation_fairness")
    print(f"Fairness fig -> {out_dir / 'fig_ablation_fairness.pdf'}")
    plt.close()


# ==================== A4 GENERALISATION SWEEP ====================
#
# The standard ablation evaluates A4 on 500×500 (the no-DR model's own
# training distribution), which produces identical results because both
# models have converged to the same near-optimal policy there.
# This sweep re-evaluates both models across four grid sizes to expose
# the generalisation gap — the real contribution of domain randomisation.

A4_GRID_CONDITIONS = [
    (100,  100,  10),   # very small — easy generalisation
    (300,  300,  20),   # medium
    (500,  500,  20),   # training distribution (should match)
    (1000, 1000, 20),   # large — hardest generalisation
]

A4_SEEDS = [42, 123, 256, 789, 1337, 2024, 999, 314, 555, 2048]


def run_a4_generalisation(main_model, a4_model, out_dir: Path):
    """
    Evaluate Full Model vs No-DR model across grid sizes.
    Produces fig_a4_generalisation.png showing where DR matters.
    """
    if a4_model is None:
        print("\n[A4 generalisation skipped] No-DR model not available.")
        return

    records = []
    total   = len(A4_GRID_CONDITIONS) * len(A4_SEEDS) * 2
    done    = 0

    print("\n" + "=" * 65)
    print("A4 Generalisation Sweep — Full Model vs No-DR Model")
    print(f"  {len(A4_GRID_CONDITIONS)} grid sizes × {len(A4_SEEDS)} seeds × 2 models = {total} episodes")
    print("=" * 65)

    for gw, gh, ns in A4_GRID_CONDITIONS:
        cfg = {
            "grid_size":          (gw, gh),
            "num_sensors":        ns,
            "max_steps":          MAX_STEPS,
            "path_loss_exponent": 3.8,
            "rssi_threshold":     -85.0,
            "sensor_duty_cycle":  10.0,
        }
        label_grid = f"{gw}×{gh}"
        print(f"\n  Grid {label_grid}  (N={ns})")

        for model_name, model in [("Full Model (DR)", main_model),
                                   ("No Domain Rand.", a4_model)]:
            for seed in A4_SEEDS:
                done += 1
                try:
                    metrics = run_episode(FixedLayoutEnv, seed, model, cfg)
                    records.append({
                        "model":      model_name,
                        "grid":       label_grid,
                        "grid_area":  gw * gh,
                        "seed":       seed,
                        **metrics,
                    })
                    print(
                        f"    [{done:3d}/{total}] {model_name:22s} seed={seed:5d} "
                        f"reward={metrics['cumulative_reward']:.0f}  J={metrics['jains_index']:.4f}"
                    )
                except Exception as exc:
                    import traceback
                    print(f"    ERROR seed={seed}: {exc}")
                    traceback.print_exc()

    if not records:
        print("  No records — skipping plot.")
        return

    df = pd.DataFrame(records)
    csv_path = out_dir / "a4_generalisation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCSV -> {csv_path}")

    _plot_a4_generalisation(df, out_dir)


def _plot_a4_generalisation(df: pd.DataFrame, out_dir: Path):
    grids      = [f"{gw}×{gh}" for gw, gh, _ in A4_GRID_CONDITIONS]
    model_full = "Full Model (DR)"
    model_no   = "No Domain Rand."
    col_full   = "#1b7837"
    col_no     = "#d73027"

    metrics    = ["cumulative_reward", "jains_index", "energy_efficiency"]
    ylabels    = [r"Cumulative Reward ($\times 10^6$)", "Jain's Fairness Index",
                  "Energy Efficiency (B/Wh)"]
    scales     = [1e6, 1.0, 1.0]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle(
        "A4 Ablation — Domain Randomisation: Generalisation Across Grid Sizes\n"
        f"($n={len(A4_SEEDS)}$ seeds per condition)",
        fontweight="bold",
    )

    x      = np.arange(len(grids))
    width  = 0.35

    for ax, metric, ylabel, sc in zip(axes, metrics, ylabels, scales):
        for offset, (model_name, col) in enumerate(
            [(model_full, col_full), (model_no, col_no)]
        ):
            means, stds, all_vals = [], [], []
            for g in grids:
                vals = df[(df["model"] == model_name) & (df["grid"] == g)][metric].values / sc
                means.append(vals.mean() if len(vals) else 0.0)
                stds.append(vals.std()   if len(vals) else 0.0)
                all_vals.append(vals)

            pos = x + (offset - 0.5) * width
            ax.bar(pos, means, width=width, yerr=stds,
                   color=col, alpha=0.75, capsize=3,
                   edgecolor="black", linewidth=0.5,
                   error_kw={"elinewidth": 1.0, "ecolor": "black"},
                   label=model_name, zorder=2)

            for xi, vals in zip(pos, all_vals):
                jitter = (np.random.default_rng(0).random(len(vals)) - 0.5) * 0.08
                ax.scatter(xi + jitter, vals, color=col, s=18, alpha=0.7,
                           edgecolors="white", linewidths=0.3, zorder=3)

        # Annotate % gap at each grid size
        for i, g in enumerate(grids):
            m_full = df[(df["model"] == model_full) & (df["grid"] == g)][metric].mean()
            m_no   = df[(df["model"] == model_no)   & (df["grid"] == g)][metric].mean()
            if m_full > 0:
                pct = (m_full - m_no) / m_full * 100
                ymax = ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 1.0
                # rough top position — recomputed after ylim set below
                ax.annotate(
                    f"Δ{pct:+.1f}%",
                    xy=(x[i], max(m_full, m_no) / sc * 1.02),
                    ha="center", va="bottom", fontsize=7.5,
                    color="#d73027" if pct > 2 else "#555555",
                    fontweight="bold",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(grids)
        ax.set_xlabel("Grid Size", fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.legend(fontsize=8)
        ieee_style.clean_axes(ax)

    plt.tight_layout()
    ieee_style.save(fig, out_dir / "fig_a4_generalisation")
    print(f"A4 gen. fig  -> {out_dir / 'fig_a4_generalisation.pdf'}")
    plt.close()

    # Print summary table
    print("\n" + "=" * 65)
    print("A4 GENERALISATION — REWARD SUMMARY (mean ± std)")
    print("=" * 65)
    print(f"{'Grid':<12} {'Full Model (DR)':>22} {'No-DR Model':>22} {'DR advantage':>14}")
    print("-" * 72)
    for g in grids:
        mf = df[(df["model"] == model_full) & (df["grid"] == g)]["cumulative_reward"]
        mn = df[(df["model"] == model_no)   & (df["grid"] == g)]["cumulative_reward"]
        if len(mf) and len(mn):
            adv = (mf.mean() - mn.mean()) / mf.mean() * 100
            print(f"{g:<12}  {mf.mean():>10.0f} ± {mf.std():>7.0f}  "
                  f"{mn.mean():>10.0f} ± {mn.std():>7.0f}  {adv:>+.1f}%")
    print("=" * 65)


def main():
    print("=" * 65)
    print("Ablation Study — DQN Protocol-Aware UAV")
    print("=" * 65)

    main_model = _load_main_model()
    a4_model   = _load_a4_model()

    models = {"main": main_model, "a4": a4_model}

    records      = []
    active_conds = []

    for label, env_class, model_key in CONDITIONS:
        model = models[model_key]
        if model is None:
            continue

        print(f"\n{'='*65}")
        print(f"Condition: {label}")
        print(f"{'='*65}")
        active_conds.append(label)

        for seed in SEEDS:
            print(f"  Seed {seed}...", end=" ", flush=True)
            try:
                metrics             = run_episode(env_class, seed, model, EVAL_CONFIG)
                metrics["condition"] = label
                metrics["seed"]      = seed
                records.append(metrics)
                print(
                    f"reward={metrics['cumulative_reward']:.0f}  "
                    f"J={metrics['jains_index']:.4f}  "
                    f"min_rate={metrics['min_collection_rate']*100:.1f}%  "
                    f"peak_aoi={metrics['peak_aoi_proxy']:.3f}  "
                    f"eff={metrics['energy_efficiency']:.1f} B/Wh  "
                    f"NDR={metrics['coverage']*100:.0f}%"
                )
            except Exception as exc:
                import traceback
                print(f"ERROR: {exc}")
                traceback.print_exc()

    if not records:
        print("No results collected — exiting.")
        return

    df       = pd.DataFrame(records)
    raw_path = OUTPUT_DIR / "ablation_results.csv"
    df.to_csv(raw_path, index=False)
    print(f"\nRaw results  -> {raw_path}")

    summary = (
        df.groupby("condition")[
            ["cumulative_reward", "jains_index", "min_collection_rate",
             "peak_aoi_proxy", "energy_efficiency", "coverage"]
        ]
        .agg(["mean", "std"])
        .round(4)
    )
    summary_path = OUTPUT_DIR / "ablation_summary.csv"
    summary.to_csv(summary_path)
    print(f"Summary      -> {summary_path}")
    print("\n", summary.to_string())

    plot_ablation(df, active_conds, OUTPUT_DIR)

    # A4 generalisation — cross-grid comparison
    run_a4_generalisation(main_model, a4_model, OUTPUT_DIR)


if __name__ == "__main__":
    main()
