"""
DQN Evaluation & Visualization with Frame Stacking + Zero-Padding Support

Works with both the original model (dqn_full_observability) and the new
domain-randomised model (dqn_domain_rand).  Zero-padding is applied
automatically based on max_sensors_limit in training_config.json.

Author: ATILADE GABRIEL OKE
Modified: domain-rand + zero-padding support
"""

import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent        # dqn/
src_dir    = script_dir.parent.parent               # src/
sys.path.insert(0, str(src_dir))

import numpy as np
import gymnasium
import json
import time
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
from environment.uav_env import UAVEnvironment

# ==================== CONFIGURATION ====================
# ← Switch between models here — only these three lines need changing

MODEL_DIR  = script_dir / "models" / "dqn_domain_rand"   # or dqn_full_observability
MODEL_PATH = MODEL_DIR / "dqn_final.zip"
CONFIG_PATH= MODEL_DIR / "training_config.json"
VEC_NORMALIZE_PATH = MODEL_DIR / "vec_normalize.pkl"      # ignored if absent

OUTPUT_DIR = script_dir / "dqn_evaluation_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EVAL_CONFIG = {
    "grid_size":         (500, 500),
    "num_sensors":       20,
    "max_battery":       274.0,
    "max_steps":         2100,
    "sensor_duty_cycle": 10.0,
    "render_mode":       None,
}

EVAL_MAX_BATTERY = 274.0

VIZ_CONFIG = {"progress_interval": 50}

print("Model:  {}".format(MODEL_PATH))
print("Output: {}".format(OUTPUT_DIR))

# ==================== ENVIRONMENT WRAPPER ====================

class AnalysisUAVEnv(UAVEnvironment):
    """
    Wrapper that:
      1. Snapshots sensor data just before VecEnv auto-reset fires.
      2. Zero-pads the observation to max_sensors_limit so it matches
         the neural network input size regardless of active sensor count.

    features_per_sensor is auto-detected — no hardcoded constants.
    """

    def __init__(self, max_sensors_limit: int = 50, **kwargs):
        self.max_sensors_limit = max_sensors_limit
        super().__init__(**kwargs)

        # Auto-detect features_per_sensor from the raw obs space
        raw = self.observation_space.shape[0]
        self._fps = 0
        for uav_f in range(raw + 1):
            rem = raw - uav_f
            if rem > 0 and rem % self.num_sensors == 0:
                self._fps = rem // self.num_sensors
                break
        if self._fps == 0:
            raise ValueError(
                "Cannot detect features_per_sensor: raw={}, num_sensors={}".format(
                    raw, self.num_sensors
                )
            )

        padded = raw + (max_sensors_limit - self.num_sensors) * self._fps
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(padded,), dtype=np.float32
        )

        self.last_episode_sensor_data = None
        self.last_episode_info        = None
        print(
            "  AnalysisUAVEnv: raw={} padded={} "
            "(n_sensors={}, max={}, fps={})".format(
                raw, padded, self.num_sensors, max_sensors_limit, self._fps
            )
        )

    def _pad(self, obs):
        n_pad = (self.max_sensors_limit - self.num_sensors) * self._fps
        return np.concatenate([obs, np.zeros(n_pad, dtype=np.float32)]).astype(np.float32)

    def reset(self, **kwargs):
        # Snapshot before parent wipes state
        if hasattr(self, "sensors") and self.current_step > 0:
            self.last_episode_sensor_data = [
                {
                    "sensor_id":              int(s.sensor_id),
                    "position":               tuple(float(x) for x in s.position),
                    "total_data_generated":   float(s.total_data_generated),
                    "total_data_transmitted": float(s.total_data_transmitted),
                    "total_data_lost":        float(s.total_data_lost),
                    "data_buffer":            float(s.data_buffer),
                }
                for s in self.sensors
            ]
            self.last_episode_info = {
                "battery":             self.uav.battery,
                "battery_percent":     self.uav.get_battery_percentage(),
                "coverage_percentage": len(self.sensors_visited) / self.num_sensors * 100,
            }
        obs, info = super().reset(**kwargs)
        return self._pad(obs), info

    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        return self._pad(obs), r, term, trunc, info


# ==================== HELPERS ====================

def load_config(config_path):
    """Load training_config.json — handles both old and new format."""
    defaults = {
        "use_frame_stacking": True,
        "n_stack":            4,
        "max_sensors_limit":  50,
    }
    try:
        with open(config_path) as f:
            loaded = json.load(f)
        merged = {**defaults, **loaded}
        print("  Config: frame_stack={} n_stack={} max_sensors={}".format(
            merged["use_frame_stacking"], merged["n_stack"],
            merged["max_sensors_limit"]
        ))
        return merged
    except FileNotFoundError:
        print("  Config not found — using defaults")
        return defaults


def _unwrap(vec):
    inner = vec
    while hasattr(inner, "venv"):
        inner = inner.venv
    env = inner.envs[0]
    while hasattr(env, "env"):
        env = env.env
    return env


def create_eval_env(eval_config, training_config):
    max_sensors_limit = training_config.get("max_sensors_limit", 50)
    env_kwargs = {**eval_config, "max_sensors_limit": max_sensors_limit}

    vec = DummyVecEnv([lambda: AnalysisUAVEnv(**env_kwargs)])

    if training_config.get("use_frame_stacking", True):
        n_stack = training_config.get("n_stack", 4)
        vec = VecFrameStack(vec, n_stack=n_stack)
        print("  Frame stacking: n_stack={}".format(n_stack))

    if VEC_NORMALIZE_PATH.exists():
        try:
            vec = VecNormalize.load(str(VEC_NORMALIZE_PATH), vec)
            vec.training    = False
            vec.norm_reward = False
            print("  VecNormalize: loaded")
        except AssertionError as e:
            print("  VecNormalize: skipped (shape mismatch — stale pkl)")
    else:
        print("  VecNormalize: not found — skipping")

    base = _unwrap(vec)
    print("  Base env: {} | battery={:.1f}".format(
        type(base).__name__, base.uav.battery
    ))
    return vec, base


def calculate_fairness(rates):
    if not rates:
        return {}
    n  = len(rates); s1 = sum(rates); s2 = sum(x**2 for x in rates)
    return {
        "mean":    np.mean(rates),
        "std":     np.std(rates),
        "min":     np.min(rates),
        "max":     np.max(rates),
        "range":   np.max(rates) - np.min(rates),
        "jains":   (s1**2) / (n * s2) if s2 > 0 else 1.0,
        "starved": sum(1 for r in rates if r < 20),
    }


def fairness_label(std):
    if std < 15: return "EXCELLENT", "+++"
    if std < 25: return "GOOD",      "++"
    if std < 35: return "MODERATE",  "+"
    return "POOR", "x"


# ==================== MAIN ====================

def main():
    print("=" * 80)
    print("DQN EVALUATION — ZERO-PADDING + FRAME STACKING")
    print("=" * 80)

    training_config = load_config(CONFIG_PATH)

    print("\nLoading model...")
    try:
        model = DQN.load(MODEL_PATH)
        print("  Model loaded: {}".format(MODEL_PATH.name))
    except Exception as e:
        print("  ERROR: {}".format(e))
        return

    print("Creating environment...")
    eval_env, base_env = create_eval_env(EVAL_CONFIG, training_config)
    print()

    # ── Evaluate ──────────────────────────────────────────────────────
    obs = eval_env.reset()
    print("Obs shape: {}  |  Model obs space: {}".format(
        obs.shape, model.observation_space.shape
    ))
    if obs.shape[1:] != model.observation_space.shape:
        print("  WARNING: shape mismatch — model may produce garbage predictions")
    print()

    done            = False
    step            = 0
    total_reward    = 0.0
    pre_battery     = base_env.uav.battery
    pre_coverage    = 0.0
    t0              = time.time()

    history = {
        "step": [], "cumulative_reward": [], "battery_percent": [],
        "battery_wh": [], "coverage_percent": [], "instant_reward": [],
    }

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        av = int(action[0]) if hasattr(action, "__len__") else int(action)

        # Pre-step snapshot — VecEnv auto-reset fires inside step()
        pre_battery  = base_env.uav.battery
        pre_batt_pct = base_env.uav.get_battery_percentage()
        pre_coverage = (
            len(base_env.sensors_visited) / base_env.num_sensors * 100
            if hasattr(base_env, "sensors_visited") else 0.0
        )

        obs, rewards, dones, infos = eval_env.step([av])
        reward = float(rewards[0])
        done   = bool(dones[0])
        total_reward += reward
        step += 1

        if done:
            history["step"].append(step)
            history["cumulative_reward"].append(total_reward)
            history["battery_percent"].append(pre_batt_pct)
            history["battery_wh"].append(pre_battery)
            history["coverage_percent"].append(pre_coverage)
            history["instant_reward"].append(reward)
            print(
                "Step {:>4}: Cov={:>5.1f}%  Bat={:>5.1f}%  "
                "Rew={:>10.1f}  [FINAL]".format(
                    step, pre_coverage, pre_batt_pct, total_reward
                )
            )

        elif step % VIZ_CONFIG["progress_interval"] == 0:
            bp  = base_env.uav.get_battery_percentage()
            bw  = base_env.uav.battery
            cov = (len(base_env.sensors_visited) / base_env.num_sensors * 100
                   if hasattr(base_env, "sensors_visited") else 0)
            history["step"].append(step)
            history["cumulative_reward"].append(total_reward)
            history["battery_percent"].append(bp)
            history["battery_wh"].append(bw)
            history["coverage_percent"].append(cov)
            history["instant_reward"].append(reward)
            print(
                "Step {:>4}: Cov={:>5.1f}%  Bat={:>5.1f}%  "
                "Rew={:>10.1f}  InstRew={:>8.1f}".format(
                    step, cov, bp, total_reward, reward
                )
            )

    elapsed = time.time() - t0

    # ── Pull sensor data ───────────────────────────────────────────────
    if base_env.last_episode_sensor_data:
        sensor_data = base_env.last_episode_sensor_data
        final_info  = base_env.last_episode_info
        print("\n  Pre-reset snapshot recovered ({} sensors)".format(len(sensor_data)))
    else:
        print("\n  WARNING: no snapshot found — sensor metrics will be empty")
        sensor_data = []
        final_info  = {
            "battery":             pre_battery,
            "battery_percent":     (pre_battery / EVAL_MAX_BATTERY) * 100,
            "coverage_percentage": pre_coverage,
        }

    # ── Summary ───────────────────────────────────────────────────────
    total_gen  = sum(s["total_data_generated"]   for s in sensor_data)
    total_col  = sum(s["total_data_transmitted"] for s in sensor_data)
    total_lost = sum(s["total_data_lost"]        for s in sensor_data)
    efficiency = (total_col / total_gen * 100) if total_gen > 0 else 0
    battery_used = EVAL_MAX_BATTERY - final_info["battery"]
    bpw          = total_col / battery_used if battery_used > 0 else 0

    print("\n" + "=" * 80)
    print("EPISODE SUMMARY")
    print("=" * 80)
    print("  Total Reward:       {:>12.1f}".format(total_reward))
    print("  Steps:              {:>12}".format(step))
    print("  Elapsed:            {:>12.1f}s".format(elapsed))
    print("  Coverage:           {:>12.1f}%".format(final_info["coverage_percentage"]))
    print("  Collection Eff:     {:>12.1f}%".format(efficiency))
    print("  Battery Used:       {:>12.1f} Wh".format(battery_used))
    print("  Bytes / Wh:         {:>12.1f}".format(bpw))

    # ── Per-sensor fairness ────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PER-SENSOR FAIRNESS")
    print("=" * 80)
    print("{:<8} {:<10} {:<10} {:<8} {}".format(
        "Sensor", "Generated", "Collected", "Eff%", "Bar"
    ))
    print("-" * 80)

    sensor_rates        = []
    sensor_fairness_rows = []

    for s in sensor_data:
        gen = s["total_data_generated"]
        col = s["total_data_transmitted"]
        pct = (col / gen * 100) if gen > 0 else 0.0
        sensor_rates.append(pct)
        sensor_fairness_rows.append({
            "sensor_id":          s["sensor_id"],
            "position_x":         s["position"][0],
            "position_y":         s["position"][1],
            "data_generated":     gen,
            "data_transmitted":   col,
            "data_lost":          s["total_data_lost"],
            "efficiency_percent": pct,
            "buffer_remaining":   s["data_buffer"],
        })
        bar = "█" * int(pct / 2.5)
        print("S{:<7} {:<10.0f} {:<10.0f} {:<8.1f} {}".format(
            s["sensor_id"], gen, col, pct, bar
        ))

    f = calculate_fairness(sensor_rates)
    if f:
        lvl, sym = fairness_label(f["std"])
        print("\n  Fairness: {}  {}".format(lvl, sym))
        print("  Std: {:.1f}%  Min: {:.1f}%  Max: {:.1f}%  "
              "Jain's: {:.4f}  Starved(<20%): {}/{}".format(
                  f["std"], f["min"], f["max"],
                  f["jains"], f["starved"], len(sensor_rates)
              ))

    # ── Save outputs ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SAVING OUTPUTS")
    print("=" * 80)

    pd.DataFrame(history).to_csv(OUTPUT_DIR / "dqn_results.csv", index=False)
    print("  Saved: dqn_results.csv")

    if sensor_fairness_rows:
        pd.DataFrame(sensor_fairness_rows).to_csv(
            OUTPUT_DIR / "dqn_sensor_fairness.csv", index=False
        )
        print("  Saved: dqn_sensor_fairness.csv")

    summary = {
        "model":                     str(MODEL_PATH),
        "total_reward":              float(total_reward),
        "steps":                     int(step),
        "elapsed_seconds":           float(elapsed),
        "coverage_pct":              float(final_info["coverage_percentage"]),
        "collection_efficiency_pct": float(efficiency),
        "battery_used_wh":           float(battery_used),
        "bytes_per_wh":              float(bpw),
        "total_generated":           float(total_gen),
        "total_collected":           float(total_col),
        "total_lost":                float(total_lost),
        "fairness": {k: float(v) for k, v in f.items()} if f else {},
    }
    with open(OUTPUT_DIR / "dqn_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("  Saved: dqn_summary.json")

    eval_env.close()
    print("\nDone. Results in: {}".format(OUTPUT_DIR))


if __name__ == "__main__":
    main()