"""
sweep_eval.py
=============
Multi-seed, multi-grid sweep evaluation across all agents.

Agents: DQN, Relational RL, Smart Greedy V2, Nearest Sensor Greedy, TSP Oracle,
        GTrXL Transformer (optional — set INCLUDE_GTRXL = True when ready).

GPU scheduling:
  - DQN and Relational RL use CUDA for inference via thread-local model instances.
  - GTrXL uses Ray with num_gpus allocated.
  - Greedy/TSP agents run on CPU via a joblib process pool.
  - N_GPU_THREADS controls how many episodes run concurrently on GPU.

Crash-safe:
  - Results appended to CSV after every episode batch.
  - On restart, completed (agent, grid_w, grid_h, n_sensors, seed, sweep_type)
    tuples are skipped automatically.

Sweeps:
  Main  — 5 curriculum configs × 50 seeds  (strict grid/sensor pairing)
  Cross — 25 grid×sensor combos × 10 seeds  (full heatmap coverage)

Usage:
    PYTHONIOENCODING=utf-8 uv run python \\
        src/agents/dqn/dqn_evaluation_results/sweep_eval.py
"""

from __future__ import annotations

import csv
import logging
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import gymnasium

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE   = Path(__file__).resolve().parent       # dqn_evaluation_results/
_DQN    = _HERE.parent                          # dqn/
_SRC    = _HERE.parents[2]                      # src/
_ROOT   = _HERE.parents[3]                      # project root

for _p in (str(_SRC), str(_DQN), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

for _sub in ("relational_policy", "transformer_policy"):
    _sp = str(_SRC / "experiments" / _sub)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

for _gnn in _DQN.rglob("gnn_extractor.py"):
    _gd = str(_gnn.parent)
    if _gd not in sys.path:
        sys.path.insert(0, _gd)

from environment.uav_env import UAVEnvironment  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------
try:
    import torch
    DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
    _CUDA_CNT  = torch.cuda.device_count()
except ImportError:
    DEVICE    = "cpu"
    _CUDA_CNT = 0

log.info("Device: %s  (CUDA devices: %d)", DEVICE, _CUDA_CNT)

# ---------------------------------------------------------------------------
# ── CONFIGURATION ────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

INCLUDE_GTRXL: bool = False   # flip True when transformer checkpoint is ready

# Main sweep — curriculum-aligned (grid size scales with sensor count)
MAIN_CONFIGS: list[dict] = [
    {"grid_size": (100, 100), "num_sensors": 10},
    {"grid_size": (200, 200), "num_sensors": 20},
    {"grid_size": (300, 300), "num_sensors": 30},
    {"grid_size": (400, 400), "num_sensors": 40},
    {"grid_size": (500, 500), "num_sensors": 50},
]
N_MAIN_SEEDS  = 50
MAIN_SEEDS    = list(range(N_MAIN_SEEDS))

# Cross sweep — full grid × sensor grid for heatmaps
CROSS_GRIDS   = [100, 200, 300, 400, 500]
CROSS_SENSORS = [10, 20, 30, 40, 50]
N_CROSS_SEEDS = 10
CROSS_SEEDS   = list(range(100, 100 + N_CROSS_SEEDS))  # different from main seeds

# Shared env physics
ENV_BASE: dict[str, Any] = {
    "max_steps":          2100,
    "path_loss_exponent": 3.8,
    "rssi_threshold":     -85.0,
    "sensor_duty_cycle":  10.0,
}
MAX_BATTERY: float = 274.0

# Parallelism
N_GPU_THREADS: int = 4 if DEVICE == "cuda" else 2   # concurrent RL episodes on GPU
N_CPU_WORKERS: int = max(1, (os.cpu_count() or 4) - 2)  # greedy process pool

# Model paths (relative — works on local and pod)
DQN_MODEL_PATH = _DQN / "models" / "dqn_v3_retrain" / "dqn_final.zip"
DQN_CONFIG_PATH = _DQN / "models" / "dqn_v3_retrain" / "training_config.json"
DQN_MAX_SENSORS = 50

REL_CKPT_DIR = (
    _DQN / "models" / "relational_rl" /
    "results" / "checkpoints" / "stage_4" / "final"
)

GTRXL_CKPT_DIR = _ROOT / "models" / "transformer_gtrxl_stage1" / "stage_1_progress"
GTRXL_FALLBACK  = _ROOT / "models" / "transformer_gtrxl" / "stage_0_progress"

# Output
OUTPUT_DIR = _HERE / "baseline_results" / "sweep"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = OUTPUT_DIR / "sweep_results.csv"

CSV_FIELDNAMES = [
    "agent", "seed", "grid_w", "grid_h", "n_sensors", "sweep_type",
    "ndr_pct", "jfi", "gini", "alpha_fairness",
    "total_data_bytes", "data_lost_bytes",
    "final_battery_wh", "battery_pct", "bytes_per_wh",
    "mean_aoi", "peak_aoi", "aoi_variance",
    "buffer_variance", "starved_sensors",
    "total_reward", "steps",
]

# ---------------------------------------------------------------------------
# ── THREAD-SAFE CSV WRITER ───────────────────────────────────────────────────
# ---------------------------------------------------------------------------
_csv_lock = threading.Lock()


def _append_row(row: dict) -> None:
    with _csv_lock:
        write_header = not RESULTS_CSV.exists()
        with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDNAMES})


def _load_done_keys() -> set[tuple]:
    if not RESULTS_CSV.exists():
        return set()
    import pandas as pd
    try:
        df = pd.read_csv(RESULTS_CSV)
        return set(
            zip(df["agent"], df["grid_w"], df["grid_h"],
                df["n_sensors"], df["seed"], df["sweep_type"])
        )
    except Exception:
        return set()


# ---------------------------------------------------------------------------
# ── METRIC HELPERS ────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def _sensor_dicts_from_env(env: UAVEnvironment) -> list[dict]:
    return [
        {
            "total_data_generated":   float(s.total_data_generated),
            "total_data_transmitted": float(s.total_data_transmitted),
            "total_data_lost":        float(s.total_data_lost),
            "data_buffer":            float(s.data_buffer),
            "max_buffer_size":        float(s.max_buffer_size),
        }
        for s in env.sensors
    ]


def _compute_metrics(
    sensor_dicts: list[dict],
    n_sensors: int,
    sensors_visited_count: int,
    total_data_bytes: float,
    data_lost_bytes: float,
    battery: float,
    total_reward: float,
    steps: int,
) -> dict:
    energy = max(MAX_BATTERY - battery, 1e-9)
    rates = [
        (s["total_data_transmitted"] / s["total_data_generated"] * 100.0)
        if s["total_data_generated"] > 0 else 0.0
        for s in sensor_dicts
    ]

    # JFI
    n  = len(rates)
    s2 = sum(r ** 2 for r in rates)
    jfi = float(sum(rates) ** 2 / (n * s2)) if n > 0 and s2 > 0 else 0.0

    # Gini
    arr   = sorted(rates)
    total = sum(arr)
    if total > 0 and n > 1:
        gini = float(
            sum((2 * i - n - 1) * r for i, r in enumerate(arr, 1)) / (n * total)
        )
    else:
        gini = 0.0

    # α-fairness at α=2 (harmonic / arithmetic ratio → [0, 1])
    pos = [r for r in rates if r > 0]
    if pos:
        harm  = len(pos) / sum(1.0 / r for r in pos)
        arith = sum(pos) / len(pos)
        alpha_f = float(harm / arith) if arith > 0 else 0.0
    else:
        alpha_f = 0.0

    # AoI proxy — buffer occupancy (high occupancy = stale data = high AoI)
    aoi = [
        s["data_buffer"] / max(s["max_buffer_size"], 1e-9)
        for s in sensor_dicts
    ]
    mean_aoi = float(np.mean(aoi))
    peak_aoi = float(max(aoi))
    aoi_var  = float(np.var(aoi))

    starved = sum(1 for r in rates if r < 20.0)

    return {
        "ndr_pct":          (sensors_visited_count / n_sensors) * 100.0,
        "jfi":              jfi,
        "gini":             gini,
        "alpha_fairness":   alpha_f,
        "total_data_bytes": total_data_bytes,
        "data_lost_bytes":  data_lost_bytes,
        "final_battery_wh": battery,
        "battery_pct":      battery / MAX_BATTERY * 100.0,
        "bytes_per_wh":     total_data_bytes / energy,
        "mean_aoi":         mean_aoi,
        "peak_aoi":         peak_aoi,
        "aoi_variance":     aoi_var,
        "buffer_variance":  aoi_var,   # same source, different framing
        "starved_sensors":  starved,
        "total_reward":     total_reward,
        "steps":            steps,
    }


# ---------------------------------------------------------------------------
# ── DQN — zero-padded env wrapper ────────────────────────────────────────────
# ---------------------------------------------------------------------------

class _DQNEnv(UAVEnvironment):
    """Zero-pads observations and snapshots episode state before auto-reset."""

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
        if self._fps == 0:
            raise ValueError(f"Cannot infer fps: raw={raw}, n={self.num_sensors}")

        padded = raw + (self.max_sensors_limit - self.num_sensors) * self._fps
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(padded,), dtype=np.float32
        )

    def _pad(self, obs: np.ndarray) -> np.ndarray:
        pad = np.zeros(
            (self.max_sensors_limit - self.num_sensors) * self._fps, dtype=np.float32
        )
        return np.concatenate([obs, pad]).astype(np.float32)

    def reset(self, **kw):
        if hasattr(self, "sensors") and self.current_step > 0:
            self._snap = {
                "sensor_dicts":         _sensor_dicts_from_env(self),
                "sensors_visited_count": len(self.sensors_visited),
                "total_data_bytes":     float(self.total_data_collected),
                "data_lost_bytes":      float(sum(s.total_data_lost for s in self.sensors)),
                "battery":              float(self.uav.battery),
                "n_sensors":            self.num_sensors,
            }
        obs, info = super().reset(**kw)
        return self._pad(obs), info

    def step(self, action):
        obs, r, te, tr, info = super().step(action)
        return self._pad(obs), r, te, tr, info


# ---------------------------------------------------------------------------
# ── THREAD-LOCAL MODEL CACHE ──────────────────────────────────────────────────
# Each GPU thread owns its own model instance to avoid SB3 / RLlib race conditions.
# ---------------------------------------------------------------------------
_tl = threading.local()


def _get_dqn():
    if not hasattr(_tl, "dqn"):
        import json as _json
        from stable_baselines3 import DQN as _DQN
        log.info("[thread %s] Loading DQN onto %s", threading.get_ident(), DEVICE)
        _tl.dqn = _DQN.load(str(DQN_MODEL_PATH), device=DEVICE)
        _tl.dqn.policy.set_training_mode(False)
        _tl.dqn_cfg = {"use_frame_stacking": True, "n_stack": 4, "max_sensors_limit": DQN_MAX_SENSORS}
        try:
            with open(DQN_CONFIG_PATH) as f:
                _tl.dqn_cfg.update(_json.load(f))
        except FileNotFoundError:
            pass
    return _tl.dqn, _tl.dqn_cfg


def _get_rel():
    if not hasattr(_tl, "rel"):
        from relational_rl_runner import load_relational_rl_module as _load
        log.info("[thread %s] Loading Relational RL onto %s", threading.get_ident(), DEVICE)
        mod = _load(REL_CKPT_DIR)
        if DEVICE == "cuda" and hasattr(mod, "to"):
            mod = mod.to(DEVICE)
        _tl.rel = mod
    return _tl.rel


# ---------------------------------------------------------------------------
# ── EPISODE RUNNERS ───────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def _run_dqn(grid_size, num_sensors, seed) -> dict:
    import torch as _torch
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

    model, cfg = _get_dqn()
    random.seed(seed); np.random.seed(seed)

    env_kw = {**ENV_BASE, "grid_size": grid_size, "num_sensors": num_sensors,
              "max_sensors_limit": cfg["max_sensors_limit"],
              "include_sensor_positions": False}
    vec = DummyVecEnv([lambda: _DQNEnv(**env_kw)])
    if cfg.get("use_frame_stacking", True):
        vec = VecFrameStack(vec, n_stack=cfg.get("n_stack", 4))

    inner = vec
    while hasattr(inner, "venv"):
        inner = inner.venv
    base: _DQNEnv = inner.envs[0]

    obs   = vec.reset()
    total_r = 0.0
    steps   = 0
    done    = np.array([False])
    while not done[0]:
        with _torch.no_grad():
            action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, _ = vec.step(action)
        total_r += float(rewards[0])
        steps   += 1
        done     = dones
    vec.close()

    snap = base._snap
    if snap is None:
        return {}
    return _compute_metrics(
        snap["sensor_dicts"], snap["n_sensors"],
        snap["sensors_visited_count"],
        snap["total_data_bytes"], snap["data_lost_bytes"],
        snap["battery"], total_r, steps,
    )


def _run_relational(grid_size, num_sensors, seed) -> dict:
    import torch as _torch
    from ray.rllib.core.columns import Columns
    from relational_rl_runner import InferenceRelationalUAVEnv

    module = _get_rel()
    random.seed(seed); np.random.seed(seed)

    env = InferenceRelationalUAVEnv(
        n_max=50, grid_size=grid_size, num_sensors=num_sensors, **ENV_BASE
    )
    obs, _ = env.reset(seed=seed)
    total_r = 0.0
    steps   = 0

    while True:
        batch = {Columns.OBS: {
            k: _torch.as_tensor(np.asarray(v), device=DEVICE).unsqueeze(0)
            for k, v in obs.items()
        }}
        with _torch.no_grad():
            out = module._forward_inference(batch)
        action = int(_torch.argmax(out[Columns.ACTION_DIST_INPUTS], dim=-1).item())
        obs, reward, terminated, truncated, _ = env.step(action)
        total_r += float(reward)
        steps   += 1
        if terminated or truncated:
            break

    return _compute_metrics(
        _sensor_dicts_from_env(env), env.num_sensors,
        len(env.sensors_visited),
        float(env.total_data_collected),
        float(sum(s.total_data_lost for s in env.sensors)),
        float(env.uav.battery), total_r, steps,
    )


# Greedy runners are called from subprocess workers (loky) — must be top-level
def _greedy_worker(task: dict) -> dict | None:
    import sys as _sys
    for _p in (task["src"], task["dqn"]):
        if _p not in _sys.path:
            _sys.path.insert(0, _p)

    import random as _rnd
    import numpy as _np
    from environment.uav_env import UAVEnvironment as _Env
    from greedy_agents import MaxThroughputGreedyV2, NearestSensorGreedy, TSPOracleAgent

    agent_name = task["agent"]
    grid_size  = tuple(task["grid_size"])
    num_sensors = task["num_sensors"]
    seed        = task["seed"]
    env_base    = task["env_base"]

    _rnd.seed(seed); _np.random.seed(seed)

    env  = _Env(grid_size=grid_size, num_sensors=num_sensors, **env_base)
    obs, _ = env.reset(seed=seed)

    if agent_name == "Smart Greedy V2":
        agent = MaxThroughputGreedyV2(env)
    elif agent_name == "Nearest Greedy":
        agent = NearestSensorGreedy(env)
    elif agent_name == "TSP Oracle":
        agent = TSPOracleAgent(env)
        agent.reset()
    else:
        return None

    total_r = 0.0
    steps   = 0
    while True:
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_r += float(reward)
        steps   += 1
        if terminated or truncated:
            break

    n = env.num_sensors
    energy = max(task["max_battery"] - float(env.uav.battery), 1e-9)
    sensor_dicts = [
        {
            "total_data_generated":   float(s.total_data_generated),
            "total_data_transmitted": float(s.total_data_transmitted),
            "total_data_lost":        float(s.total_data_lost),
            "data_buffer":            float(s.data_buffer),
            "max_buffer_size":        float(s.max_buffer_size),
        }
        for s in env.sensors
    ]
    # compute metrics inline (avoid importing from parent module in subprocess)
    rates = [
        (s["total_data_transmitted"] / s["total_data_generated"] * 100.0)
        if s["total_data_generated"] > 0 else 0.0
        for s in sensor_dicts
    ]
    s2   = sum(r ** 2 for r in rates)
    jfi  = float(sum(rates) ** 2 / (n * s2)) if n > 0 and s2 > 0 else 0.0
    arr  = sorted(rates); tot = sum(arr)
    gini = float(sum((2*i - n - 1) * r for i, r in enumerate(arr, 1)) / (n * tot)) if tot > 0 and n > 1 else 0.0
    pos  = [r for r in rates if r > 0]
    harm = len(pos) / sum(1.0 / r for r in pos) if pos else 0.0
    arith = sum(pos) / len(pos) if pos else 0.0
    alpha_f = float(harm / arith) if arith > 0 else 0.0
    aoi  = [s["data_buffer"] / max(s["max_buffer_size"], 1e-9) for s in sensor_dicts]

    return {
        "ndr_pct":          (len(env.sensors_visited) / n) * 100.0,
        "jfi":              jfi,
        "gini":             gini,
        "alpha_fairness":   alpha_f,
        "total_data_bytes": float(env.total_data_collected),
        "data_lost_bytes":  float(sum(s["total_data_lost"] for s in sensor_dicts)),
        "final_battery_wh": float(env.uav.battery),
        "battery_pct":      float(env.uav.battery) / task["max_battery"] * 100.0,
        "bytes_per_wh":     float(env.total_data_collected) / energy,
        "mean_aoi":         float(_np.mean(aoi)),
        "peak_aoi":         float(max(aoi)),
        "aoi_variance":     float(_np.var(aoi)),
        "buffer_variance":  float(_np.var(aoi)),
        "starved_sensors":  sum(1 for r in rates if r < 20.0),
        "total_reward":     total_r,
        "steps":            steps,
    }


# ---------------------------------------------------------------------------
# ── GTRXL RUNNER (sequential, Ray-backed) ────────────────────────────────────
# ---------------------------------------------------------------------------
_gtrxl_algo = None


def _load_gtrxl():
    global _gtrxl_algo
    ckpt = GTRXL_CKPT_DIR if GTRXL_CKPT_DIR.exists() else (
        GTRXL_FALLBACK if GTRXL_FALLBACK.exists() else None
    )
    if ckpt is None:
        log.warning("GTrXL checkpoint not found — skipping GTrXL")
        return False

    import ray
    from ray import tune
    from ray.rllib.algorithms.ppo import PPOConfig
    from transformer_model import register_model
    from env_wrapper import TransformerObsWrapper

    n_gpus = min(_CUDA_CNT, 1)
    ray.init(ignore_reinit_error=True, num_cpus=4, num_gpus=n_gpus)
    tune.register_env("UAVTransformerEnv", lambda cfg: TransformerObsWrapper(cfg))
    model_cfg = register_model()

    cfg = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False,
                   enable_env_runner_and_connector_v2=False)
        .environment("UAVTransformerEnv",
                     env_config={"grid_size": (100, 100), "num_sensors": 10})
        .training(model=model_cfg)
        .env_runners(num_env_runners=0)
        .resources(num_gpus=n_gpus)
        .debugging(log_level="ERROR")
    )
    _gtrxl_algo = cfg.build_algo()
    _gtrxl_algo.restore(str(ckpt))
    log.info("GTrXL loaded from %s", ckpt)
    return True


def _run_gtrxl(grid_size, num_sensors, seed) -> dict:
    from env_wrapper import TransformerObsWrapper
    random.seed(seed); np.random.seed(seed)

    env = TransformerObsWrapper({"grid_size": grid_size, "num_sensors": num_sensors, **ENV_BASE})
    obs, _ = env.reset(seed=seed)
    policy = _gtrxl_algo.get_policy()
    state  = policy.get_initial_state()
    total_r = 0.0
    steps   = 0

    while True:
        action, state, _ = policy.compute_single_action(obs, state=state, explore=False)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_r += float(reward)
        steps   += 1
        if terminated or truncated:
            break

    base = env.env
    return _compute_metrics(
        _sensor_dicts_from_env(base), base.num_sensors,
        len(base.sensors_visited),
        float(base.total_data_collected),
        float(sum(s.total_data_lost for s in base.sensors)),
        float(base.uav.battery), total_r, steps,
    )


# ---------------------------------------------------------------------------
# ── TASK GENERATION ────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def _build_tasks() -> list[dict]:
    tasks = []
    for cfg in MAIN_CONFIGS:
        g = cfg["grid_size"]
        n = cfg["num_sensors"]
        for seed in MAIN_SEEDS:
            tasks.append({"grid_size": g, "num_sensors": n,
                          "seed": seed, "sweep_type": "main"})
    for gw in CROSS_GRIDS:
        for n in CROSS_SENSORS:
            for seed in CROSS_SEEDS:
                tasks.append({"grid_size": (gw, gw), "num_sensors": n,
                              "seed": seed, "sweep_type": "cross"})
    return tasks


def _task_key(agent, t) -> tuple:
    g = t["grid_size"]
    return (agent, g[0], g[1], t["num_sensors"], t["seed"], t["sweep_type"])


# ---------------------------------------------------------------------------
# ── MAIN SWEEP ─────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def run_sweep() -> None:
    all_tasks = _build_tasks()
    done_keys = _load_done_keys()
    log.info("Total task slots already done: %d", len(done_keys))

    RL_AGENTS      = []
    GREEDY_AGENTS  = ["Smart Greedy V2", "Nearest Greedy", "TSP Oracle"]

    # ── Discover available RL agents ─────────────────────────────────────
    if DQN_MODEL_PATH.exists():
        RL_AGENTS.append("DQN")
        log.info("DQN ✓  (%s)", DQN_MODEL_PATH.name)
    else:
        log.warning("DQN model not found at %s — skipping", DQN_MODEL_PATH)

    if REL_CKPT_DIR.exists():
        RL_AGENTS.append("Relational RL")
        log.info("Relational RL ✓")
    else:
        log.warning("Relational RL checkpoint not found at %s — skipping", REL_CKPT_DIR)

    gtrxl_ok = False
    if INCLUDE_GTRXL:
        gtrxl_ok = _load_gtrxl()
        if gtrxl_ok:
            log.info("GTrXL ✓")

    all_agents = RL_AGENTS + GREEDY_AGENTS + (["GTrXL"] if gtrxl_ok else [])
    log.info("Active agents: %s", all_agents)

    total = len(all_tasks) * len(all_agents)
    pending_rl     = [(a, t) for a in RL_AGENTS     for t in all_tasks if _task_key(a, t) not in done_keys]
    pending_greedy = [(a, t) for a in GREEDY_AGENTS  for t in all_tasks if _task_key(a, t) not in done_keys]
    pending_gtrxl  = [t       for t in all_tasks       if gtrxl_ok and _task_key("GTrXL", t) not in done_keys]

    log.info(
        "Pending — RL: %d | Greedy: %d | GTrXL: %d | Total already done: %d/%d",
        len(pending_rl), len(pending_greedy), len(pending_gtrxl),
        total - len(pending_rl) - len(pending_greedy) - len(pending_gtrxl), total,
    )

    # ── RL agents via GPU thread pool ────────────────────────────────────
    if pending_rl:
        log.info("Starting RL sweep (%d episodes, %d GPU threads)...",
                 len(pending_rl), N_GPU_THREADS)
        _runner = {"DQN": _run_dqn, "Relational RL": _run_relational}
        completed = 0
        with ThreadPoolExecutor(max_workers=N_GPU_THREADS) as pool:
            futures = {
                pool.submit(_runner[a], t["grid_size"], t["num_sensors"], t["seed"]): (a, t)
                for a, t in pending_rl if a in _runner
            }
            for fut in as_completed(futures):
                agent, t = futures[fut]
                g = t["grid_size"]
                try:
                    metrics = fut.result()
                    if metrics:
                        row = {
                            "agent": agent, "seed": t["seed"],
                            "grid_w": g[0], "grid_h": g[1],
                            "n_sensors": t["num_sensors"],
                            "sweep_type": t["sweep_type"],
                            **metrics,
                        }
                        _append_row(row)
                except Exception as exc:
                    log.warning("FAILED %s %s seed=%d: %s", agent, g, t["seed"], exc)
                completed += 1
                if completed % 50 == 0:
                    log.info("RL progress: %d / %d", completed, len(pending_rl))

    # ── Greedy agents via CPU process pool ───────────────────────────────
    if pending_greedy:
        from joblib import Parallel, delayed
        log.info("Starting greedy sweep (%d episodes, %d CPU workers)...",
                 len(pending_greedy), N_CPU_WORKERS)

        greedy_tasks = [
            {
                "agent": a, "grid_size": list(t["grid_size"]),
                "num_sensors": t["num_sensors"], "seed": t["seed"],
                "sweep_type": t["sweep_type"],
                "env_base": ENV_BASE, "max_battery": MAX_BATTERY,
                "src": str(_SRC), "dqn": str(_DQN),
            }
            for a, t in pending_greedy
        ]

        results = Parallel(n_jobs=N_CPU_WORKERS, backend="loky", verbose=5)(
            delayed(_greedy_worker)(task) for task in greedy_tasks
        )

        for task, metrics in zip(greedy_tasks, results):
            if metrics:
                row = {
                    "agent": task["agent"], "seed": task["seed"],
                    "grid_w": task["grid_size"][0], "grid_h": task["grid_size"][1],
                    "n_sensors": task["num_sensors"],
                    "sweep_type": task["sweep_type"],
                    **metrics,
                }
                _append_row(row)

    # ── GTrXL sequential ─────────────────────────────────────────────────
    if pending_gtrxl:
        log.info("Starting GTrXL sweep (%d episodes, sequential)...", len(pending_gtrxl))
        for i, t in enumerate(pending_gtrxl):
            g = t["grid_size"]
            try:
                metrics = _run_gtrxl(g, t["num_sensors"], t["seed"])
                row = {
                    "agent": "GTrXL", "seed": t["seed"],
                    "grid_w": g[0], "grid_h": g[1],
                    "n_sensors": t["num_sensors"],
                    "sweep_type": t["sweep_type"],
                    **metrics,
                }
                _append_row(row)
            except Exception as exc:
                log.warning("GTrXL FAILED %s seed=%d: %s", g, t["seed"], exc)
            if (i + 1) % 20 == 0:
                log.info("GTrXL progress: %d / %d", i + 1, len(pending_gtrxl))

    log.info("Sweep complete. Results → %s", RESULTS_CSV)


if __name__ == "__main__":
    run_sweep()
