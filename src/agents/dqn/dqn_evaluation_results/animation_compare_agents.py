"""
UAV Side-by-Side Live Animation
================================
Shows DQN, SF-Aware Greedy V2, and Nearest Sensor Greedy agents
animated simultaneously in a 3-panel live matplotlib window.

KEY FIX: All three environments share the EXACT same sensor layout.
  - One master environment is seeded and reset to generate positions.
  - Those positions are extracted and hard-injected into every other env.
  - No RNG drift between environments — every panel is identical ground truth.

FIX (DQN AUTO-RESET): VecEnv auto-resets base_env the instant done=True fires
  inside stacked_env.step(). After that call returns, base_env.current_step,
  uav.battery, sensors_visited, and total_data_collected all reflect the NEW
  episode (step=0). Fix: capture every piece of display state BEFORE the step
  call, and use those pre-step values when done=True is detected.

Usage:
    python uav_side_by_side_animation.py

Tune at the top of the file:
    ANIMATION_STEP_DELAY   — seconds per step  (lower = faster)
    RENDER_EVERY_N_STEPS   — redraw frequency  (higher = faster but choppier)
"""

import sys
import numpy as np
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import ieee_style
ieee_style.apply()

# ==================== PATH SETUP ====================
script_dir         = Path(__file__).resolve().parent
src_dir            = script_dir.parent.parent.parent   # up to src/
script_dir_results = script_dir.parent                 # dqn/

sys.path.insert(0, str(src_dir))

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from environment.uav_env import UAVEnvironment
from greedy_agents import MaxThroughputGreedyV2, NearestSensorGreedy

# ==================== CONFIG ====================
PLOT_CONFIG = {
    "grid_size":          (500, 500),
    "num_sensors":        20,
    "max_steps":          2100,
    "path_loss_exponent": 3.8,
    "rssi_threshold": -85.0,
    "sensor_duty_cycle":  10.0,
    "seed":               42,
}

EVAL_MAX_BATTERY     = 274.0
ANIMATION_STEP_DELAY = 0.02   # seconds between steps (lower = faster)
RENDER_EVERY_N_STEPS = 1      # redraw every N steps  (higher = faster/choppier)

DQN_MODEL_PATH = (
    script_dir_results / "models" / "dqn_domain_rand" / "dqn_final.zip"
)
DQN_CONFIG_PATH = (
    script_dir_results / "models" / "dqn_domain_rand" / "frame_stacking_config.json"
)

# ==================== VISUAL THEME ====================
COLORS = {
    "dqn":            "#1b9e77",
    "smart_greedy":   "#d95f02",
    "dumb_greedy":    "#7570b3",
    "sensor":         "#FF6F00",    # amber  = unvisited
    "sensor_visited": "#00C853",    # green  = visited
    "uav":            "#FFFFFF",
    "bg":             "#0D1117",
    "grid":           "#1A2332",
    "text":           "#E0E0E0",
    "border_dqn":     "#1b9e77",
    "border_smart":   "#d95f02",
    "border_dumb":    "#7570b3",
}

AGENT_META = [
    # (display label,           path colour,            border colour)
    ("DQN Agent\n(Proposed)",   COLORS["dqn"],          COLORS["border_dqn"]),
    ("SF-Aware Greedy V2",      COLORS["smart_greedy"], COLORS["border_smart"]),
    ("Nearest Sensor Greedy",   COLORS["dumb_greedy"],  COLORS["border_dumb"]),
]


# ==================== ENVIRONMENT WRAPPERS ====================

class SnapshotUAVEnv(UAVEnvironment):
    """Saves sensor telemetry before each reset (used by DQN)."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_sensor_data = None

    def reset(self, **kwargs):
        if hasattr(self, "sensors") and self.current_step > 0:
            self.last_sensor_data = [
                {
                    "sensor_id":              s.sensor_id,
                    "position":               tuple(s.position),
                    "total_data_transmitted": float(s.total_data_transmitted),
                }
                for s in self.sensors
            ]
        return super().reset(**kwargs)


class FixedLayoutEnv(UAVEnvironment):
    """
    UAVEnvironment whose sensor positions are always overwritten with
    `fixed_positions` immediately after each parent reset().
    This guarantees the same map regardless of RNG state.
    """
    def __init__(self, fixed_positions, **kwargs):
        self._fixed_positions = fixed_positions
        super().__init__(**kwargs)

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        for sensor, pos in zip(self.sensors, self._fixed_positions):
            sensor.position = np.array(pos, dtype=np.float32)
        return obs


class FixedLayoutSnapshotEnv(SnapshotUAVEnv):
    """FixedLayout + Snapshot combined — for the DQN agent."""
    def __init__(self, fixed_positions, **kwargs):
        self._fixed_positions = fixed_positions
        super().__init__(**kwargs)

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        for sensor, pos in zip(self.sensors, self._fixed_positions):
            sensor.position = np.array(pos, dtype=np.float32)
        return obs


# ==================== HELPERS ====================

def load_frame_stacking_config(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"use_frame_stacking": True, "n_stack": 4}


def _base_env_kwargs():
    return {
        "grid_size":          PLOT_CONFIG["grid_size"],
        "num_sensors":        PLOT_CONFIG["num_sensors"],
        "max_steps":          PLOT_CONFIG["max_steps"],
        "path_loss_exponent": PLOT_CONFIG["path_loss_exponent"],
        "rssi_threshold":     PLOT_CONFIG["rssi_threshold"],
        "sensor_duty_cycle":  PLOT_CONFIG["sensor_duty_cycle"],
        "render_mode":        None,
    }


def get_canonical_sensor_positions(seed):
    """
    Seed one throwaway environment and extract its sensor positions.
    All three real environments will be forced to use these exact positions.
    """
    master = UAVEnvironment(**_base_env_kwargs())
    master.reset(seed=seed)
    positions = [tuple(float(v) for v in s.position) for s in master.sensors]
    master.close()
    print(f"  ✓ Canonical layout: {len(positions)} sensors from seed={seed}")
    for i, p in enumerate(positions):
        print(f"    S{i:02d}: ({p[0]:.1f}, {p[1]:.1f})")
    return positions


def build_envs(fixed_positions, seed):
    """Build three environments with identical sensor layout, all reset."""
    kwargs = _base_env_kwargs()

    dqn_base  = FixedLayoutSnapshotEnv(fixed_positions, **kwargs)
    dqn_base.reset(seed=seed)

    env_smart = FixedLayoutEnv(fixed_positions, **kwargs)
    env_smart.reset(seed=seed)

    env_dumb  = FixedLayoutEnv(fixed_positions, **kwargs)
    env_dumb.reset(seed=seed)

    # Sanity check
    p0_dqn   = tuple(dqn_base.sensors[0].position)
    p0_smart = tuple(env_smart.sensors[0].position)
    p0_dumb  = tuple(env_dumb.sensors[0].position)
    match = (p0_dqn == p0_smart == p0_dumb)
    print(f"\n  Layout match check — sensor[0] positions:")
    print(f"    DQN   : {p0_dqn}")
    print(f"    Smart : {p0_smart}")
    print(f"    Dumb  : {p0_dumb}")
    print(f"    Match : {'✓ YES' if match else '✗ NO — check FixedLayout reset order'}")

    return dqn_base, env_smart, env_dumb


# ==================== ANIMATOR ====================

class SideBySideAnimator:
    """
    3-panel trajectory view + shared cumulative-reward strip.
    Sensor squares flip amber → green as the UAV visits them.
    """

    def __init__(self, envs, titles):
        self.envs   = envs
        self.titles = titles
        self.n      = len(envs)
        self.grid   = PLOT_CONFIG["grid_size"][0]

        self.path_x      = [[] for _ in range(self.n)]
        self.path_y      = [[] for _ in range(self.n)]
        self.cum_rewards = [[] for _ in range(self.n)]
        self.steps_log   = [[] for _ in range(self.n)]

        # ---- Figure ----
        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(20, 11), facecolor=COLORS["bg"])
        self.fig.suptitle(
            "UAV Data Collection — Live Agent Comparison\n"
            "(identical sensor layout · same environment · different policies)",
            fontsize=13, fontweight="bold", color=COLORS["text"], y=0.995
        )

        gs = gridspec.GridSpec(
            2, 3, figure=self.fig,
            height_ratios=[3, 1],
            hspace=0.38, wspace=0.14,
            left=0.05, right=0.97, top=0.93, bottom=0.06
        )
        self.traj_axes = [self.fig.add_subplot(gs[0, i]) for i in range(3)]
        self.reward_ax = self.fig.add_subplot(gs[1, :])

        for i, ax in enumerate(self.traj_axes):
            self._style_traj_ax(ax, i)
        self._style_reward_ax()

        # ---- Artists ----
        self.path_lines      = []
        self.uav_dots        = []
        self.sensor_scatters = []
        self.hud_texts       = []
        self.reward_lines    = []

        for i, (env, (label, color, border)) in enumerate(zip(self.envs, self.titles)):
            ax = self.traj_axes[i]

            line, = ax.plot([], [], color=color, linewidth=1.8,
                            alpha=0.75, zorder=2, solid_capstyle='round')
            self.path_lines.append(line)

            dot, = ax.plot([], [], 'o', color=COLORS["uav"], markersize=11,
                           markeredgecolor=color, markeredgewidth=2.5, zorder=5)
            self.uav_dots.append(dot)

            sensor_pos = np.array([s.position for s in env.sensors])
            sc = ax.scatter(
                sensor_pos[:, 0], sensor_pos[:, 1],
                s=130, c=[COLORS["sensor"]] * len(sensor_pos),
                marker='s', edgecolors='#BF360C', linewidth=1.2, zorder=3
            )
            self.sensor_scatters.append(sc)

            txt = ax.text(
                0.02, 0.98, "", transform=ax.transAxes,
                fontsize=8.5, color=COLORS["text"],
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5',
                          facecolor='#111827', edgecolor=color,
                          alpha=0.88, linewidth=1.5),
                zorder=6
            )
            self.hud_texts.append(txt)

            rl, = self.reward_ax.plot(
                [], [], color=color, linewidth=2.2,
                label=label.replace('\n', ' '), zorder=3
            )
            self.reward_lines.append(rl)

        self.reward_ax.legend(
            loc='upper left', fontsize=9,
            facecolor='#111827', edgecolor='#333', labelcolor=COLORS["text"]
        )

        plt.ion()
        plt.show()

    def _style_traj_ax(self, ax, idx):
        label, color, border = self.titles[idx]
        ax.set_facecolor(COLORS["grid"])
        ax.set_xlim(0, self.grid)
        ax.set_ylim(0, self.grid)
        ax.set_aspect('equal')
        ax.tick_params(colors=COLORS["text"], labelsize=7)
        ax.set_xlabel("X (m)", fontsize=8, color=COLORS["text"])
        ax.set_ylabel("Y (m)", fontsize=8, color=COLORS["text"])
        ax.set_title(label, fontsize=11, fontweight='bold', color=color, pad=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(border)
            spine.set_linewidth(2.5)
        ax.grid(True, color='#1E3040', linewidth=0.5, linestyle='--', alpha=0.7)

    def _style_reward_ax(self):
        ax = self.reward_ax
        ax.set_facecolor(COLORS["grid"])
        ax.set_xlabel("Step", fontsize=9, color=COLORS["text"])
        ax.set_ylabel("Cumulative Reward", fontsize=9, color=COLORS["text"])
        ax.tick_params(colors=COLORS["text"], labelsize=7)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')
        ax.grid(True, color='#1E3040', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.set_title("Cumulative Reward Comparison", fontsize=10,
                     color=COLORS["text"], pad=4)

    def update_agent(self, idx, x, y, step, cum_reward,
                     battery_pct, coverage_pct, data_bytes, sensors_visited_ids):
        """
        sensors_visited_ids: a set/list of sensor_id values that have been visited.
        Accepts either the live env.sensors_visited set OR a frozen copy passed in
        from pre-step capture (used by DQN on the done step).
        """
        self.path_x[idx].append(x)
        self.path_y[idx].append(y)
        self.cum_rewards[idx].append(cum_reward)
        self.steps_log[idx].append(step)

        self.path_lines[idx].set_data(self.path_x[idx], self.path_y[idx])
        self.uav_dots[idx].set_data([x], [y])

        colours = [
            COLORS["sensor_visited"] if s.sensor_id in sensors_visited_ids
            else COLORS["sensor"]
            for s in self.envs[idx].sensors
        ]
        self.sensor_scatters[idx].set_facecolor(colours)

        self.hud_texts[idx].set_text(
            f"Step     : {step:>5}\n"
            f"Battery  : {battery_pct:>5.1f}%\n"
            f"Coverage : {coverage_pct:>5.1f}%\n"
            f"Data     : {data_bytes / 1e6:>6.2f} MB\n"
            f"Reward   : {cum_reward:>9.0f}"
        )
        self.reward_lines[idx].set_data(self.steps_log[idx], self.cum_rewards[idx])

    def redraw(self):
        all_steps   = [s for sl in self.steps_log   for s in sl]
        all_rewards = [r for rl in self.cum_rewards  for r in rl]
        if all_steps:
            self.reward_ax.set_xlim(0, max(all_steps) + 50)
        if all_rewards:
            rmin, rmax = min(all_rewards), max(all_rewards)
            pad = max(abs(rmax - rmin) * 0.1, 1)
            self.reward_ax.set_ylim(rmin - pad, rmax + pad)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def mark_done(self, idx):
        _, color, _ = self.titles[idx]
        self.traj_axes[idx].text(
            0.5, 0.5, "DONE ✓",
            transform=self.traj_axes[idx].transAxes,
            fontsize=22, fontweight='bold',
            color=color, alpha=0.35,
            ha='center', va='center', rotation=30
        )
        self.redraw()

    def keep_open(self):
        plt.ioff()
        plt.tight_layout()
        plt.show(block=True)


# ==================== MAIN ====================

def run_animation(seed=PLOT_CONFIG["seed"]):
    print("=" * 70)
    print("UAV SIDE-BY-SIDE LIVE ANIMATION")
    print("All three agents run on the EXACT SAME sensor layout")
    print("=" * 70)

    # 1. Extract one canonical layout from a seeded master env
    print("\n[1] Generating canonical sensor layout...")
    fixed_positions = get_canonical_sensor_positions(seed)

    # 2. Load DQN
    dqn_model       = None
    fs_config       = {"use_frame_stacking": True, "n_stack": 4}
    dqn_stacked_env = None

    print("\n[2] Loading DQN model...")
    if DQN_MODEL_PATH.exists():
        dqn_model = DQN.load(DQN_MODEL_PATH)
        fs_config = load_frame_stacking_config(DQN_CONFIG_PATH)
        print(f"  ✓ Loaded | frame_stacking={fs_config}")
    else:
        print(f"  ⚠ Not found — DQN panel will stay blank")

    # 3. Build all three environments with shared layout
    print("\n[3] Building environments...")
    dqn_base, env_smart, env_dumb = build_envs(fixed_positions, seed)

    # 4. Wrap DQN env
    dqn_obs = None
    if dqn_model is not None:
        print("\n[4] Wrapping DQN env in VecFrameStack...")
        _fp = fixed_positions
        _kw = _base_env_kwargs()

        def _make_env():
            e = FixedLayoutSnapshotEnv(_fp, **_kw)
            return e

        vec = DummyVecEnv([_make_env])
        if fs_config.get("use_frame_stacking", True):
            dqn_stacked_env = VecFrameStack(vec, n_stack=fs_config.get("n_stack", 4))
        else:
            dqn_stacked_env = vec

        dqn_obs  = dqn_stacked_env.reset()
        dqn_base = vec.envs[0]          # reference to actual base env

    # 5. Init greedy agents
    obs_smart, _ = env_smart.reset(seed=seed)
    obs_dumb,  _ = env_dumb.reset(seed=seed)

    agent_smart = MaxThroughputGreedyV2(env_smart)
    agent_dumb  = NearestSensorGreedy(env_dumb)

    # 6. Create animator
    envs = [dqn_base, env_smart, env_dumb]
    anim = SideBySideAnimator(envs, AGENT_META)

    # 7. Simulation loop
    cum_rewards = [0.0, 0.0, 0.0]
    done_flags  = [dqn_model is None, False, False]
    if dqn_model is None:
        anim.mark_done(0)

    step_counter = 0
    last_render  = time.time()

    print("\nAnimating... (Ctrl+C to stop)\n")

    try:
        while not all(done_flags):
            step_counter += 1

            # ----------------------------------------------------------------
            # DQN
            # ----------------------------------------------------------------
            # ROOT CAUSE: VecEnv auto-resets base_env INSIDE stacked_env.step().
            # By the time step() returns with done=True, base_env already holds
            # the NEW episode state (step=0, full battery, empty visited set).
            # FIX: snapshot every piece of display state BEFORE the step call,
            # then use those pre-step values unconditionally when done=True.
            # ----------------------------------------------------------------
            if not done_flags[0]:
                action, _ = dqn_model.predict(dqn_obs, deterministic=True)
                av = int(action[0]) if hasattr(action, '__len__') else int(action)

                # ── PRE-STEP capture — before auto-reset can fire ──────────
                pre_x         = float(dqn_base.uav.position[0])
                pre_y         = float(dqn_base.uav.position[1])
                pre_step      = int(dqn_base.current_step)
                pre_bat_pct   = float(dqn_base.uav.get_battery_percentage())
                pre_n_sensors = int(dqn_base.num_sensors)
                pre_visited   = set(dqn_base.sensors_visited)   # frozen copy
                pre_cov       = (len(pre_visited) / pre_n_sensors) * 100
                pre_data      = float(dqn_base.total_data_collected)
                # ───────────────────────────────────────────────────────────

                dqn_obs, rwds, dns, _ = dqn_stacked_env.step([av])
                cum_rewards[0] += float(rwds[0])
                done_now = bool(dns[0])

                if done_now:
                    # base_env has already been auto-reset here —
                    # use ONLY the pre-step snapshot.
                    anim.update_agent(
                        0,
                        pre_x, pre_y,
                        pre_step,
                        cum_rewards[0],
                        pre_bat_pct,
                        pre_cov,
                        pre_data,
                        pre_visited,
                    )
                    done_flags[0] = True
                    anim.mark_done(0)
                    print(
                        f"  [DQN]          DONE  step={pre_step}"
                        f"  reward={cum_rewards[0]:.0f}"
                        f"  cov={pre_cov:.1f}%"
                        f"  battery={pre_bat_pct:.1f}%"
                    )
                else:
                    # Normal mid-episode update — read live state directly.
                    x   = float(dqn_base.uav.position[0])
                    y   = float(dqn_base.uav.position[1])
                    bat = float(dqn_base.uav.get_battery_percentage())
                    cov = (len(dqn_base.sensors_visited) / dqn_base.num_sensors) * 100
                    dat = float(dqn_base.total_data_collected)
                    anim.update_agent(
                        0,
                        x, y,
                        dqn_base.current_step,
                        cum_rewards[0],
                        bat, cov, dat,
                        dqn_base.sensors_visited,
                    )

            # ----------------------------------------------------------------
            # Smart Greedy
            # ----------------------------------------------------------------
            if not done_flags[1]:
                action = agent_smart.select_action(obs_smart)
                obs_smart, rwd, done, trunc, _ = env_smart.step(action)
                cum_rewards[1] += rwd
                x, y = env_smart.uav.position[0], env_smart.uav.position[1]
                bat  = env_smart.uav.get_battery_percentage()
                cov  = (len(env_smart.sensors_visited) / env_smart.num_sensors) * 100
                dat  = env_smart.total_data_collected
                anim.update_agent(1, x, y, env_smart.current_step,
                                  cum_rewards[1], bat, cov, dat,
                                  env_smart.sensors_visited)
                if done or trunc:
                    done_flags[1] = True
                    anim.mark_done(1)
                    print(f"  [Smart Greedy] DONE  step={env_smart.current_step}"
                          f"  reward={cum_rewards[1]:.0f}  cov={cov:.1f}%")

            # ----------------------------------------------------------------
            # Dumb Greedy
            # ----------------------------------------------------------------
            if not done_flags[2]:
                action = agent_dumb.select_action(obs_dumb)
                obs_dumb, rwd, done, trunc, _ = env_dumb.step(action)
                cum_rewards[2] += rwd
                x, y = env_dumb.uav.position[0], env_dumb.uav.position[1]
                bat  = env_dumb.uav.get_battery_percentage()
                cov  = (len(env_dumb.sensors_visited) / env_dumb.num_sensors) * 100
                dat  = env_dumb.total_data_collected
                anim.update_agent(2, x, y, env_dumb.current_step,
                                  cum_rewards[2], bat, cov, dat,
                                  env_dumb.sensors_visited)
                if done or trunc:
                    done_flags[2] = True
                    anim.mark_done(2)
                    print(f"  [Dumb Greedy]  DONE  step={env_dumb.current_step}"
                          f"  reward={cum_rewards[2]:.0f}  cov={cov:.1f}%")

            # ----------------------------------------------------------------
            # Render
            # ----------------------------------------------------------------
            if step_counter % RENDER_EVERY_N_STEPS == 0:
                anim.redraw()
                elapsed = time.time() - last_render
                sleep_t = max(0.0, ANIMATION_STEP_DELAY - elapsed)
                if sleep_t > 0:
                    time.sleep(sleep_t)
                last_render = time.time()

    except KeyboardInterrupt:
        print("\nStopped by user.")

    anim.redraw()

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    for lbl, r in zip(["DQN Agent", "Smart Greedy V2", "Nearest Greedy"], cum_rewards):
        print(f"  {lbl:<22}: {r:>12.0f}")
    print("=" * 70)
    print("Close the window to exit.")

    for env in [env_smart, env_dumb]:
        env.close()
    if dqn_stacked_env:
        dqn_stacked_env.close()

    anim.keep_open()


if __name__ == "__main__":
    run_animation(seed=PLOT_CONFIG["seed"])