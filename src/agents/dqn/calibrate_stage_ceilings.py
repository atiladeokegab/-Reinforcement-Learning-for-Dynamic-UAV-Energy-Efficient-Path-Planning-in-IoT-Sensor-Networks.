"""
Stage-ceiling calibration for the competence gate.

Runs MaxThroughputGreedyV2 on 300x300, 400x400, 500x500 with N=20 sensors
to measure the physical NDR/Jain's ceiling achievable under the current
2,100-step budget. Use the output to set STAGE_GATES per-stage overrides.

Also runs N=40 on the same grids — the worst-case worker — so the gate can
be tuned to the weakest config that actually feeds the rolling window.

Usage:
    PYTHONIOENCODING=utf-8 uv run python src/agents/dqn/calibrate_stage_ceilings.py
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np

import dqn as _dqn
from environment.uav_env import UAVEnvironment

sys.path.insert(0, str(Path(__file__).parent / "dqn_evaluation_results"))
from greedy_agents import MaxThroughputGreedyV2  # noqa: E402


GRIDS = [(300, 300), (400, 400), (500, 500)]
SENSOR_COUNTS = [20, 40]
N_EPISODES = 50


def run(grid, n_sensors, n_episodes):
    ndrs, jains = [], []
    min_dist = _dqn.NAV_CONFIG["min_start_dist"]

    for ep in range(n_episodes):
        env = UAVEnvironment(
            grid_size=grid,
            num_sensors=n_sensors,
            **_dqn.BASE_ENV_CONFIG,
        )
        env.reset(seed=ep)

        W, H = float(grid[0]), float(grid[1])
        rng = np.random.default_rng(ep + 99991)
        s_pos = [s.position for s in env.sensors]
        best_pos, best_d = env.uav.position.copy(), -1.0
        for _ in range(_dqn.NAV_CONFIG["max_start_tries"]):
            candidate = np.array(
                [float(rng.uniform(0.05 * W, 0.95 * W)),
                 float(rng.uniform(0.05 * H, 0.95 * H))],
                dtype=np.float32,
            )
            d = float(min(np.linalg.norm(candidate - sp) for sp in s_pos))
            if d > best_d:
                best_d, best_pos = d, candidate
            if d >= min_dist:
                break
        env.uav.position = best_pos
        env.uav.start_position = best_pos
        obs = env._get_observation()

        agent = MaxThroughputGreedyV2(env)
        done = False
        while not done:
            action = agent.select_action(obs)
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc

        ndr = len(env.sensors_visited) / env.num_sensors * 100
        rates = [
            s.total_data_transmitted / s.total_data_generated * 100
            for s in env.sensors if s.total_data_generated > 0
        ]
        ndrs.append(ndr)
        jains.append(_dqn.DomainRandEnv._jains(rates))
        env.close()

    return float(np.mean(ndrs)), float(np.std(ndrs)), float(np.mean(jains)), float(np.std(jains))


def main():
    print("=" * 78)
    print("Greedy ceiling calibration — MaxThroughputGreedyV2, {} episodes per cell".format(N_EPISODES))
    print("Step budget: {} timesteps".format(_dqn.BASE_ENV_CONFIG["max_steps"]))
    print("=" * 78)
    print("{:>10s}  {:>3s}   {:>14s}   {:>14s}".format("Grid", "N", "NDR%", "Jain"))
    print("-" * 78)
    for grid in GRIDS:
        for n in SENSOR_COUNTS:
            ndr_m, ndr_s, j_m, j_s = run(grid, n, N_EPISODES)
            print("{:>10s}  {:>3d}   {:>6.1f} ± {:>4.1f}   {:>6.3f} ± {:>5.3f}".format(
                "{}x{}".format(grid[0], grid[1]), n, ndr_m, ndr_s, j_m, j_s,
            ))
    print("=" * 78)
    print("Set STAGE_GATES[stage] = {'ndr_pct': <mean_NDR+2>, 'jains': <mean_Jain+0.02>}")


if __name__ == "__main__":
    main()