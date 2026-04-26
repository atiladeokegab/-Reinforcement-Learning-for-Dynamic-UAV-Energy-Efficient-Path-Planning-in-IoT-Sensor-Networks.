"""
Ablation study: Full Relational RL vs Relational Ablation Baseline.

Evaluates both models on the same neutral InferenceRelationalUAVEnv
(raw UAVEnvironment rewards, no dwell bonus, no potential shaping)
across N_SEEDS seeds, then reports Welch's t-test and Cohen's d for
NDR, Jain's fairness, Gini coefficient, and cumulative reward.

Usage:
    PYTHONIOENCODING=utf-8 uv run python src/agents/dqn/dqn_evaluation_results/ablation_study.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parent.parent.parent
_ROOT = _SRC.parent
for _p in (str(_SRC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agents.dqn.dqn_evaluation_results.relational_rl_runner import (
    InferenceRelationalUAVEnv,
    load_relational_rl_module,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
_MODELS = _SRC / "agents" / "dqn" / "models"

FULL_CHECKPOINT = (
    _MODELS / "relational_rl" / "results" / "checkpoints" / "stage_4" / "final"
)
ABLATION_CHECKPOINT = (
    _MODELS / "relational_rl_ablation" / "checkpoints" / "stage_4" / "final"
)

OUT_DIR = _HERE / "baseline_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Eval config ───────────────────────────────────────────────────────────────
N_SEEDS      = 20
GRID_SIZE    = (500, 500)
NUM_SENSORS  = 20
MAX_BATTERY  = 274.0
MAX_STEPS    = 2100
N_MAX        = 50


# ── Helpers ───────────────────────────────────────────────────────────────────

def _jains(values: list[float]) -> float:
    arr = np.array(values, dtype=float)
    if arr.sum() == 0:
        return 0.0
    return float(arr.sum() ** 2 / (len(arr) * (arr ** 2).sum()))


def _gini(values: list[float]) -> float:
    arr = np.sort(np.array(values, dtype=float))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * arr).sum()) / (n * arr.sum()) - (n + 1) / n)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled_std = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    return float((a.mean() - b.mean()) / pooled_std) if pooled_std > 0 else 0.0


def _effect_label(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def run_episode(rl_module, seed: int) -> dict:
    import torch
    from ray.rllib.core.columns import Columns

    env = InferenceRelationalUAVEnv(
        n_max=N_MAX,
        grid_size=GRID_SIZE,
        num_sensors=NUM_SENSORS,
        max_battery=MAX_BATTERY,
        max_steps=MAX_STEPS,
        include_sensor_positions=True,
    )
    obs, _ = env.reset(seed=seed)

    cumulative_reward = 0.0
    while True:
        batch = {Columns.OBS: {
            k: torch.as_tensor(np.asarray(v)).unsqueeze(0)
            for k, v in obs.items()
        }}
        with torch.no_grad():
            out = rl_module._forward_inference(batch)
        logits = out[Columns.ACTION_DIST_INPUTS]
        action = int(torch.argmax(logits, dim=-1).item())

        obs, reward, terminated, truncated, _ = env.step(action)
        cumulative_reward += float(reward)

        if terminated or truncated:
            break

    cr_list = [
        s.total_data_transmitted / max(s.total_data_generated, 1e-9)
        for s in env.sensors
    ]
    ndr = (
        sum(s.total_data_transmitted for s in env.sensors)
        / max(sum(s.total_data_generated for s in env.sensors), 1e-9)
    )
    energy_used = MAX_BATTERY - env.uav.battery
    efficiency  = env.total_data_collected / energy_used if energy_used > 0 else 0.0

    return {
        "cumulative_reward": cumulative_reward,
        "ndr":               ndr,
        "jains":             _jains(cr_list),
        "gini":              _gini(cr_list),
        "efficiency":        efficiency,
    }


def evaluate_model(label: str, checkpoint: Path, seeds: list[int]) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"Checkpoint: {checkpoint}")
    print(f"{'='*60}")
    module = load_relational_rl_module(checkpoint)
    rows = []
    for i, seed in enumerate(seeds):
        result = run_episode(module, seed)
        result["seed"] = seed
        rows.append(result)
        print(
            f"  seed {seed:>3}: reward={result['cumulative_reward']:>10.2f}  "
            f"NDR={result['ndr']:.3f}  Jain={result['jains']:.3f}  "
            f"Gini={result['gini']:.3f}"
        )
    df = pd.DataFrame(rows)
    df["model"] = label
    return df


def print_stats(label: str, df: pd.DataFrame) -> None:
    print(f"\n{label} — summary (n={len(df)})")
    for col in ("cumulative_reward", "ndr", "jains", "gini", "efficiency"):
        print(f"  {col:22s}: {df[col].mean():.4f} ± {df[col].std(ddof=1):.4f}")


def welch_test(col: str, full: pd.DataFrame, ablation: pd.DataFrame) -> dict:
    a = full[col].values
    b = ablation[col].values
    t, p = stats.ttest_ind(a, b, equal_var=False)
    d = _cohens_d(a, b)
    return {"metric": col, "full_mean": a.mean(), "full_std": a.std(ddof=1),
            "ablation_mean": b.mean(), "ablation_std": b.std(ddof=1),
            "t": t, "p": p, "cohens_d": d, "effect": _effect_label(d)}


def plot_comparison(full_df: pd.DataFrame, ablation_df: pd.DataFrame) -> None:
    metrics = [
        ("ndr",               "NDR"),
        ("jains",             "Jain's Fairness"),
        ("gini",              "Gini Coefficient"),
        ("cumulative_reward", "Cumulative Reward"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4))
    colors = {"Full Model": "#1f77b4", "Relational Ablation": "#d62728"}

    for ax, (col, title) in zip(axes, metrics):
        for label, df in [("Full Model", full_df), ("Relational Ablation", ablation_df)]:
            vals = df[col].values
            ax.bar(label, vals.mean(), yerr=vals.std(ddof=1),
                   color=colors[label], capsize=5, alpha=0.85,
                   label=label, width=0.5)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.tick_params(axis="y", labelsize=8)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors["Full Model"]),
        plt.Rectangle((0, 0), 1, 1, color=colors["Relational Ablation"]),
    ]
    fig.legend(handles, ["Full Model", "Relational Ablation"],
               loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Ablation Study: Full Model vs Relational Ablation Baseline\n"
                 f"(500×500, N=20, {N_SEEDS} seeds)", fontsize=11)
    plt.tight_layout()
    out = OUT_DIR / "ablation_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out}")
    plt.close()


def main() -> None:
    seeds = list(range(N_SEEDS))

    full_df     = evaluate_model("Full Model",          FULL_CHECKPOINT,     seeds)
    ablation_df = evaluate_model("Relational Ablation", ABLATION_CHECKPOINT, seeds)

    print_stats("Full Model",          full_df)
    print_stats("Relational Ablation", ablation_df)

    print("\n" + "="*70)
    print("Welch's t-test (Full Model vs Relational Ablation)")
    print("="*70)
    results = []
    for col in ("cumulative_reward", "ndr", "jains", "gini", "efficiency"):
        r = welch_test(col, full_df, ablation_df)
        results.append(r)
        sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else "ns"
        print(
            f"  {r['metric']:22s}: Full={r['full_mean']:.4f}±{r['full_std']:.4f}  "
            f"Abl={r['ablation_mean']:.4f}±{r['ablation_std']:.4f}  "
            f"t={r['t']:+.3f}  p={r['p']:.4f}{sig}  "
            f"d={r['cohens_d']:+.3f} ({r['effect']})"
        )

    stats_df = pd.DataFrame(results)
    csv_path = OUT_DIR / "ablation_study_results.csv"
    stats_df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")

    plot_comparison(full_df, ablation_df)


if __name__ == "__main__":
    main()
