"""
Full 3M-step retrain — Competence-Based (Self-Adaptive) Curriculum
===================================================================

What changed from the time-based version
-----------------------------------------
  • CURRICULUM_THRESHOLDS removed entirely — the time-based gates are gone.
  • Graduation is now driven by performance milestones set in COMPETENCE_GATE.
  • Demotion support is wired through DEMOTION_GATE.
  • N_EVAL_EPISODES = 100 for a true Monte Carlo cross-validation pass.
  • Per-stage target KPIs are defined here for future difficulty tuning.
  • A "Time to Graduation" summary is printed after training.

Competence gate (defaults, overridden below)
--------------------------------------------
  Stage graduates when BOTH hold over a rolling window of 50 episodes:
    • NDR   >= 95 %    (New Discovery Rate — % of sensors visited per episode)
    • Jain's >= 0.85   (fairness index)
  AND at least min_steps = 500,000 timesteps have elapsed in the current stage.

Demotion gate
-------------
  Stage is reverted if BOTH drop below their demotion thresholds after
  at least 30 episodes in the new stage:
    • NDR   < 70 %
    • Jain's < 0.60

Per-stage target KPIs (for future tuning)
-----------------------------------------
  These are reference targets only — tighten ndr_pct / jains in COMPETENCE_GATE
  or per-stage overrides (not yet implemented) to raise the difficulty bar.

  Stage 0 (100×100)  — NDR ≥ 97 %,  Jain ≥ 0.90  (small grid, should be easy)
  Stage 1 (200×200)  — NDR ≥ 96 %,  Jain ≥ 0.88
  Stage 2 (300×300)  — NDR ≥ 95 %,  Jain ≥ 0.86
  Stage 3 (400×400)  — NDR ≥ 95 %,  Jain ≥ 0.85
  Stage 4 (500×500)  — deployment target; no graduation (final stage)

Saves to: models/dqn_v3_retrain/

Bug fixes inherited from retrain branch
----------------------------------------
  Fix 1 — Urgency bug: sensor_urgency now uses _calculate_urgency() [0,1]
  Fix 2 — penalty_unvisited raised -15,000 → -100,000
  Fix 3 — penalty_starved = -30,000 for sensors with CR < 20% at episode end
  Fix 4 — IoTSensor.reset() initialises total_data_generated = initial_buffer_fill
"""

from pathlib import Path
import dqn as _dqn

# ── Output directories ────────────────────────────────────────────────────────
_dqn.SAVE_DIR = Path("models/dqn_v3_retrain")
_dqn.LOG_DIR  = Path("logs/dqn_v3_retrain")
_dqn.SAVE_DIR.mkdir(parents=True, exist_ok=True)
_dqn.LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Remove time-based thresholds — competence gate drives graduation ──────────
# Setting this to an empty list ensures the legacy CurriculumCallback branch
# (which checks CURRICULUM_THRESHOLDS) is never triggered.
_dqn.CURRICULUM_THRESHOLDS = []

# ── Competence Gate ───────────────────────────────────────────────────────────
# Both milestones must be sustained over `window` episodes to graduate.
# Increase ndr_pct / jains to make stages harder to pass.
# Increase min_steps to force longer exposure to each difficulty level.
_dqn.COMPETENCE_GATE = {
    "ndr_pct":   95.0,    # % of sensors visited per episode (rolling mean)
    "jains":     0.85,    # Jain's fairness index (rolling mean)
    "window":    50,      # rolling average over last N episodes
    "min_steps": 500_000, # minimum timesteps in stage before graduation is allowed
                          # 500k × 5 stages ≤ 3M budget, with headroom for demotion
}

# ── Demotion Gate ─────────────────────────────────────────────────────────────
# Reverts one stage if performance collapses after an advancement.
_dqn.DEMOTION_GATE = {
    "ndr_pct":      70.0,  # rolling NDR below this ...
    "jains":        0.60,  # ... AND rolling Jain's below this triggers demotion
    "min_episodes": 30,    # must have at least this many episodes in new stage
}

# ── Navigation fixes (Fix 1 + Fix 2) ─────────────────────────────────────────
_dqn.NAV_CONFIG = {
    "min_start_dist":  50.0,   # UAV must start ≥ 50 units from every sensor
    "max_start_tries": 200,
    "prox_eta":        2.0,    # shaping gain η
}

# ── Greedy benchmark gate — tighter margin for the full retrain ───────────────
_dqn.GREEDY_BENCHMARK = {
    "enabled":      True,
    "n_episodes":   50,    # 50 episodes → stable greedy baseline estimate
    "sensor_count": 20,
    "margin_ndr":   3.0,   # DQN must beat greedy NDR by 3 pp (threshold capped at 98%)
    "margin_jains": 0.03,  # DQN must beat greedy Jain's by 0.03 (threshold capped at 0.97)
    "floor_ndr":    70.0,  # hard minimum even if greedy is weak
    "floor_jains":  0.55,
}

# ── Eval settings — 100 episodes for Monte Carlo cross-validation ─────────────
# 100 episodes × 2,100 steps = 210,000 steps per eval pass.
# With EVAL_FREQ = 25,000 that is ~12 eval passes over 3M steps — acceptable overhead.
_dqn.EVAL_FREQ       = 25_000
_dqn.N_EVAL_EPISODES = 100

# ── Per-stage target KPIs (reference only — not programmatically enforced yet) ─
# Tighten COMPETENCE_GATE["ndr_pct"] / ["jains"] above to raise the difficulty bar,
# or implement per-stage overrides in CurriculumCallback._try_advance_stage().
STAGE_TARGET_KPIS = {
    0: {"description": "100×100 only",          "ndr_pct": 97.0, "jains": 0.90},
    1: {"description": "up to 200×200",          "ndr_pct": 96.0, "jains": 0.88},
    2: {"description": "up to 300×300",          "ndr_pct": 95.0, "jains": 0.86},
    3: {"description": "up to 400×400",          "ndr_pct": 95.0, "jains": 0.85},
    4: {"description": "full range (500×500)",   "ndr_pct": None, "jains": None},  # final stage
}


# ── 4090 overrides (24 GB VRAM — remove these if running on laptop GPU) ──────
_dqn.N_ENVS               = 8
_dqn.WORKER_SENSOR_COUNTS = [10, 15, 20, 25, 30, 35, 40, 40]  # 8 workers, diverse coverage
_dqn.HYPERPARAMS["buffer_size"] = 500_000   # was 150k — ~2.3 GB on 24 GB VRAM
_dqn.HYPERPARAMS["batch_size"]  = 512       # was 256 — larger minibatch for stable gradients

# ── Entry point ───────────────────────────────────────────────────────────────

def _print_kpi_table():
    """Print the per-stage KPI targets at startup for easy reference."""
    print("\nPer-stage Target KPIs (graduation difficulty reference):")
    print("  {:>7s}  {:>20s}  {:>10s}  {:>10s}".format(
        "Stage", "Description", "NDR ≥", "Jain ≥"))
    for stage, kpi in STAGE_TARGET_KPIS.items():
        ndr_str   = "{:.1f}%".format(kpi["ndr_pct"]) if kpi["ndr_pct"] else "final"
        jains_str = "{:.2f}".format(kpi["jains"])    if kpi["jains"]   else "final"
        print("  {:>7d}  {:>20s}  {:>10s}  {:>10s}".format(
            stage, kpi["description"], ndr_str, jains_str))
    print()


if __name__ == "__main__":
    _print_kpi_table()
    _dqn.main()