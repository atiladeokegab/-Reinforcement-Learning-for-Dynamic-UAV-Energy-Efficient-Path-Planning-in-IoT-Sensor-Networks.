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

import sys
from pathlib import Path

# Ensure src/ is on sys.path before any local imports
_SRC = Path(__file__).resolve().parent.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

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
# Global defaults (Stages 0, 1). Stages 2–3 are overridden in STAGE_GATES
# below because the 2,100-step episode budget caps spatial coverage —
# even MaxThroughputGreedyV2 cannot sustain Jain's ≥ 0.85 on grids > 200×200
# (measured: see src/agents/dqn/calibrate_stage_ceilings.py output).
_dqn.COMPETENCE_GATE = {
    "ndr_pct":   95.0,
    "jains":     0.85,
    "window":    50,
    "min_steps": 500_000,
}

# Per-stage overrides derived from greedy ceiling calibration (50-ep sweep,
# weighted across WORKER_SENSOR_COUNTS). Gate is set just above the greedy
# mean so "graduation" means the DQN has matched-or-beaten the heuristic on
# that stage's hardest grid.
#   Greedy Jain's (mean across N=20 & N=40):
#     300×300 ≈ 0.65,  400×400 ≈ 0.49,  500×500 ≈ 0.36
#   Greedy NDR (mean):
#     300×300 ≈ 98%,   400×400 ≈ 83%,   500×500 ≈ 64%
_dqn.STAGE_GATES = {
    2: {"ndr_pct": 95.0, "jains": 0.65},  # 300×300 — above greedy weighted mean
    3: {"ndr_pct": 80.0, "jains": 0.48},  # 400×400 — coverage-limited by 2100-step budget
    # Stage 4 (500×500) is final — no graduation gate applied.
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

# ── Greedy benchmark gate — disabled to unblock Stage 1 graduation ────────────
# Rationale: greedy on 200×200/N=20 hits ~100% NDR / ~0.95+ Jain's, pushing the
# gate to the 98% / 0.97 caps. That is unreachable over a rolling 50-ep window
# that mixes workers pinned at 10/15/25/35/40 sensors (WORKER_SENSOR_COUNTS
# below is wider than CURRICULUM_STAGES[1]'s [20,30,40]). Falling back to the
# fixed COMPETENCE_GATE (95% / 0.85) lets under-represented worker configs
# contribute without permanently gating advancement.
_dqn.GREEDY_BENCHMARK = {
    "enabled":      False,
    "n_episodes":   50,
    "sensor_count": 20,
    "margin_ndr":   3.0,
    "margin_jains": 0.03,
    "floor_ndr":    70.0,
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

# ── Extended budget — 3M was insufficient (prev run stalled in Stage 2) ──────
# Stage 0+1 took ~1M combined. Stages 2, 3, 4 need ~1.5M each to learn the
# larger grids well, so 5M total gives room for Stage 4 to see ≥ 1.5M.
_dqn.TRAINING_CONFIG["total_timesteps"] = 5_000_000

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