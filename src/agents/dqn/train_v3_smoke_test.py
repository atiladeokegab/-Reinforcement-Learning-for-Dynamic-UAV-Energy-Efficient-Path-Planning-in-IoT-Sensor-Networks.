"""
Smoke test — Competence-Based Curriculum (400×400 cap, 450k steps)
==================================================================

Verifies that the competence gate advances stages correctly without
running a full 3M-step retrain.

Overrides from dqn.py
---------------------
  • CURRICULUM_STAGES  — 3 stages only (100→200→300→400), no 500×500
  • COMPETENCE_GATE    — relaxed: NDR ≥ 80%, Jain's ≥ 0.70, window=20, min_steps=50k
  • DEMOTION_GATE      — min_episodes=10 so demotion triggers quickly if needed
  • total_timesteps    — 450,000whta what si the eval
  • N_EVAL_EPISODES    — 20 (fast eval pass)
  • EVAL_FREQ          — 10,000
  • WORKER_SENSOR_COUNTS — [10, 20] (2 workers, fewer sensors for speed)
  • N_ENVS             — 2
  • SAVE_DIR / LOG_DIR — smoke_test subdirectory so nothing overwrites main models

Expected behaviour
------------------
  With relaxed gates the agent should reach Stage 1 (200×200) within ~150k
  steps and Stage 2 (300×300) by ~300k.  Stage 3 (400×400) may or may not
  be reached in 450k — that is intentional: it tells you whether the
  curriculum is progressing at a reasonable pace.
"""

from pathlib import Path
import dqn as _dqn

# ── Output dirs ───────────────────────────────────────────────────────────────
_dqn.SAVE_DIR = Path("models/smoke_test_400")
_dqn.LOG_DIR  = Path("logs/smoke_test_400")
_dqn.SAVE_DIR.mkdir(parents=True, exist_ok=True)
_dqn.LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Curriculum — 4 stages capped at 400×400 ───────────────────────────────────
_dqn.CURRICULUM_STAGES = [
    ([(100, 100)],                                         [10, 20], "Stage 0 — 100×100 only"),
    ([(100, 100), (200, 200)],                             [10, 20], "Stage 1 — up to 200×200"),
    ([(100, 100), (200, 200), (300, 300)],                 [10, 20], "Stage 2 — up to 300×300"),
    ([(100, 100), (200, 200), (300, 300), (400, 400)],     [10, 20], "Stage 3 — up to 400×400"),
]

# ── Relaxed competence gate so a short run can still graduate ─────────────────
_dqn.COMPETENCE_GATE = {
    "ndr_pct":   80.0,   # relaxed: 80% sensors visited (full run uses 95%)
    "jains":     0.70,   # relaxed: Jain's 0.70 (full run uses 0.85)
    "window":    20,     # shorter window for faster feedback
    "min_steps": 50_000, # 50k per stage × 4 stages ≤ 200k; plenty of headroom
}

# ── Demotion gate ─────────────────────────────────────────────────────────────
_dqn.DEMOTION_GATE = {
    "ndr_pct":      50.0,
    "jains":        0.40,
    "min_episodes": 10,
}

# ── Navigation fixes (Fix 1 + Fix 2) ─────────────────────────────────────────
_dqn.NAV_CONFIG = {
    "min_start_dist":  30.0,   # lighter threshold for 100×100 grids
    "max_start_tries": 200,
    "prox_eta":        2.0,
}

# ── Greedy benchmark gate ─────────────────────────────────────────────────────
_dqn.GREEDY_BENCHMARK = {
    "enabled":      True,
    "n_episodes":   10,    # 10 episodes — fast enough for a smoke test
    "sensor_count": 20,
    "margin_ndr":   2.0,   # lighter margin so a short run can still graduate
    "margin_jains": 0.01,  # threshold capped at 0.97 in dqn.py
    "floor_ndr":    40.0,
    "floor_jains":  0.30,
}

# ── Eval settings ─────────────────────────────────────────────────────────────
_dqn.EVAL_FREQ       = 10_000
_dqn.N_EVAL_EPISODES = 20
_dqn.EVAL_GRID       = (400, 400)
_dqn.EVAL_N_SENSORS  = 20

# ── 2 workers (faster, less VRAM) ────────────────────────────────────────────
_dqn.N_ENVS                = 2
_dqn.WORKER_SENSOR_COUNTS  = [10, 20]

# ── Short training run ────────────────────────────────────────────────────────
_dqn.TRAINING_CONFIG = {
    "total_timesteps": 450_000,
    "save_freq":       50_000,
    "n_stack":         4,
}

# ── Faster learning starts (smaller buffer fill needed at 2 workers) ──────────
_dqn.HYPERPARAMS["learning_starts"]    = 5_000
_dqn.HYPERPARAMS["batch_size"]         = 128
_dqn.HYPERPARAMS["buffer_size"]        = 50_000
_dqn.HYPERPARAMS["exploration_final_eps"] = 0.03   # Fix 3 — force deterministic policy


if __name__ == "__main__":
    print("=" * 70)
    print("SMOKE TEST — Competence-Based Curriculum (400×400 cap, 450k steps)")
    print("=" * 70)
    print("Competence Gate (RELAXED for smoke test):")
    print("  NDR   >= {:.0f}%  (production: 95%)".format(_dqn.COMPETENCE_GATE["ndr_pct"]))
    print("  Jain's >= {:.2f}  (production: 0.85)".format(_dqn.COMPETENCE_GATE["jains"]))
    print("  Window = {} episodes, min_steps = {:,}".format(
        _dqn.COMPETENCE_GATE["window"], _dqn.COMPETENCE_GATE["min_steps"]))
    print()
    print("Stages (4, capped at 400×400):")
    for i, (grids, sensors, desc) in enumerate(_dqn.CURRICULUM_STAGES):
        print("  Stage {}  {}  sensors={}".format(i, desc, sensors))
    print()
    _dqn.main()
