"""
400×400 reposition test — v3 (apples-to-apples with v1).

Why this exists:
  v1 accidentally used default hyperparams (buffer=150k, batch=256,
  learning_starts=25k) because the overrides were in TRAINING_CONFIG
  (wrong dict) and were silently ignored.

  v2 fixed the dict but also changed the values (buffer=50k, batch=64,
  learning_starts=5k) — making v1 vs v2 not directly comparable.

  v3 = v2's correct wiring BUT the same default values as v1, so the
  ONLY difference between v1 and v3 is the starvation penalty (-30k
  for sensors with CR<20% at episode end).

Hyperparams (identical to dqn.py defaults = what v1 effectively used):
  buffer_size      = 150,000
  batch_size       = 256
  learning_starts  = 25,000

Saves to: models/dqn_400_reposition_test_v3/
"""

from pathlib import Path
import dqn as _dqn

# ── Curriculum: locked to 400×400, 20 sensors ────────────────────────────────
_dqn.CURRICULUM_STAGES     = [([(400, 400)], [20], "Fixed 400×400 / 20 sensors")]
_dqn.CURRICULUM_THRESHOLDS = []
_dqn.EVAL_GRID             = (400, 400)
_dqn.EVAL_N_SENSORS        = 20

# ── Training schedule ─────────────────────────────────────────────────────────
_dqn.TRAINING_CONFIG["total_timesteps"] = 200_000
_dqn.TRAINING_CONFIG["save_freq"]       = 50_000

# ── Hyperparams: leave buffer/batch/learning_starts at dqn.py defaults ───────
# (buffer_size=150k, batch_size=256, learning_starts=25k)
# DO NOT override — this matches what v1 actually trained with.

# ── Output dirs ───────────────────────────────────────────────────────────────
_dqn.SAVE_DIR = Path("models/dqn_400_reposition_test_v3")
_dqn.LOG_DIR  = Path("logs/dqn_400_reposition_test_v3")
_dqn.SAVE_DIR.mkdir(parents=True, exist_ok=True)
_dqn.LOG_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    _dqn.main()
