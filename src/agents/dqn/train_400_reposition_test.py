"""
Focused repositioning test — 400×400 grid, 20 sensors, random UAV start.
Trains from scratch for 200k steps with the reward fixes in place.
No curriculum — locked to 400×400 from step 1 so the model HAS to move.

Saves to: models/dqn_400_reposition_test/
"""

import sys
from pathlib import Path
import numpy as np
import json
import gymnasium
import random

# ── pull in the base training infrastructure ──────────────────────────────────
import dqn as _dqn

# ── Override the curriculum so every episode is 400×400, 20 sensors ──────────
_dqn.CURRICULUM_STAGES = [
    ([(400, 400)], [20], "Fixed 400×400 / 20 sensors"),
]
_dqn.CURRICULUM_THRESHOLDS = []          # no stage transitions
_dqn.EVAL_GRID      = (400, 400)
_dqn.EVAL_N_SENSORS = 20

_dqn.TRAINING_CONFIG["total_timesteps"] = 200_000
_dqn.TRAINING_CONFIG["save_freq"]       = 50_000
_dqn.HYPERPARAMS["buffer_size"]         = 50_000
_dqn.HYPERPARAMS["batch_size"]          = 64
_dqn.HYPERPARAMS["learning_starts"]     = 5_000

_dqn.SAVE_DIR = Path("models/dqn_400_reposition_test_v2")
_dqn.LOG_DIR  = Path("logs/dqn_400_reposition_test_v2")
_dqn.SAVE_DIR.mkdir(parents=True, exist_ok=True)
_dqn.LOG_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    _dqn.main()
