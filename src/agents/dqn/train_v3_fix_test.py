"""
Quick sanity-check retrain for the two reward fixes applied to dqn_v3:
  1. sensor_urgency now uses _calculate_urgency() [0,1] not raw AoI
  2. penalty_unvisited raised from -15,000 → -100,000

Runs 300k timesteps — enough to see whether the agent starts moving.
Saves to: models/dqn_v3_fix_test/
"""

import sys
from pathlib import Path

import dqn as _dqn_module

_dqn_module.TRAINING_CONFIG["total_timesteps"] = 300_000
_dqn_module.TRAINING_CONFIG["save_freq"]       = 50_000
_dqn_module.TRAINING_CONFIG["learning_rate"]   = 3e-4
_dqn_module.TRAINING_CONFIG["buffer_size"]     = 50_000   # smaller for quick test
_dqn_module.TRAINING_CONFIG["batch_size"]      = 64

# Keep curriculum simple: only unlock Stage 0-1 in 300k steps
_dqn_module.CURRICULUM_THRESHOLDS = [100_000, 200_000]

_dqn_module.SAVE_DIR = Path("models/dqn_v3_fix_test")
_dqn_module.LOG_DIR  = Path("logs/dqn_v3_fix_test")
_dqn_module.SAVE_DIR.mkdir(parents=True, exist_ok=True)
_dqn_module.LOG_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    _dqn_module.main()
