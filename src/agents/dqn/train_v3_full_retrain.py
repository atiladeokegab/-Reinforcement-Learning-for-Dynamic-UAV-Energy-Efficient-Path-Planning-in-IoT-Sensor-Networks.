"""
Full 3M-step retrain with all three fixes applied:

  Fix 1 — Urgency bug: sensor_urgency now uses _calculate_urgency() [0,1]
           instead of raw AoI (which was inflating byte rewards 250×)
  Fix 2 — penalty_unvisited raised -15,000 → -100,000 (opportunity-cost justified)
  Fix 3 — penalty_starved = -30,000 for sensors with CR < 20% at episode end
  Fix 4 — IoTSensor.reset() initialises total_data_generated = initial_buffer_fill
           so CR stays bounded to [0, 1]

Full curriculum: Stage 0 (100×100) → Stage 4 (500×500) over 3M steps.
Saves to: models/dqn_v3_fixed/
"""

import sys
from pathlib import Path

import dqn as _dqn

# ── Compress curriculum so all 5 stages fit within 3M steps ──────────────────
# Default thresholds (1M/2M/4M/7M) mean stage 4 (500×500) is never reached.
# Compressed schedule:
#   Stage 0 (100×100): 0 – 400k
#   Stage 1 (200×200): 400k – 900k
#   Stage 2 (300×300): 900k – 1.5M
#   Stage 3 (400×400): 1.5M – 2.25M
#   Stage 4 (500×500): 2.25M – 3M
_dqn.CURRICULUM_THRESHOLDS = [400_000, 900_000, 1_500_000, 2_250_000]

_dqn.SAVE_DIR = Path("models/dqn_v3_fixed")
_dqn.LOG_DIR  = Path("logs/dqn_v3_fixed")
_dqn.SAVE_DIR.mkdir(parents=True, exist_ok=True)
_dqn.LOG_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    _dqn.main()
