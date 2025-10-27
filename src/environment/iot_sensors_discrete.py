# Discrete Process Flow: Sensor State Update (Step t -> t+1)

# --- Phase 1: Data Generation and Buffer Accumulation  ---

# 1.0 Initial State
# Current buffer B(t) and UAV position are known at the start of the step.

# 1.1 Data Generation
# D_new is the fixed, discrete amount of data generated per step.
# B_new_gen represents the buffer *after* generation but *before* overflow check.
# B_new_gen = B(t) + D_new

# 1.2 Buffer Overflow Check
# B_max is the sensor's maximum buffer capacity.
# B_interim is the buffer level after generation, respecting the max size.
# B_interim = min(B_new_gen, B_max)

# --- Phase 2: UAV Communication and Data Collection  ---

# 2.1 Distance & RSSI Calculation
# Compute the distance 'd' between the UAV and the sensor node.
# Compute RSSI(d) using the path loss model (Equation 3).

# 2.2 Communication Check
# RSSI_th is the minimum required signal strength (sensitivity threshold).
# IF RSSI(d) >= RSSI_th (In Range) THEN:

    # 2.3 Max Collection Capacity
    # R_data is the continuous data rate (bytes/sec) based on the current SF.
    # T_col is the fixed communication duration (seconds) within this step.
    # D_max = R_data * T_col

    # 2.4 Actual Collected Data
    # The actual data collected is limited by what's in the buffer (B_interim) or the capacity (D_max).
    # D_collected = min(B_interim, D_max)

# ELSE (RSSI(d) < RSSI_th - Out of Range) THEN:

    # 2.5 No Collection
    # D_collected = 0

# --- Phase 3: Final State Update ⬆ ---

# 3.1 Final Buffer Update
# The buffer for the next step (t+1) is the interim buffer minus the collected data.
# B(t+1) = B_interim - D_collected

# 3.2 Next Step
# The simulation proceeds to Step t+1 with the new buffer state B(t+1).
# Buffer Utilization U_buffer = (B(t+1) / B_max) * 100