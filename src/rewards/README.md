# Mathematical Formulation of the UAV IoT Reward Function

## Overview

The reward function combines **weighted continuous objectives** with **static unweighted bonuses/penalties** to guide the UAV towards efficient data collection while respecting energy constraints and broadcast timing.

---

## 1. Collection Reward (Primary Reward Function)

### General Form

$$R_{\text{collect}} = R_{\text{weighted}} + R_{\text{static}}$$

Where:
- $R_{\text{weighted}}$ = weighted sum of four continuous objectives
- $R_{\text{static}}$ = static bonuses/penalties (not affected by weights)

---

### 1.1 Weighted Component (Four Objectives)

$$R_{\text{weighted}} = w_D \cdot r_D + w_E \cdot r_E + w_{AoI} \cdot r_{AoI} + w_S \cdot r_S$$

Where:
- $w_D, w_E, w_{AoI}, w_S$ = objective weights (sum to 1.0)
- $r_D, r_E, r_{AoI}, r_S$ = scaled reward components (described below)

**Weight Configuration:**
- $w_D = 0.6$ (Data collection priority)
- $w_E = 0.2$ (Energy efficiency)
- $w_{AoI} = 0.1$ (Age of Information)
- $w_S = 0.1$ (Broadcast Synchronization)

---

### 1.2 Objective 1: Data Collection Reward ($r_D$)

#### Scaled Component (Weighted)

$$r_D = \begin{cases}
\alpha_D \cdot \frac{b}{B_{\max}} & \text{if } b > 0 \\
0 & \text{otherwise}
\end{cases}$$

Where:
- $b$ = bytes collected in this step
- $B_{\max}$ = maximum buffer size (1000 bytes)
- $\alpha_D$ = data scale factor (10.0)

**Contribution to $R_{\text{weighted}}$:**
$$w_D \cdot r_D = 0.6 \times 10.0 \times \frac{b}{1000}$$

#### Static Bonuses (Unweighted)

$$B_D = \begin{cases}
+5.0 & \text{if } b > 0 \text{ AND } \text{sensor is new} \\
0 & \text{otherwise}
\end{cases}$$

**Example:** Collecting 100 bytes from a new sensor:
$$r_D = 10.0 \times \frac{100}{1000} = 1.0 \text{ (scaled)}$$
$$w_D \cdot r_D = 0.6 \times 1.0 = 0.6$$
$$B_D = 5.0 \text{ (bonus)}$$
$$\text{Total} = 0.6 + 5.0 = 5.6$$

---

### 1.3 Objective 2: Energy Efficiency Penalty ($r_E$)

#### Scaled Component (Weighted)

$$r_E = -\beta_E \cdot \frac{e}{E_{\max}}$$

Where:
- $e$ = battery energy consumed in this step
- $E_{\max}$ = maximum battery capacity (274 mAh)
- $\beta_E$ = energy penalty scale factor (5.0)

**Contribution to $R_{\text{weighted}}$:**
$$w_E \cdot r_E = 0.2 \times (-5.0) \times \frac{e}{274}$$

#### Static Penalty (Unweighted)

$$P_E = \begin{cases}
-20.0 & \text{if battery} < 20\% \\
0 & \text{otherwise}
\end{cases}$$

**Example:** Using 0.08 mAh with 20% battery remaining (54.8 mAh):
$$r_E = -5.0 \times \frac{0.08}{274} = -0.00146 \text{ (scaled)}$$
$$w_E \cdot r_E = 0.2 \times (-0.00146) \approx 0 \text{ (negligible)}$$
$$P_E = 0 \text{ (no critical penalty)}$$

---

### 1.4 Objective 3: Age of Information Reward ($r_{AoI}$)

#### Scaled Component (Weighted) - **NORMALIZED**

$$r_{AoI} = \gamma_{AoI} \cdot \frac{\Delta \text{AoI}}{T_{\max}}$$

Where:
- $\Delta \text{AoI}$ = AoI reduction (difference before and after collection)
- $T_{\max}$ = maximum possible AoI (3600 seconds, ~episode duration)
- $\gamma_{AoI}$ = AoI scale factor (10.0)

**Contribution to $R_{\text{weighted}}$:**
$$w_{AoI} \cdot r_{AoI} = 0.1 \times 10.0 \times \frac{\Delta \text{AoI}}{3600}$$

**Example:** AoI decreases from 500s to 450s (reduction of 50s):
$$r_{AoI} = 10.0 \times \frac{50}{3600} = 0.139 \text{ (scaled)}$$
$$w_{AoI} \cdot r_{AoI} = 0.1 \times 0.139 = 0.0139$$

**Why Normalize?** Without normalization, AoI rewards would depend on episode length. Normalization ensures consistent reward magnitude across different episode durations.

---

### 1.5 Objective 4: Broadcast Synchronization Reward ($r_S$)

#### Scaled Component (Weighted)

$$r_S = \begin{cases}
\gamma_S & \text{if } b > 0 \text{ AND synchronized with broadcast} \\
0 & \text{otherwise}
\end{cases}$$

Where:
- $\gamma_S$ = sync scale factor (5.0)

**Contribution to $R_{\text{weighted}}$:**
$$w_S \cdot r_S = 0.1 \times 5.0 = 0.5 \text{ (when synced)}$$

#### Static Penalty (Unweighted)

$$P_S = \begin{cases}
-1.0 & \text{if } b = 0 \text{ AND no broadcasts nearby} \\
0 & \text{otherwise}
\end{cases}$$

**Intuition:** Rewards the UAV for timing its hover with sensor broadcasts (efficient). Penalizes wasting energy hovering when no data is available.

---

### 1.6 Static Bonuses/Penalties (Unweighted)

$$R_{\text{static}} = P_{\text{step}} + P_{\text{boundary}} + B_D + P_E + P_S + B_{\text{mission}}$$

| Component | Value | Condition |
|-----------|-------|-----------|
| $P_{\text{step}}$ | -0.05 | Applied to every collection action |
| $P_{\text{boundary}}$ | -5.0 | Movement collision with boundary |
| $B_D$ | +5.0 | Collection from new sensor |
| $P_E$ | -20.0 | Battery < 20% |
| $P_S$ | -1.0 | Wasted hover (no broadcast) |
| $B_{\text{mission}}$ | +100.0 | All sensors collected |

**Why Separate Static Penalties?**
- They're not part of the multi-objective optimization
- They enforce hard constraints (energy critical, boundary violations)
- They don't interact with weight tuning

---

## 2. Movement Reward (Secondary Reward Function)

$$R_{\text{move}} = P_{\text{step}} + P_{\text{boundary}} + w_E \cdot r_E + P_E$$

Where:
- $P_{\text{step}} = -0.05$ (tiny penalty per step)
- $P_{\text{boundary}} = -5.0$ (if collision)
- $w_E \cdot r_E$ = weighted energy penalty (as above)
- $P_E = -20.0$ (if critical battery)

**Example:** Successful move using 0.08 mAh:
$$R_{\text{move}} = -0.05 + 0 + (0.2 \times (-5.0) \times \frac{0.08}{274}) + 0 = -0.05$$

---

## 3. Complete Reward Calculation Algorithm

```
Input: bytes_collected, was_new_sensor, battery_used, battery_remaining,
       aoi_before, aoi_after, any_broadcasting, successfully_synced

Output: R_total

1. Calculate scaled components:
   r_D = 10.0 * (bytes_collected / 1000)           [if bytes > 0]
   r_E = -5.0 * (battery_used / 274)
   r_AoI = 10.0 * ((aoi_before - aoi_after) / 3600)
   r_S = 5.0                                        [if synced & bytes > 0]

2. Apply weights and sum:
   R_weighted = 0.6 * r_D + 0.2 * r_E + 0.1 * r_AoI + 0.1 * r_S

3. Add static bonuses/penalties:
   R_static = -0.05                                 [step penalty]
   R_static += +5.0   if (was_new_sensor & bytes > 0)
   R_static += -2.0   if (bytes == 0 & was_empty)
   R_static += -1.0   if (bytes == 0 & !any_broadcasting)
   R_static += -20.0  if (battery_percent < 20%)
   R_static += +100.0 if (all_sensors_collected)

4. Total reward:
   R_total = R_weighted + R_static
```

---

## 4. Example: Perfect Collection Scenario

**Scenario:** UAV collects 100 bytes from a new sensor, synchronized with broadcast, AoI drops 50s, uses 0.11 mAh battery.

### Step 1: Scaled Components
$$r_D = 10.0 \times \frac{100}{1000} = 1.0$$
$$r_E = -5.0 \times \frac{0.11}{274} = -0.00201$$
$$r_{AoI} = 10.0 \times \frac{50}{3600} = 0.139$$
$$r_S = 5.0 \text{ (synced)}$$

### Step 2: Weighted Sum
$$R_{\text{weighted}} = 0.6(1.0) + 0.2(-0.00201) + 0.1(0.139) + 0.1(5.0)$$
$$= 0.6 - 0.0004 + 0.0139 + 0.5$$
$$= 1.1135$$

### Step 3: Static Bonuses
$$R_{\text{static}} = -0.05 + 5.0 = 4.95$$

### Step 4: Total Reward
$$R_{\text{total}} = 1.1135 + 4.95 = 6.0635 \approx 6.06$$

---

## 5. Design Rationale

### Why Weight Sum to 1.0?

Ensures that the magnitude of objectives is comparable:
- Data reward can be at most: $0.6 \times 10.0 = 6.0$ (collecting full buffer)
- Energy penalty can be at most: $0.2 \times (-5.0) = -1.0$ (draining full battery)
- This prevents any single objective from dominating by scale alone

### Why Normalize AoI?

Without normalization ($\Delta AoI / T_{\max}$):
- A 50-step reduction in a 100-step episode yields $r_{AoI} = 10.0 \times 50 = 500$
- Same reduction in a 1000-step episode yields $r_{AoI} = 10.0 \times 50 = 500$
- Both should be rewarded equally (same proportional improvement)

With normalization:
- 50-step reduction in 100-step episode: $r_{AoI} = 10.0 \times (50/100) = 5.0$
- 50-step reduction in 1000-step episode: $r_{AoI} = 10.0 \times (50/1000) = 0.5$
- Correctly reflects relative improvement

### Why Separate Static Penalties?

Multi-objective RL literature distinguishes:
1. **Continuous objectives** (data, energy, AoI, sync) → Weighted sum allows tuning trade-offs
2. **Hard constraints** (boundary, critical battery) → Static penalties enforce non-negotiable constraints
3. **Milestone bonuses** (new sensor, mission complete) → Unweighted to avoid weight-tuning interference

---

## 6. Key Parameters Summary

| Parameter | Value | Role |
|-----------|-------|------|
| $w_D$ | 0.6 | Prioritize data collection |
| $w_E$ | 0.2 | Moderate energy concerns |
| $w_{AoI}$ | 0.1 | Light emphasis on freshness |
| $w_S$ | 0.1 | Light emphasis on sync timing |
| $\alpha_D$ | 10.0 | Data scale |
| $\beta_E$ | 5.0 | Energy scale |
| $\gamma_{AoI}$ | 10.0 | AoI scale (normalized) |
| $\gamma_S$ | 5.0 | Sync scale |
| $T_{\max}$ | 3600s | Max episode duration |

---

## 7. Tuning Guide

**To prioritize energy efficiency:** Increase $w_E$, decrease $w_D$
- Example: $w_D = 0.4, w_E = 0.4, w_{AoI} = 0.1, w_S = 0.1$

**To penalize stale data:** Increase $w_{AoI}$ and $\gamma_{AoI}$
- Example: $w_{AoI} = 0.3, \gamma_{AoI} = 20.0$

**To enforce strict synchronization:** Increase $P_S$ penalty (wasted hover)
- Example: $P_S = -5.0$ instead of $-1.0$

---

## References

**Multi-Objective Reinforcement Learning:**
- Parisi et al., "Multi-objective reinforcement learning: A survey" (2022)
- Standard approach: Weighted sum of normalized objectives

**UAV Path Planning:**
- Recent works normalize energy by total capacity
- Normalize AoI by episode length to ensure scale-invariance

**IoT Sensor Networks:**
- Broadcast synchronization studied in duty-cycling networks
- Common to penalize idle listening (wasted hover)