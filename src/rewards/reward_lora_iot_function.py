import numpy as np
from typing import Dict, Optional


class RewardFunction:
    """
    Literature-aligned reward function for UAV IoT data collection.

    1. Weighted sum for continuous objectives (Data, Energy, AoI, Sync).
    2. Static penalties/bonuses applied unweighted.
    3. AoI reduction is normalized by max_possible_aoi (max_episode_steps).

    """

    def __init__(self,
                 # Normalization parameters
                 max_buffer_size: float = 1000.0,
                 max_battery: float = 274.0,
                 max_possible_aoi: float = 3600.0,  # e.g., Max steps * Time step (or a known max AoI)

                 # Reward weights (sum to 1.0 for main objectives)
                 weight_data: float = 0.6,
                 weight_energy: float = 0.2,
                 weight_aoi: float = 0.1,
                 weight_sync: float = 0.1,

                 # Data collection rewards (scale factors)
                 reward_data_scale: float = 10.0,
                 reward_new_sensor: float = 5.0,  # Static bonus
                 reward_completion: float = 100.0,  # Static bonus

                 # Broadcast timing rewards (scale factor/static penalty)
                 reward_broadcast_sync_scale: float = 5.0,  # Scale factor for sync objective
                 penalty_wasted_hover: float = -1.0,  # Static penalty

                 # Energy penalties (scale factor/static penalty)
                 penalty_energy_scale: float = 5.0,  # Scale factor for energy objective
                 penalty_battery_critical: float = -20.0,  # Static penalty
                 battery_critical_threshold: float = 20.0,  # percent

                 # AoI rewards (scale factor)
                 reward_aoi_scale: float = 10.0,  # Increased scale since input is normalized

                 # Operation penalties (static penalties)
                 penalty_empty_collection: float = -2.0,
                 penalty_boundary: float = -5.0,
                 penalty_step: float = -0.05):

        # Normalization
        self.max_buffer_size = max_buffer_size
        self.max_battery = max_battery
        self.max_possible_aoi = max_possible_aoi  # NEW

        # Weights
        self.weight_data = weight_data
        self.weight_energy = weight_energy
        self.weight_aoi = weight_aoi
        self.weight_sync = weight_sync

        # Rewards & Penalties
        self.reward_data_scale = reward_data_scale
        self.reward_new_sensor = reward_new_sensor
        self.reward_completion = reward_completion

        self.reward_broadcast_sync_scale = reward_broadcast_sync_scale
        self.penalty_wasted_hover = penalty_wasted_hover

        self.penalty_energy_scale = penalty_energy_scale
        self.penalty_battery_critical = penalty_battery_critical
        self.battery_critical_threshold = battery_critical_threshold

        self.reward_aoi_scale = reward_aoi_scale

        self.penalty_empty_collection = penalty_empty_collection
        self.penalty_boundary = penalty_boundary
        self.penalty_step = penalty_step

        # Validation
        total_weight = weight_data + weight_energy + weight_aoi + weight_sync
        if not (0.99 <= total_weight <= 1.01):
            print(f"Warning: Weights sum to {total_weight:.3f}, not 1.0")

    def calculate_movement_reward(self,
                                  move_success: bool,
                                  battery_used: float,
                                  battery_remaining: float) -> float:
        """
        Calculate reward for movement action. (Logic remains correct from previous fix)
        """
        reward = 0.0

        # --- STATIC PENALTIES ---
        reward += self.penalty_step
        if not move_success:
            reward += self.penalty_boundary

        # --- ENERGY OBJECTIVE (WEIGHTED) ---
        energy_normalized = battery_used / self.max_battery
        r_energy = -self.penalty_energy_scale * energy_normalized
        reward += r_energy * self.weight_energy

        # --- STATIC CRITICAL BATTERY PENALTY ---
        battery_percent = (battery_remaining / self.max_battery) * 100
        if battery_percent < self.battery_critical_threshold:
            reward += self.penalty_battery_critical

        return reward

    def calculate_collection_reward(self,
                                    bytes_collected: float,
                                    was_new_sensor: bool,
                                    was_empty: bool,
                                    all_sensors_collected: bool,
                                    battery_used: float,
                                    battery_remaining: float,
                                    # AoI parameters
                                    aoi_before: Optional[float] = None,
                                    aoi_after: Optional[float] = None,
                                    # Broadcast timing parameters
                                    any_broadcasting: bool = False,
                                    successfully_synced: bool = False) -> float:
        """
        Calculate reward for data collection/hover action.
        Incorporates AoI normalization fix.
        """
        reward = 0.0

        # Initialize continuous objective scores (unweighted, scaled)
        r_data, r_energy, r_aoi, r_sync = 0.0, 0.0, 0.0, 0.0

        # --- 1. DATA COLLECTION (Scaled Component & Static Bonuses) ---
        if bytes_collected > 0:
            # Scaled Reward Component
            normalized_bytes = bytes_collected / self.max_buffer_size
            r_data = self.reward_data_scale * normalized_bytes

            # Static Bonus
            if was_new_sensor:
                reward += self.reward_new_sensor
        else:
            # Static Penalties
            reward += self.penalty_step
            if was_empty:
                reward += self.penalty_empty_collection

        # --- 2. ENERGY EFFICIENCY (Scaled Component & Static Penalty) ---
        # Scaled Penalty Component
        energy_normalized = battery_used / self.max_battery
        r_energy = -self.penalty_energy_scale * energy_normalized

        # Static Critical Battery Penalty
        battery_percent = (battery_remaining / self.max_battery) * 100
        if battery_percent < self.battery_critical_threshold:
            reward += self.penalty_battery_critical

        # --- 3. AGE OF INFORMATION (AoI) (Scaled Component) ---
        if aoi_before is not None and aoi_after is not None:
            aoi_reduction = aoi_before - aoi_after

            # AoI NORMALIZATION FIX: Scale by maximum possible AoI (max time)
            normalized_aoi_reduction = aoi_reduction / self.max_possible_aoi

            # Scaled Component
            r_aoi = self.reward_aoi_scale * normalized_aoi_reduction

        # --- 4. BROADCAST SYNCHRONIZATION (Scaled Component & Static Penalty) ---
        if bytes_collected > 0 and successfully_synced:
            # Scaled Bonus Component
            r_sync = self.reward_broadcast_sync_scale
        elif bytes_collected == 0 and not any_broadcasting:
            # Static Penalty
            reward += self.penalty_wasted_hover

        # =====================================================================
        # SUM WEIGHTED OBJECTIVES
        # =====================================================================
        weighted_reward = (
                r_data * self.weight_data +
                r_energy * self.weight_energy +
                r_aoi * self.weight_aoi +
                r_sync * self.weight_sync
        )
        reward += weighted_reward

        # --- 6. MISSION COMPLETION BONUS (Static Bonus) ---
        if all_sensors_collected:
            reward += self.reward_completion

        return reward

    def get_reward_breakdown(self,
                             bytes_collected: float,
                             battery_used: float,
                             aoi_reduction: float = 0.0,
                             synced: bool = False) -> Dict[str, float]:
        """
        Get breakdown of reward components for analysis/debugging.
        Note: This breakdown shows the *weighted* contribution of the four main objectives.

        FIXED: Now applies AoI normalization consistently with calculate_collection_reward()
        """
        normalized_bytes = bytes_collected / self.max_buffer_size
        normalized_energy = battery_used / self.max_battery

        # Calculate the scaled (unweighted) components
        r_data = self.reward_data_scale * normalized_bytes
        r_energy = -self.penalty_energy_scale * normalized_energy

        # FIXED: Apply AoI normalization here too
        normalized_aoi_reduction = aoi_reduction / self.max_possible_aoi
        r_aoi = self.reward_aoi_scale * normalized_aoi_reduction

        r_sync = self.reward_broadcast_sync_scale if synced else 0.0

        breakdown = {
            # Weighted Components
            'data_reward_weighted': r_data * self.weight_data,
            'energy_penalty_weighted': r_energy * self.weight_energy,
            'aoi_reward_weighted': r_aoi * self.weight_aoi,
            'sync_reward_weighted': r_sync * self.weight_sync,

            # Sum of weighted components
            'total_weighted': (r_data * self.weight_data + r_energy * self.weight_energy +
                               r_aoi * self.weight_aoi + r_sync * self.weight_sync)
        }

        return breakdown

    def get_config(self) -> Dict:
        """
        Get reward function configuration for logging/reproducibility.

        Returns:
            Dictionary with all reward parameters
        """
        return {
            # Normalization - FIXED: Added max_possible_aoi
            'max_buffer_size': self.max_buffer_size,
            'max_battery': self.max_battery,
            'max_possible_aoi': self.max_possible_aoi,

            # Weights
            'weight_data': self.weight_data,
            'weight_energy': self.weight_energy,
            'weight_aoi': self.weight_aoi,
            'weight_sync': self.weight_sync,

            # Data rewards
            'reward_data_scale': self.reward_data_scale,
            'reward_new_sensor': self.reward_new_sensor,
            'reward_completion': self.reward_completion,

            # Broadcast timing
            'reward_broadcast_sync_scale': self.reward_broadcast_sync_scale,
            'penalty_wasted_hover': self.penalty_wasted_hover,

            # Energy
            'penalty_energy_scale': self.penalty_energy_scale,
            'penalty_battery_critical': self.penalty_battery_critical,
            'battery_critical_threshold': self.battery_critical_threshold,

            # AoI
            'reward_aoi_scale': self.reward_aoi_scale,

            # Operations
            'penalty_empty_collection': self.penalty_empty_collection,
            'penalty_boundary': self.penalty_boundary,
            'penalty_step': self.penalty_step,
        }

