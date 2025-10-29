"""
Custom Gymnasium Environment Registration for UAV Data Collection

This module registers custom environments for the UAV IoT data collection task.
"""

from gymnasium.envs.registration import register

# Import the environment class to make it available
from uav_env import UAVEnvironment

# Register the main environment
register(
    id='UAVDataCollection-v0',
    entry_point='environment.uav_env:UAVEnvironment',
    max_episode_steps=500,
    kwargs={
        'grid_size': (20, 20),
        'num_sensors': 20,
        'data_generation_rate': 2.2,
        'max_battery': 274.0,
    }
)

# Register easy variant (for debugging/testing)
register(
    id='UAVDataCollection-Easy-v0',
    entry_point='environment.uav_env:UAVEnvironment',
    max_episode_steps=300,
    kwargs={
        'grid_size': (10, 10),
        'num_sensors': 5,
        'data_generation_rate': 1.0,
        'max_battery': 274.0,
    }
)

# Register medium variant
register(
    id='UAVDataCollection-Medium-v0',
    entry_point='environment.uav_env:UAVEnvironment',
    max_episode_steps=500,
    kwargs={
        'grid_size': (20, 20),
        'num_sensors': 20,
        'data_generation_rate': 2.2,
        'max_battery': 274.0,
    }
)

# Register hard variant (for scalability testing)
register(
    id='UAVDataCollection-Hard-v0',
    entry_point='environment.uav_env:UAVEnvironment',
    max_episode_steps=1000,
    kwargs={
        'grid_size': (30, 30),
        'num_sensors': 50,
        'data_generation_rate': 3.0,
        'max_battery': 274.0,
    }
)

# Make environment class available for direct import
__all__ = ['UAVEnvironment']