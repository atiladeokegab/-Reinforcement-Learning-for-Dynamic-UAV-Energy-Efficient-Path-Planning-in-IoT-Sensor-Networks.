"""
Q-Learning Agent for UAV Path Planning

Implements tabular Q-Learning with epsilon-greedy exploration
for discrete state-action spaces.

Author: ATILADE GABRIEL OKE
Date: October 2025
Project: Reinforcement Learning for Dynamic UAV Energy-Efficient Path Planning
         in IoT Sensor Networks
"""

import numpy as np
import pickle
from typing import Tuple, Dict, Optional
from collections import defaultdict
import os


class QLearningAgent:
    """
    Q-Learning agent with epsilon-greedy exploration.

    Uses a Q-table (dictionary) to store state-action values.
    States are discretized from continuous observations.

    Attributes:
        num_actions: Number of possible actions
        learning_rate: Learning rate (alpha)
        discount_factor: Discount factor (gamma)
        epsilon: Exploration rate
        epsilon_min: Minimum exploration rate
        epsilon_decay: Epsilon decay rate per episode
        q_table: Dictionary mapping (state, action) -> Q-value

    Example:
        >>> agent = QLearningAgent(num_actions=5)
        >>> state = discretize_state(observation)
        >>> action = agent.select_action(state)
        >>> agent.update(state, action, reward, next_state, done)
    """

    def __init__(self,
                 num_actions: int = 5,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995):
        """
        Initialize Q-Learning agent.

        Args:
            num_actions: Number of possible actions
            learning_rate: Learning rate alpha (0 < alpha <= 1)
            discount_factor: Discount factor gamma (0 < gamma < 1)
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon per episode
        """
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: (state, action) -> Q-value
        # Using defaultdict for automatic initialization
        self.q_table: Dict[Tuple, float] = defaultdict(float)

        # Statistics
        self.total_updates = 0
        self.episodes_trained = 0

    def discretize_state(self, observation: np.ndarray,
                         grid_size: Tuple[int, int] = (10, 10)) -> Tuple:
        """
        Discretize continuous observation into discrete state.

        Simplifies state space by discretizing continuous values.

        Args:
            observation: [uav_x, uav_y, battery, sensor1_buf, ..., sensorN_buf]
            grid_size: Grid dimensions for position discretization

        Returns:
            Discretized state tuple
        """
        # Extract components
        uav_x = int(np.clip(observation[0], 0, grid_size[0] - 1))
        uav_y = int(np.clip(observation[1], 0, grid_size[1] - 1))
        battery = observation[2]
        sensor_buffers = observation[3:]

        # Discretize battery into 10 levels (0-10%, 10-20%, ..., 90-100%)
        battery_level = int(np.clip(battery / 274.0 * 10, 0, 9))

        # Simplify sensor states: just track if all sensors are collected
        all_sensors_empty = int(np.all(sensor_buffers <= 0))

        # Create discrete state
        state = (uav_x, uav_y, battery_level, all_sensors_empty)

        return state

    def select_action(self, state: Tuple, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Discretized state
            training: If True, use epsilon-greedy. If False, use greedy.

        Returns:
            Selected action (0-4)
        """
        # Exploration: random action
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.num_actions)

        # Exploitation: best known action
        q_values = [self.q_table[(state, a)] for a in range(self.num_actions)]
        max_q = max(q_values)

        # Handle ties randomly
        best_actions = [a for a in range(self.num_actions) if q_values[a] == max_q]
        return np.random.choice(best_actions)

    def update(self,
               state: Tuple,
               action: int,
               reward: float,
               next_state: Tuple,
               done: bool) -> float:
        """
        Update Q-value using Q-Learning update rule.

        Q(s,a) = Q(s,a) + α * [r + γ * max Q(s',a') - Q(s,a)]

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended

        Returns:
            TD error (for logging)
        """
        # Current Q-value
        current_q = self.q_table[(state, action)]

        # Maximum Q-value for next state
        if done:
            max_next_q = 0.0
        else:
            next_q_values = [self.q_table[(next_state, a)] for a in range(self.num_actions)]
            max_next_q = max(next_q_values)

        # TD target
        target_q = reward + self.discount_factor * max_next_q

        # TD error
        td_error = target_q - current_q

        # Q-Learning update
        new_q = current_q + self.learning_rate * td_error
        self.q_table[(state, action)] = new_q

        self.total_updates += 1

        return td_error

    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episodes_trained += 1

    def get_statistics(self) -> Dict:
        """Get agent statistics."""
        return {
            'q_table_size': len(self.q_table),
            'epsilon': self.epsilon,
            'total_updates': self.total_updates,
            'episodes_trained': self.episodes_trained,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor
        }

    def save(self, filepath: str):
        """
        Save agent to file.

        Args:
            filepath: Path to save file
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        agent_data = {
            'q_table': dict(self.q_table),
            'num_actions': self.num_actions,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'total_updates': self.total_updates,
            'episodes_trained': self.episodes_trained
        }

        with open(filepath, 'wb') as f:
            pickle.dump(agent_data, f)

        print(f"✓ Agent saved to {filepath}")

    def load(self, filepath: str):
        """
        Load agent from file.

        Args:
            filepath: Path to load file
        """
        with open(filepath, 'rb') as f:
            agent_data = pickle.load(f)

        self.q_table = defaultdict(float, agent_data['q_table'])
        self.num_actions = agent_data['num_actions']
        self.learning_rate = agent_data['learning_rate']
        self.discount_factor = agent_data['discount_factor']
        self.epsilon = agent_data['epsilon']
        self.epsilon_min = agent_data['epsilon_min']
        self.epsilon_decay = agent_data['epsilon_decay']
        self.total_updates = agent_data['total_updates']
        self.episodes_trained = agent_data['episodes_trained']

        print(f"✓ Agent loaded from {filepath}")
        print(f"  Q-table size: {len(self.q_table)}")
        print(f"  Episodes trained: {self.episodes_trained}")

    def __repr__(self) -> str:
        """String representation."""
        return (f"QLearningAgent(actions={self.num_actions}, "
                f"α={self.learning_rate}, γ={self.discount_factor}, "
                f"ε={self.epsilon:.3f}, q_size={len(self.q_table)})")


# Testing
if __name__ == "__main__":
    print("=" * 70)
    print("Testing Q-Learning Agent")
    print("=" * 70)
    print()

    # Create agent
    agent = QLearningAgent(
        num_actions=5,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0
    )

    print(f"✓ Agent created: {agent}")
    print()

    # Test state discretization
    print("Test 1: State Discretization")
    print("-" * 70)

    # Mock observation: [uav_x, uav_y, battery, sensor_buffers...]
    obs = np.array([5.3, 7.8, 137.0] + [500.0] * 20)
    state = agent.discretize_state(obs)

    print(f"  Observation: x={obs[0]:.1f}, y={obs[1]:.1f}, battery={obs[2]:.1f}")
    print(f"  Discretized state: {state}")
    print(f"  State components: pos=({state[0]}, {state[1]}), battery_level={state[2]}, all_empty={state[3]}")
    print()

    # Test action selection
    print("Test 2: Action Selection")
    print("-" * 70)

    actions = []
    for _ in range(10):
        action = agent.select_action(state, training=True)
        actions.append(action)

    print(f"  10 random actions (ε={agent.epsilon}): {actions}")
    print(f"  Action distribution: {np.bincount(actions, minlength=5)}")
    print()

    # Test Q-value update
    print("Test 3: Q-Value Updates")
    print("-" * 70)

    state = (5, 5, 5, 0)
    action = 0
    reward = 10.0
    next_state = (5, 6, 5, 0)
    done = False

    print(f"  Initial Q({state}, {action}) = {agent.q_table[(state, action)]:.3f}")

    td_error = agent.update(state, action, reward, next_state, done)

    print(f"  After update: Q({state}, {action}) = {agent.q_table[(state, action)]:.3f}")
    print(f"  TD error: {td_error:.3f}")
    print()

    # Test multiple updates
    print("Test 4: Multiple Updates")
    print("-" * 70)

    for i in range(100):
        action = agent.select_action(state, training=True)
        reward = np.random.uniform(-1, 1)
        next_state = (
            np.random.randint(0, 10),
            np.random.randint(0, 10),
            np.random.randint(0, 10),
            np.random.randint(0, 2)
        )
        done = np.random.random() < 0.1
        agent.update(state, action, reward, next_state, done)
        state = next_state

    stats = agent.get_statistics()
    print(f"  Total updates: {stats['total_updates']}")
    print(f"  Q-table size: {stats['q_table_size']}")
    print()

    # Test epsilon decay
    print("Test 5: Epsilon Decay")
    print("-" * 70)

    print(f"  Initial epsilon: {agent.epsilon:.4f}")

    for episode in range(100):
        agent.decay_epsilon()

    print(f"  After 100 episodes: {agent.epsilon:.4f}")
    print(f"  Episodes trained: {agent.episodes_trained}")
    print()

    # Test save/load
    print("Test 6: Save and Load")
    print("-" * 70)

    # Save
    agent.save("checkpoints/test_agent.pkl")

    # Create new agent and load
    agent2 = QLearningAgent()
    agent2.load("checkpoints/test_agent.pkl")

    print(f"  Loaded agent: {agent2}")
    print()

    print("=" * 70)
    print("✓ All tests complete!")
    print("=" * 70)