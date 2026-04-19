import sys
import json
import numpy as np
import gymnasium
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/agents/dqn')

print('Step 1: importing GNNExtractor...')
from gnn_extractor import GNNExtractor
print('Step 2: importing SB3...')
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
print('Step 3: loading model...')
model = DQN.load('src/agents/dqn/models/dqn_v5_gnn/dqn_final.zip', device='cpu')
print('Model loaded OK')

# Check training config
try:
    with open('src/agents/dqn/models/dqn_v5_gnn/training_config.json') as f:
        config = json.load(f)
    print('Config:', config)
except FileNotFoundError:
    config = {"use_frame_stacking": True, "n_stack": 10, "max_sensors_limit": 50, "features_per_sensor": 3, "include_sensor_positions": False}
    print('No config found, using defaults:', config)

print('Step 4: creating environment...')
from environment.uav_env import UAVEnvironment

class PaddedEnv(UAVEnvironment):
    _MSL = config["max_sensors_limit"]
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        fps = self._features_per_sensor
        pad_n = (self._MSL - self.num_sensors) * fps
        padded_dim = self.observation_space.shape[0] + pad_n
        self._pad_n = pad_n
        self.observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(padded_dim,), dtype=np.float32)
    def _pad(self, obs):
        if self._pad_n == 0:
            return obs
        return np.concatenate([obs, np.zeros(self._pad_n, dtype=np.float32)])
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        return self._pad(obs), info
    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        return self._pad(obs), r, term, trunc, info

env_kwargs = dict(grid_size=(200,200), num_sensors=10, max_steps=2100,
                  path_loss_exponent=3.8, rssi_threshold=-85.0,
                  sensor_duty_cycle=10.0, max_battery=274.0,
                  uav_start_position=(100,100),
                  include_sensor_positions=config.get("include_sensor_positions", False),
                  render_mode=None)

vec_env = DummyVecEnv([lambda: PaddedEnv(**env_kwargs)])
print(f'Obs space shape: {vec_env.observation_space.shape}')
vec_env = VecFrameStack(vec_env, n_stack=config.get("n_stack", 10))
print(f'Stacked obs shape: {vec_env.observation_space.shape}')

print('Step 5: running one episode...')
obs = vec_env.reset()
print(f'Reset obs shape: {obs.shape}')
step = 0
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, dones, _ = vec_env.step([int(action[0])])
    step += 1
    if step % 100 == 0:
        print(f'  step {step}...')
    if bool(dones[0]):
        print(f'Episode done at step {step}')
        break
vec_env.close()
print('ALL OK')
