from src.agents.baselines.greedy_agents import ActiveSensorGreedy, test_greedy_agent
from src.environment.uav_env import UAVEnvironment

env = UAVEnvironment(
    grid_size=(20, 20),
    num_sensors=20,
    sensor_duty_cycle=10.0,
    render_mode='human'
)

agent = ActiveSensorGreedy(env)
results = test_greedy_agent(agent, env, num_episodes=1, render=True)