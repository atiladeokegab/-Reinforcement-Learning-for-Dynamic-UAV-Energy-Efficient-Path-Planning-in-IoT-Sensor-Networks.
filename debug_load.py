import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/agents/dqn')

print('Step 1: importing GNNExtractor...')
from gnn_extractor import GNNExtractor
print('Step 2: importing SB3...')
from stable_baselines3 import DQN
print('Step 3: loading model...')
model = DQN.load('src/agents/dqn/models/dqn_v5_gnn/dqn_final.zip', device='cpu')
print('Model loaded OK:', model.policy)
