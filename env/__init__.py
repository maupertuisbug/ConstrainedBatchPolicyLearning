from env.frozen_lake import FrozenLakeEnv
import gymnasium 
from gym.envs.registration import register
import sys
sys.path.append('/workspace/fqe/env')



register(id="FQEFrozenLake-v1", entry_point='env.frozen_lake:FrozenLakeEnv')


