import gym 
import minari

dataset = minari.load_dataset('D4RL/antmaze/umaze-diverse-v1', download=True)
env = dataset.recover_environment()

print(env.action_space.sample())
