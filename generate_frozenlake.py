import gym 
import numpy as np 
import pickle


env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False)

dataset = [] 
n_episodes = 8000
max_steps = 100 


for episode in range(n_episodes):
    state = env.reset()
    done = False 

    for _ in range(max_steps):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        dataset.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            break
print("Dataset size: ", len(dataset))
with open('frozenlake_dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f)
print("Dataset saved to frozenlake_dataset.pkl")





