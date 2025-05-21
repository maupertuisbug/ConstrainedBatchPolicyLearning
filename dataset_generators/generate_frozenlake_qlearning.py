import gym
import numpy as np 
import matplotlib.pyplot as plt 
import wandb
import pickle
from env.frozen_lake import FrozenLakeEnv

env = gym.make("FQEFrozenLake-v1", is_slippery=True)

wandb_run = wandb.init(project="fitted q_function")


n_states = env.observation_space.n 
n_actions = env.action_space.n 
q_table  = np.zeros((n_states, n_actions))

alpha = 0.99 
gamma = 0.95 
epsilon = 1.0 
epsilon_decay = 0.998 
epsilon_min  = 0.01
episodes = 80000 
max_steps = 100

dataset = []


rewards = [] 

def one_hot(n, state):
    vec  = np.zeros(n)
    vec[state] = 1
    return vec 

for episode in range(episodes):

    state, _ = env.reset()
    state_n = env.observation_space.n 
    action_n = env.action_space.n
    total_reward = 0 
    done = False

    for step in range(max_steps):

        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else :
            action = np.argmax(q_table[state])
        
        next_state, reward, terminated, cost = env.step(action)
        done = terminated 

        old_q_value = q_table[state, action]
        next_max    = np.max(q_table[next_state])

        q_table[state, action] = old_q_value + alpha * (reward + gamma * next_max - old_q_value)

        if episode > 78000:
            dataset.append((one_hot(state_n, state), one_hot(action_n, action), one_hot(state_n, next_state), reward, cost, done))

        state = next_state 
        total_reward += reward 
        if done :
            break 

    epsilon = max(epsilon_min, epsilon*epsilon_decay)
    rewards.append(total_reward)
    wandb_run.log({"Total Reward" : np.mean(rewards)})

print("Dataset size: ", len(dataset))
print(dataset[0])
with open('frozenlake_qldataset.pkl', 'wb') as f:
    pickle.dump(dataset, f)
print("Dataset saved to frozenlake_qldataset.pkl")


state, _ = env.reset()
epsilon = 1.0
done = False 
rewards =  []
total_reward = 0 
evaluation_episodes = 1000
for episode in range(evaluation_episodes):
    state, _ = env.reset()
    done = False 
    total_reward = 0 
    while not done :
        action = np.argmax(q_table[state])
        
        next_state, reward, terminated, cost = env.step(action)
        done = terminated
        state  = next_state
        total_reward += reward 
        if done : 
            break
    rewards.append(total_reward)
    wandb_run.log({"Total Reward in Eval" : np.mean(rewards)})
