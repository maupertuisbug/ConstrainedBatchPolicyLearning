import torch
import numpy as np
import gym

from networks.network import Network
from torch.utils.data import TensorDataset, DataLoader
from env.frozen_lake import FrozenLakeEnv


def one_hot(n, state):
    vec  = np.zeros(n)
    vec[state] = 1
    return vec 

class FQI:
    def __init__(self, input_dataset, cost, dones, model, config, wandb_run, name):
        self.input_dataset = input_dataset
        self.cost          = cost 
        self.dones         = dones 
        self.model         = model
        self.config        = config
        self.wandb_run     = wandb_run
        self.name          = name
        

    def one_hot_to_state(one_hot_vector):
        return int(np.argmax(one_hot_vector))


    def train(self):

        dataset = TensorDataset(self.input_dataset, self.cost, self.dones)
        loader = DataLoader(dataset, batch_size=128, shuffle=True)
        print("Input dataset shape: ", self.input_dataset.shape)
        losses = []
        for i in range(self.config.num_iterations):
            losses = []
            for input_batch, reward_batch, done_batch in loader:
                q_values = self.model(input_batch)
                reward_batch = reward_batch.unsqueeze(1)
                done_batch   = done_batch.unsqueeze(1)

                target_q_values = reward_batch + (1 - done_batch) * self.config.gamma * q_values

                loss = torch.nn.MSELoss()(q_values, target_q_values)
                losses.append(loss.item())

                self.model.zero_grad()
                loss.backward(retain_graph=True)
                self.model.optimizer.step()
            
            self.wandb_run.log({self.name+"_loss": np.mean(losses)}, step=i)

    def evaluate(self):

        env = gym.make('FQEFrozenLake-v1', desc=None, is_slippery=False)
        for episodes in range(self.config.evaluation_episodes):
            state, _ = env.reset()
            state_n = env.observation_space.n 
            action_n = env.action_space.n
            done = False 
            total_reward = 0
            print(episodes)
            while not done:
                q_values = [self.model(torch.cat((torch.tensor(one_hot(state_n, state), dtype=torch.float32, device=self.device), torch.tensor(one_hot(action_n, a), dtype=torch.float32, device=self.device)), dim=0)).detach().cpu().numpy() for a in range(env.action_space.n)]
                print(q_values)
                action = int(np.argmax(np.argmax(q_values)))
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                state = next_state
                if done :
                    break
            self.wandb_run.log({"total_reward": total_reward},  step=episodes)

        










