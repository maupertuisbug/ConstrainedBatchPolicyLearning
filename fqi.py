import torch
import numpy as np
import gym

from networks.network import Network
from torch.utils.data import TensorDataset, DataLoader


def one_hot(n, state):
    vec  = np.zeros(n)
    vec[state] = 1
    return vec 

class FQI:
    def __init__(self, tuple_dataset, config, model, input_batch=None, target=None, dones = None, wandb=None):
        self.wandb = wandb
        self.tuple_dataset = tuple_dataset
        self.config = config
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.input_batch = input_batch
        self.cost = target
        self.dones = dones

    def one_hot_to_state(one_hot_vector):
        return int(np.argmax(one_hot_vector))


    def train(self):

        if self.tuple_dataset is not None:
            # You need the dataset as torch tensors
            states, actions, next_states, rewards, dones = zip(*self.tuple_dataset)

            # You need to split the dataset into states, actions, rewards, next_states, and dones
            states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
            actions = torch.tensor(np.array(actions), dtype=torch.float32, device=self.device)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
            rewards     = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
            dones       = torch.tensor(np.array(dones), dtype=torch.float32, device=self.device)

            # You need to create the target Q-values
            input_dataset = torch.cat((states, actions), dim=1)
            dataset = TensorDataset(input_dataset, rewards, dones)
        
        else :
            dataset = TensorDataset(self.input_dataset, self.cost, self.dones)
        loader = DataLoader(dataset, batch_size=128, shuffle=True)
        print("Input dataset shape: ", input_dataset.shape)
        losses = []
        for i in range(self.config.num_iterations):
            for input_batch, reward_batch, done_batch in loader:
                q_values = self.model(input_batch)
                reward_batch = reward_batch.unsqueeze(1)
                done_batch   = done_batch.unsqueeze(1)

                target_q_values = reward_batch + (1 - done_batch) * self.config.gamma * q_values

                loss = torch.nn.MSELoss()(q_values, target_q_values)
                losses.append(loss.item())

                self.model.zero_grad()
                loss.backward()
                self.model.optimizer.step()
            
            self.wandb.log({"loss": np.mean(losses)})

    def evaluate(self):

        env = gym.make('FrozenLake-v1', desc=None, is_slippery=False)
        for episodes in range(self.config.evaluation_episodes):
            state = env.reset()
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
            self.wandb.log({"total_reward": total_reward})

        










