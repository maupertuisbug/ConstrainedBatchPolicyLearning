import torch
import numpy as np
import gym

from networks.network import Network


class FQI:
    def __init__(self, tuple_dataset, config, model, wandb=None):
        self.wandb = wandb
        self.tuple_dataset = tuple_dataset
        self.config = config
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self):
        # You need the dataset as torch tensors
        dataset = torch.tensor(self.tuple_dataset, dtype=torch.float32).to(self.device)
        print("Dataset shape: ", dataset.shape)

        # You need to split the dataset into states, actions, rewards, next_states, and dones
        states = dataset[:, 0].unsqueeze(1)
        actions = dataset[:, 1].unsqueeze(1)
        rewards = dataset[:, 2].unsqueeze(1)
        next_states = dataset[:, 3].unsqueeze(1)
        dones = dataset[:, 4].unsqueeze(1)

        # You need to create the target Q-values
        input_dataset = torch.cat((states, actions), dim=1)
        print("Input dataset shape: ", input_dataset.shape)
        for i in range(self.config.num_iterations):
            q_values = self.model(input_dataset)

            target_q_values = rewards + (1 - dones) * self.config.gamma * q_values

            loss = torch.nn.MSELoss()(q_values, target_q_values)
            self.wandb.log({"loss": loss.item()})

            self.model.zero_grad()
            loss.backward()
            self.model.optimizer.step()

    def evaluate(self):

        env = gym.make('FrozenLake-v1', desc=None, is_slippery=False)
        state = env.reset()
        done = False
        total_reward = 0
        for episodes in range(self.config.evaluation_episodes):
            state = env.reset()
            done = False 
            total_reward = 0
            while not done:
                q_values = [self.model(torch.cat((torch.tensor(state, dtype=torch.float32).unsqueeze(0), torch.tensor(a, dtype=torch.float32, device=device).unsqueeze(0)), dim=0)).detach().numpy() for a in range(env.action_space.n)]
                action = np.argmax(q_values)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                state = next_state
            self.wandb.log({"total_reward": total_reward})

        










