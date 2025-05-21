import torch 
import numpy as np 
import gym 


from networks.network import Network 
from torch.utils.data import TensorDataset, DataLoader


def one_hot(n, state):
    vec = np.zeros(n)
    vec[state] = 1
    return vec


class FQE:
    def __init__(self, states, actions, cost, dones, policy_eval, model, wandb_run, name):

        self.states = states
        self.actions  = actions
        self.cost     = cost
        self.dones         = dones 
        self.policy_eval   = policy_eval
        self.model         = model
        self.wandb_run     = wandb_run
        self.name          = name

    
    def one_hot_to_state(one_hot_vector):
        return int(np.argmax(one_hot_vector))

    
    def train(self):

        dataset = TensorDataset(self.states, self.actions, self.cost, self.dones)
        loader  = DataLoader(dataset, batch_size=128, shuffle=True)
        losses = [] 
        num_iterations = 100 
        gamma = 0.99

        for i in range(num_iterations):
            losses = []
            for states, actions, reward_batch, done_batch in loader:
                # for optimal policy 
                action_eval = self.policy_eval.get_best_action_batch(states)
                # prepare the new dataset 
                target_dataset = torch.cat((states, action_eval), dim=1)
                q_values       = self.model(target_dataset)
                reward_batch = reward_batch.unsqueeze(1)
                done_batch   = done_batch.unsqueeze(1)
                target = reward_batch + (1-done_batch) * gamma * q_values

                loss = torch.nn.MSELoss()(q_values, target)
                losses.append(loss.item())

                self.model.zero_grad()
                loss.backward(retain_graph=True)
                self.model.optimizer.step()
            
            self.wandb_run.log({self.name+"_loss-eval" : np.mean(losses)})