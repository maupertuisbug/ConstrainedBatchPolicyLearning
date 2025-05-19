import torch 
import numpy as np 
import copy

from networks.network import Network
from cbpl.fqe import FQE
from cbpl.fqi import FQI

from torch.utils.data import TensorDataset, DataLoader


class CBPL:
    def __init__(self, dataset, B, learning_rate, env, config, wandb_run):
        self.dataset = dataset 
        self.bound = B 
        self.lr = learning_rate 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lmda = torch.nn.Parameter(torch.full((1,), (B/1+1), dtype=torch.float32, device=self.device))
        self.env = env
        self.config = config
        self.wandb_run = wandb_run

        # You need to initialise the dataset such that rewards and cost from constraints 
        states, actions, next_states, rewards, constraints, dones = zip(*self.dataset)

        self.states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        self.actions = torch.tensor(np.array(actions), dtype=torch.float32, device=self.device)
        self.next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        self.rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        self.constraints = torch.tensor(np.array(constraints), dtype=torch.float32, device=self.device)
        self.dones  = torch.tensor(np.array(dones), dtype=torch.float32, device=self.device)

    def initialize_q_functions(self):

        input_size = self.env.observation_space.n + self.env.action_space.n 
        output_size = 1 

        layers1 = [] 
        hidden_units = self.config.q1_hidden_units 
        num_layers   = self.config.q1_num_layers
        layers1.append([input_size, 'relu', hidden_units])
        for layer in range(num_layers-1):
            layers1.append([hidden_units, 'relu', hidden_units])
        layers1.append([hidden_units, 'linear', output_size])

        self.q1_policy = Network(num_layers+1, layers1, self.env).to(self.device).to(torch.float32)

        layers2 = [] 
        hidden_units = self.config.q2_hidden_units 
        num_layers   = self.config.q2_num_layers
        layers2.append([input_size, 'relu', hidden_units])
        for layer in range(num_layers-1):
            layers2.append([hidden_units, 'relu', hidden_units])
        layers2.append([hidden_units, 'linear', output_size])

        self.q2_eval = Network(num_layers+1, layers2, self.env).to(self.device).to(torch.float32)

        layers3 = [] 
        hidden_units = self.config.q3_hidden_units 
        num_layers   = self.config.q3_num_layers
        layers3.append([input_size, 'relu', hidden_units])
        for layer in range(num_layers-1):
            layers3.append([hidden_units, 'relu', hidden_units])
        layers3.append([hidden_units, 'linear', output_size])

        self.q3_eval = Network(num_layers+1, layers3, self.env).to(self.device).to(torch.float32)

        layers4 = [] 
        hidden_units = self.config.q4_hidden_units 
        num_layers   = self.config.q4_num_layers
        layers4.append([input_size, 'relu', hidden_units])
        for layer in range(num_layers-1):
            layers4.append([hidden_units, 'relu', hidden_units])
        layers4.append([hidden_units, 'linear', output_size])

        self.q4_policy = Network(num_layers+1, layers4, self.env).to(self.device).to(torch.float32)

    
    def update(self, avg_model, sample_model, input_dataset, t):

        dataset = TensorDataset(input_dataset) 
        loader = DataLoader(dataset, batch_size=128, shuffle=True)

        for batch in loader:
            loss = float(1/t)*[sample_model(batch[0]) - avg_model(batch[0])]
            avg_model.zero_grad()
            loss.backward()
            avg_model.optimizer()

    def get_values(self, input_dataset, model):

        dataset  = TensorDataset(input_dataset)
        loader   = DataLoader(dataset, batch_size=128, shuffle=True)
        value = []
        for batch in loader:
            value.append(model(batch))
        
        value = torch.mean(torch.stack(value))
        return value

    def run_single_iteration(self, t):
        # you need to prepare dataset 
        input_dataset = torch.cat((self.states, self.actions), dim=1)
        cost = self.rewards + self.lmda*self.constraints

        fqi_a = FQI(input_dataset, cost, self.dones, self.q1_policy, self.config, self.wandb_run)
        fqi_a.train()

        fqe_a = FQE(self.states, self.actions, self.rewards, self.dones, self.q1_policy, self.q2_eval, self.wandb_run)
        fqe_a.train()

        fqe_b = FQE(self.states, self.actions, self.constraints, self.dones, self.q1_policy, self.q3_eval, self.wandb_run)
        fqe_b.train()

        self.update(self.q1_avg, self.q1_policy, input_dataset, t)
        self.update(self.q2_avg, self.q2_eval, input_dataset, t)
        self.update(self.q3_avg, self.q3_eval, input_dataset, t)

        loss = self.lmda_avg - self.lmda
        self.lmda_avg = self.lmda_avg + (1/t)*loss

        cost = self.rewards + self.lmda_avg * self.constraints
        fqi_b = FQI(input_dataset, cost, self.dones, self.q4_policy, self.config, self.wandb_run)

        fqe_c = FQE(self.states, self.actions. self.rewards, self.dones, self.q5_eval, self.q4_policy, self.wandb_run)
        fqe_c.train()

        fqe_d = FQE(self.states, self.actions, self.constraints, self.dones, self.q6_eval, self.q4_policy, self.wandb_run)
        fqe_d.train()

        l_max = self.get_values(input_dataset, self.q2_eval) + self.lmda * max(self.get_values(input_dataset, self.q3_eval) - 0.1, 0)
        l_max = self.lmda + self.lr * l_max 

        l_min = self.get_values(input_dataset, self.q5_eval) + self.lmda_avg * max(self.get_values(input_dataset, self.q6_eval) - 0.1, 0)

        if l_max - l_min <= 0.0001 :
            return self.q1_policy

        self.wandb_run.log({"l_max", l_max})
        self.wandb_run.log({"l_min", l_min})

    
    def run(self, t):

        self.q1_avg = copy.deepcopy(self.q1_policy)
        self.q2_avg = copy.deepcopy(self.q2_eval)
        self.q3_avg = copy.deepcopy(self.q3_eval)
        self.q4_avg = copy.deepcopy(self.q4_policy)
        self.q5_eval = copy.deepcopy(self.q1_policy)
        self.q6_eval = copy.deepcopy(self.q1_policy)
        self.lmda_avg = copy.deepcopy(self.lmda)

        for i in range(t):
            self.run_single_iteration(i+1)
        








  
