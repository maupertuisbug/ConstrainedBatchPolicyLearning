import torch 
import numpy as np 






class CBPL:
    def __init__(self, dataset, l1, B, learning_rate, env):
        self.dataset = dataset 
        self.l1 = l1 
        self.bound = B 
        self.lr = learning_rate 
        self.lmda = torch.nn.Parameter(torch.full((1,), (B/1+1)))
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # You need to initialise the dataset such that rewards and cost from constraints 
        states, actions, next_states, rewards, constraints, dones = zip(*self.dataset)

        self.states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        self.actions = torch.tensor(np.array(actions), dtype=torch.float32, device=self.device)
        self.next_states = torch.tensor(np.array(next_states), dtype=torch.float3, device=self.device)
        self.rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        self.constraints = torch.tensor(np.array(constraints), dtype=torch.float32, device=self.device)
        self.dones  = torch.tensor(np.array(dones), dtype=torch.float32, device=self.device)




    
    def run(self, t):
        input_size = self.env.observation_space.n + self.env.action_space.
        output_size = 1
        layers = [] 
        layers.append([input_size, 'relu', hidden_units])
        hidden_units = 64 
        num_layers = 1
        for layer in range(num_layers - 1):
            layers.append([hidden_units, 'relu', hidden_units])
        layers.append([hidden_units, 'linear', output_size])
        print("Layers: ", layers)

        avg_q_function = Network(num_layers+1, layers).to(self.device)




        


        for i in range(t):

            # You need to prepare data for this 
            
            fitted_q_function = FQI() # You need to pass policy that you learnt using the regulazarization 
            current_cost = FQE(fitted_q_function, self.rewards)
            current_constraint = FQE(fitted_q_function, self.constraints)

            fitted_q_function = avg_fitted_qf + (1/t)*[fitted_q_function - avg_fitted_qf]
