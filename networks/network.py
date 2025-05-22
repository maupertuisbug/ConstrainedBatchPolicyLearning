import torch 
import numpy as np



# Your layer_input is a list of tuples with three elements: (layer_input, activation, layer_output)
# You will create a linear layer for each tuple in the list
# You will have a model with the layers and the forward function


def one_hot(n, state):
    vec  = np.zeros(n)
    vec[state] = 1
    return vec 

class Network(torch.nn.Module):
    def __init__(self, num_layers, layer_input, env):
        super(Network, self).__init__()
        self.layers = [] 
        for i in range(num_layers):
            layer_input_size, activation, layer_output_size = layer_input[i]
            if activation == 'relu':
                activation_fn = torch.nn.ReLU()
            elif activation == 'sigmoid':
                activation_fn = torch.nn.Sigmoid()
            elif activation == 'tanh':
                activation_fn = torch.nn.Tanh()
            elif activation == 'linear':    
                activation_fn = torch.nn.Identity()
            else:
                raise ValueError(f"Unsupported activation function: {activation}")

            self.layers.append(torch.nn.Linear(layer_input_size, layer_output_size))
            self.layers.append(activation_fn)
        print(self.layers)
        self.model = torch.nn.Sequential(*self.layers)
        self.model.apply(init_weights)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.env = env 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def init_weights(m):
        if isinstance(m, mm.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        network_output = self.model(x)
        return network_output

    def get_best_action(self, x):
        actions  = []
        for i in range(self.env.action_space.n):
            action = one_hot(self.env.action_space.n, action)
            network_output = torch.cat((x, action), dim=1)
            network_output = self.model(network_output)
            actions.append(network_output)
        
        best_action = torch.argmax(actions).item()
        return one_hot(best_action)
    
    def get_best_action_batch(self, x):

        batch_size = x.shape[0]
        n_actions = self.env.action_space.n
        actions_one_hot = torch.eye(n_actions, device=self.device)
        
        x_expanded = x.unsqueeze(1).repeat(1, n_actions, 1)
        actions_expanded = actions_one_hot.unsqueeze(0).repeat(batch_size, 1, 1)

        input_combined = torch.cat([x_expanded, actions_expanded], dim=-1)
        input_flat     = input_combined.view(-1, input_combined.shape[-1])

        q_values       = self.model(input_flat)
        q_values       = q_values.view(batch_size, n_actions)

        best_action_indices = torch.argmax(q_values, dim=1)
        best_actions        = torch.nn.functional.one_hot(best_action_indices, num_classes=n_actions)

        return best_actions


    

        