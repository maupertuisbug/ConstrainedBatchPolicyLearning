import torch 



# Your layer_input is a list of tuples with three elements: (layer_input, activation, layer_output)
# You will create a linear layer for each tuple in the list
# You will have a model with the layers and the forward function

class Network(torch.nn.Module):
    def __init__(self, num_layers, layer_input):
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
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, x):
        network_output = self.model(x)
        return network_output