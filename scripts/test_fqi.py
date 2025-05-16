import torch 
import torch.nn as nn
import wandb
import argparse
from omegaconf import OmegaConf
import minari
import pickle
import gym
from networks.network import Network
from fqi import FQI

def run_exp():
    # Initialize wandb
    wandb_run = wandb.init(project="fitted q_function")
    config = wandb.config

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device being used ", device)

    num_layers = config.num_layers
    hidden_units = config.hidden_units

        
    with open('frozenlake_qldataset.pkl', 'rb') as f:
        loaded_dataset = pickle.load(f)
    print("Loaded dataset size: ", len(loaded_dataset))
    print("First 5 samples of loaded dataset: ", loaded_dataset[:5])

    # Your frozen lake does not have an obervation space that is a box
    env = gym.make('FrozenLake-v1', desc=None, is_slippery=False)

    print("Environment action space: ", env.observation_space.n)
    input_size = 2
    output_size = 1

    layers = []
    layers.append([input_size, 'relu', hidden_units])
    for layer in range(num_layers - 1):
        layers.append([hidden_units, 'relu', hidden_units])
    layers.append([hidden_units, 'linear', output_size])
    print("Layers: ", layers)

    model = Network(num_layers+1, layers).to(device)
    model = model.to(torch.float32)
    fqi = FQI(loaded_dataset, config, model, wandb=wandb_run)
    fqi.train()
    fqi.evaluate()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-config')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    config_dict = OmegaConf.to_container(config, resolve=True)

    project_name = "fitted q_function"
    sweep_id = wandb.sweep(sweep=config_dict, project=project_name)
    agent = wandb.agent(sweep_id, function=run_exp, count = 5)