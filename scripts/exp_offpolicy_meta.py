import torch 
import torch.nn as nn 
import wandb
import argparse
from omegaconf import OmegaConf
import minari 
import pickle 
import gym 
from env.frozen_lake import FrozenLakeEnv
from cbpl.constrained_batch_policy_learning import CBPL





def run_exp():

    wandb_run = wandb.init(project="MetalearningForOffPolicy")
    config = wandb.config 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device being used ", device)

    with open('frozenlake_qldataset.pkl', 'rb') as f:
        loaded_dataset = pickle.load(f)
    
    env = gym.make('FQEFrozenLake-v1', desc=None, is_slippery=False)

    lr = 0.01
    B  = 0.5 

    cbpl = CBPL(loaded_dataset, B, lr, env, config, wandb_run)
    cbpl.initialize_q_functions()
    cbpl.run(50)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-config')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    config_dict = OmegaConf.to_container(config, resolve=True)
    project_name = "MetaLearningforOffPolicyEval"
    sweep_id   = wandb.sweep(sweep=config_dict, project=project_name)
    agent      = wandb.agent(sweep_id, function=run_exp, count = 5)



    
    
