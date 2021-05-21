from argparse import ArgumentParser
from gym import make
from gym.spaces import Box, Discrete
import gym
import gym_fairrec
# import roboschool
from yaml import load
import yaml

from models import build_diag_gauss_policy, build_mlp, build_multinomial_policy
from simulators import *
from transforms import *
from torch_utils import get_device
from trpo import TRPO

import pandas as pd
import numpy as np
from read_data import read_file, read_embeddings, Embeddings

config_filename = 'config.yaml'

parser = ArgumentParser(prog='train.py',
                        description='Train a policy on the specified environment' \
                        ' using Trust Region Policy Optimization (Schulman 2015)' \
                        ' with Generalized Advantage Estimation (Schulman 2016).')
parser.add_argument('--continue', dest='continue_from_file', action='store_true',
                    help='Set this flag to continue training from a previously ' \
                    'saved session. Session will be overwritten if this flag is ' \
                    'not set and a saved file associated with model-name already ' \
                    'exists.')
parser.add_argument('--model-name', type=str, dest='model_name', required=True,
                    help='The entry in trpo_experiments.yaml from which settings' \
                    'should be loaded.')
parser.add_argument('--simulator', dest='simulator_type', type=str, default='single-path',
                    choices=['single-path', 'vine'], help='The type of simulator' \
                    ' to use when collecting training experiences.')

args = parser.parse_args()
continue_from_file = args.continue_from_file
model_name = args.model_name
simulator_type = args.simulator_type

all_configs = load(open(config_filename, 'r'), Loader=yaml.FullLoader)
config = all_configs[model_name]

device = get_device()

# Find the input size, hidden dim sizes, and output size
env_name = config['env_name']
# env = gym.make(env_name)
data = read_file('./data/train.csv')
embeddings = Embeddings(read_embeddings('./data/embeddings.csv'))
env = gym.make(env_name, data=data, embeddings=embeddings, alpha=0.5, gamma=0.9, fixed_length=True, trajectory_length=5)
action_space = env.action_space
observation_space = env.observation_space
policy_hidden_dims = config['policy_hidden_dims']
vf_hidden_dims = config['vf_hidden_dims']
vf_args = (observation_space.shape[0] + 1, vf_hidden_dims, 1)

# Initialize the policy
if type(action_space) is Box:
    policy_args = (observation_space.shape[0], policy_hidden_dims, action_space.shape[0])
    policy = build_diag_gauss_policy(*policy_args)
elif type(action_space) is Discrete:
    policy_args = (observation_space.shape[0], policy_hidden_dims, action_space.n)
    policy = build_multinomial_policy(*policy_args)
else:
    raise NotImplementedError

# Initalize the value function
value_fun = build_mlp(*vf_args)
policy.to(device)
value_fun.to(device)

# Initialize the state transformation
z_filter = ZFilter()
state_bound = Bound(-5, 5)
state_filter = Transform(state_bound, z_filter)

# Initialize the simulator
n_trajectories = config['n_trajectories']
max_timesteps = config['max_timesteps']
# try:
#     env_args = config['env_args']
# except:
env_args = {}
env_args['data'] = data
env_args['embeddings'] = embeddings
env_args['alpha'] = 0.5
env_args['gamma'] = 0.9
env_args['fixed_length'] = True
env_args['trajectory_length'] = 5

if simulator_type == 'single-path':
    simulator = SinglePathSimulator(env_name, policy, n_trajectories,
                                    max_timesteps, state_filter=state_filter,
                                    **env_args)
elif simulator_type == 'vine':
    raise NotImplementedError

try:
    trpo_args = config['trpo_args']
except:
    trpo_args = {}

trpo = TRPO(policy, value_fun, simulator, model_name=model_name,
            continue_from_file=continue_from_file, **trpo_args)

print(f'Training policy {model_name} on {env_name} environment...\n')

trpo.train(config['n_episodes'])

print('\nTraining complete.\n')
