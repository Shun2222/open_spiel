import os
import os.path as osp
import pyspiel
from utils import onehot, multionehot
# 
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=6 Mainly controlles the number of spawned threateds 
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=6

import argparse
from tqdm import tqdm
from distutils.util import strtobool
import time
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import copy
from matplotlib import animation

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import logger
from open_spiel.python.mfg import utils
from open_spiel.python import rl_environment
from open_spiel.python import policy as policy_std
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms.nash_conv import NashConv
from open_spiel.python.mfg.algorithms import policy_value
from open_spiel.python.mfg.games import factory
from open_spiel.python.mfg import value
from open_spiel.python.mfg.algorithms import best_response_value
from games.predator_prey import goal_distance, divide_obs



class MultiDiscriminatorLoader():
    def __init__(self, env):
        self._env = env
        self._discriminators = None
        self._net_inputs = None

    def get_net_inputs(self):
        return self._net_inputs
    def get_discriminators(self):
        return self._discriminators


    def load(self, disc_path_update_info):
        from open_spiel.python.mfg.algorithms.discriminator_networks_divided_value import * 
        
        num_agent = nmu = 3 
        disc_path = [[disc_path_update_info[i][j][0] for j in range(len(disc_path_update_info[i]))] for i in range(num_agent)]
        update_eps_info = [[disc_path_update_info[i][j][1] for j in range(len(disc_path_update_info[i]))] for i in range(num_agent)]

        def check_path(path):
            is_nets = is_networks(path) 
            is_divided = is_divided_value(path)
            is_exist = osp.isfile(path)
            return is_nets and is_divided and is_exist

        for i in range(num_agent):
            for j in range(disc_path[i]):
                assert self.check_path(disc_path[i][j]), f"Bad path: {disc_path[i][j]}"


        net_inputs = []
        for i in range(3):
            net_input = []
            for j in range(len(disc_path[i])):
                net_input.append(get_net_input(disc_path[i][j]))
            net_inputs.append(net_input)
        self._net_inputs = net_inputs

        nacs = self._env.action_spec()['num_actions']
        nobs = self._env.observation_spec()['info_state'][0]
        horizon = self._env.game.get_parameters()['horizon']

        size = self._env.game.get_parameters()['size']
        state_size = nobs - 1 - horizon # nobs-1: obs size (exposed own mu), nmu: all agent mu size, horizon: horizon size
        obs_xym_size = nobs - 1 - horizon + nmu # nobs-1: obs size (exposed own mu), nmu: all agent mu size, horizon: horizon size
        discriminators = [[] for _ in range(num_agent)]

        device = torch.device("cpu")
        for i in range(num_agent):
            for j in range(len(disc_path)):
                inputs = get_input_shape(net_inputs[i][j], self._env, num_agent)
                labels = get_net_labels(net_inputs[i][j])
                num_hidden = 1
                print(num_hidden)
                if len(labels)==1:
                    discriminator = Discriminator(inputs, obs_xym_size, labels, device, num_hidden=num_hidden)
                if len(labels)==2:
                    discriminator = Discriminator_2nets(inputs, obs_xym_size, labels, device, num_hidden=num_hidden)
                if len(labels)==3:
                    discriminator = Discriminator_3nets(inputs, obs_xym_size, labels, device, num_hidden=num_hidden)

                print(f'jth disc of Agent i is loaded from {disc_path[i][j]}')
                discriminators[i][j].load(disc_path[i][j], update_eps_info[i][j], use_eval=True)
                discriminators[i][j].print_weights()
            discriminators[i].append(discriminator)

        self._discriminators = discriminators
        return discriminators 
        
if __name__ == "__main__":
    env = None

    disc_path_update_info = [
                    [["/mnt/shunsuke/result/09xx/multi_maze2_dxy_mu-divided_value", "200_2-0"],
                    ["/mnt/shunsuke/result/09xx/predator_prey_mu-divided_value_group0", "200_2-0"],
                    ],
                    [["/mnt/shunsuke/result/09xx/multi_maze2_dxy_mu-divided_value", "200_2-1"]],
                    [["/mnt/shunsuke/result/09xx/multi_maze2_dxy_mu-divided_value", "200_2-2"]],
                ]

    mdl = MultiDiscriminatorLoader(env)
    mdl.load(disc_path_update_info)