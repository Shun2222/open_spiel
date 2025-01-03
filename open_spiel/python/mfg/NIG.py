import os
# 
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=6 Mainly controlles the number of spawned threateds 
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=6

import argparse
import pyspiel
import copy

import pickle as pkl
from distutils.util import strtobool
import time
import logging
import os.path as osp
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from open_spiel.python.mfg import utils
from utils import onehot, multionehot
from open_spiel.python import rl_environment
from open_spiel.python import policy as policy_std
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms.nash_conv import NashConv
from open_spiel.python.mfg.algorithms import policy_value
from open_spiel.python.mfg.algorithms.multi_type_mfg_ppo import *
from open_spiel.python.mfg.multi_render_reward import * 
from open_spiel.python.mfg.games import factory
from games.predator_prey import *
from open_spiel.python.mfg import value
from diff_utils import *
from gif_maker import *

plt.rcParams["font.size"] = 20
plt.rcParams["animation.ffmpeg_path"] = "/usr/bin/ffmpeg"

def calc_true_reward(obs_shape, horizon, mu_dists):
    inputs = [{} for _ in range(len(mu_dists))]
    rew = [np.zeros((horizon, obs_shape[0], obs_shape[1])) for _ in range(len(mu_dists))]
    rew_xy = [np.zeros((horizon, obs_shape[0], obs_shape[1])) for _ in range(len(mu_dists))]
    rew_mu = [np.zeros((horizon, obs_shape[0], obs_shape[1])) for _ in range(len(mu_dists))]
    for x in range(obs_shape[1]):
        for y in range(obs_shape[0]):
            for t in range(horizon):
                for idx in range(len(mu_dists)):
                    mu = np.array([mu_dists[idx][t, y, x] for idx in range(len(mu_dists))])
                    pos = np.array([x, y])
                    r, r_xy, r_mu = onetime_true_reward(pos, mu)
                    rew[idx][t, y, x] = r[idx]
                    rew_xy[idx][t, y, x] = r_xy[idx]
                    rew_mu[idx][t, y, x] = r_mu[idx]
    return rew, rew_xy, rew_mu 

def onetime_true_reward(pos, densities):

    _MODE = "Maze" # Maze or Predator-Prey

    if _MODE=="Predator_Prey":
        _DEFAULT_REWARD_MATRIX = np.array([[0, 100, 100], [-100, 0, 100], [-100, -100, 0]])
        _DEFAULT_FORBIDDEN_POSITION = np.array([])
    else:
        _DEFAULT_REWARD_MATRIX = np.array([[0, -50, -50], [-50, 0, -50], [-50, -50, 0]])
        _DEFAULT_FORBIDDEN_POSITION = np.array([[2, 4], [2, 5], [4, 2], [4, 7], [5, 2], [5, 7], [7, 4], [7, 5]])
    _DEFAULT_GOAL_POSITION = np.array([[5, 4], [4, 5], [5, 5]])

    eps = 1e-25
    goal_pos = _DEFAULT_GOAL_POSITION
    reward_matrix = _DEFAULT_REWARD_MATRIX
    forbidden_pos = _DEFAULT_FORBIDDEN_POSITION
    tf = pos==forbidden_pos
    tf = [tf2[0] and tf2[1] for tf2 in tf]
    if True in tf: 
        nans = [np.nan for _ in range(len(densities))]
        return nans, nans, nans

    if _MODE=="Predator-Prey":
        r_mu = -1.0 * np.log(densities + eps) + 10 * np.dot(reward_matrix, densities)
        rew = r_mu
    else:
        r_mu = -1.0 * np.log(densities + eps) + np.dot(reward_matrix, densities)
        r_xy = np.array([-np.sum(np.abs(goal_pos[i] - pos)) for i in range(len(goal_pos))])
        rew = r_mu + r_xy

    return rew, r_xy, r_mu


def create_rew_input(obs_shape, nacs, horizon, mu_dists, single, notmu, state_only=False):
    inputs = [{} for _ in range(len(mu_dists))]
    for x in range(obs_shape[1]):
        x_onehot = onehot(x, obs_shape[1]).tolist()
        for y in range(obs_shape[0]):
            for t in range(horizon):
                for idx in range(len(mu_dists)):
                    xy_onehot = x_onehot + onehot(y, obs_shape[0]).tolist()
                    if single:
                        for i in range(len(mu_dists)):
                            xym_onehot = xy_onehot + [mu_dists[i][t, y, x]]
                            inputs[f'{x}-{y}-{t}-m-{i}'] = xym_onehot
                    elif notmu:
                        inputs[f'{x}-{y}-{t}'] = xy_onehot
                    else:
                        mu = [mu_dists[idx][t, y, x]]
                        #for pop in range(len(mu_dists)):
                        #    if pop!=idx:
                        #        mu.append(mu_dists[pop][t, y, x])
                        xym_onehot = xy_onehot + mu 
                        inputs[idx][f'{x}-{y}-{t}-m'] = xym_onehot
    return inputs

def create_rew_with_tieme_input(obs_shape, nacs, horizon, mu_dists, single, notmu, state_only=False):
    inputs = {}
    for x in range(obs_shape[1]):
        x_onehot = onehot(x, obs_shape[1]).tolist()
        for y in range(obs_shape[0]):
            xy_onehot = x_onehot + onehot(y, obs_shape[0]).tolist()
            for t in range(horizon):
                if single:
                    for i in range(len(mu_dists)):
                        xytm_onehot = xy_onehot + onehot(t, horizon).tolist() + [0.0] + [mu_dists[i][t, y, x]]
                        inputs[f'{x}-{y}-{t}-m-{i}'] = xytm_onehot
                elif notmu:
                    xyt_onehot = xy_onehot + onehot(t, horizon).tolist() + [0.0] 
                    inputs[f'{x}-{y}-{t}'] = xyt_onehot
                else:
                    xytm_onehot = xy_onehot + onehot(t, horizon).tolist() + [0.0] + [mu_dists[i][t, y, x] for i in range(len(mu_dists))]
                    inputs[f'{x}-{y}-{t}-m'] = xytm_onehot
    return inputs
    

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42, help="set a random seed")
    parser.add_argument("--use_rate", action='store_true', help="")
    
    args = parser.parse_args()
    return args

filename = "disc_actor"
#"/mnt/shunsuke/result/icaart/conventional_method/multi_maze2_airl_deltaxy_1000trajs/seed-42",
#"/mnt/shunsuke/result/icaart/proposed_method/multi_maze2_dxy_mu-divided_value_particle_common-1000-1000-1000_calcMF/seed-42",
pathes = [
            "/mnt/shunsuke/result/icaart/eval/NIG_1000_MFAIRL/seed-4/",
            "/mnt/shunsuke/result/icaart/eval/NIG_1000-1000-1000_calcMF/seed-4",
         ] 
pathnames = [
                "MF-AIRL_NIG",
                "DRMF-AIRL_NIG",
            ] 

update_infos = [
                "179_19",
                "179_19",
              ]

reward_filename = disc_filename = 'disc_reward'
value_filename = 'disc_value'
distance_filename = 'disc_distance'
mu_filename = 'disc_mu'
actor_filename = 'actor'

if __name__ == "__main__":
    args = parse_args()

    from open_spiel.python.mfg.algorithms.discriminator_networks_divided_value import * 
    print(f'len pathes = {len(pathes)}')
    print(f'len pathnames = {len(pathnames)}')
    for ip, target_path in enumerate(pathes):
        for i in range(3):

            fname = actor_filename
            fname = fname + f'{update_infos[ip]}-{i}.pth' 
            fpath = osp.join(target_path, fname)
            assert osp.isfile(fpath), f'isFileError: {fpath}'
    print(f'Checked path: OK')


    res = []
    outputs = []
    for p in range(len(pathes)):
        num_agent = 3

        # Set the seed 
        seed = args.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        print(f"Random seed set as {seed}")

        game = pyspiel.load_game('python_mfg_predator_prey')
        game.new_initial_state()

        mfg_dists = []
        for i in range(num_agent):
            uniform_policy = policy_std.UniformRandomPolicy(game)
            mfg_dist = distribution.DistributionPolicy(game, uniform_policy)
            mfg_dists.append(mfg_dist)
        merge_dist = distribution.MergeDistribution(game, mfg_dists)

        envs = []
        for i in range(num_agent):
            envs.append(rl_environment.Environment(game, mfg_distribution=merge_dist, mfg_population=i))
            envs[-1].seed(args.seed)

        env = envs[0]
        horizon = env.game.get_parameters()['horizon']
        nacs = env.action_spec()['num_actions']
        nobs = env.observation_spec()['info_state'][0]

        agents = []
        actor_models = []
        ppo_policies = []
        mfg_dists = []
        device = "cpu"
        for i in range(num_agent):
            agent = Agent(nobs, nacs).to(device)
            actor_model = agent.actor
            critic_model = agent.critic

            
            actor_path = os.path.join(pathes[p], "actor"+update_infos[p]+f"-{i}.pth")
            actor_model.load_state_dict(torch.load(actor_path))
            actor_model.eval()

            critic_path = os.path.join(pathes[p], "critic"+update_infos[p]+f"-{i}.pth")
            critic_model.load_state_dict(torch.load(critic_path))
            critic_model.eval()
            print("load actor model from", actor_path)

            agents.append(agent)
            actor_models.append(actor_model)

            ppo_policies.append(PPOpolicy(game, agent, None, device))
            mfg_dist = distribution.DistributionPolicy(game, ppo_policies[-1])
            mfg_dists.append(mfg_dist)

        merge_dist = distribution.MergeDistribution(game, mfg_dists)
        for env in envs:
          env.update_mfg_distribution(merge_dist)
        size = envs[0].game.get_parameters()['size']

        mu_dists= [np.zeros((horizon,size,size)) for _ in range(num_agent)]
        for k,v in merge_dist.distribution.items():
            if "mu" in k:
                tt = k.split(",")
                pop = int(tt[0][-1])
                t = int(tt[1].split('=')[1].split('_')[0])
                xy = tt[2].split(" ")
                x = int(xy[1].split("[")[-1])
                y = int(xy[2].split("]")[0])
                mu_dists[pop][t,y,x] = v


        inputs = create_rew_input([size, size], nacs, horizon, mu_dists, False, False, state_only=False)

        save_path = os.path.join(pathes[p], filename+str(update_infos[p]))

        true_reward, true_reward_xy, true_reward_mu = calc_true_reward([size, size], horizon, mu_dists)

        nig = np.array([true_reward[idx]*mu_dists[idx] for idx in range(num_agent)])
        print([f"{np.round(np.nanmean(nig[i]), 4)}" for i in range(len(nig))])
        path = osp.join(pathes[p], f"nig-{update_infos[p]}.pkl")
        pkl.dump(nig , open(path, 'wb'))

        path = osp.join(save_path + f'-true_reward.gif')
        labels = [f'Group {i}' for i in range(num_agent)]
        multi_render(true_reward, path, labels, use_kde=False)

        path = osp.join(save_path + f'-true_reward_xy.gif')
        multi_render(true_reward_xy, path, labels, use_kde=False)

        path = osp.join(save_path + f'-true_reward_mu.gif')
        multi_render(true_reward_mu, path, labels, use_kde=False)



