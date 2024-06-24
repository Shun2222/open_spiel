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
from open_spiel.python.mfg.algorithms.mfg_ppo import *
from open_spiel.python.mfg.games import factory
from open_spiel.python.mfg import value
from open_spiel.python.mfg.algorithms.mfg_ppo import Agent, PPOpolicy
from gif_maker import *

plt.rcParams["font.size"] = 20
plt.rcParams["animation.ffmpeg_path"] = "/usr/bin/ffmpeg"

def multi_render_reward_nets(size, nacs, horizon, inputs, discriminator, save=False, filename="agent_dist"):
    from open_spiel.python.mfg.algorithms.discriminator_networks import Discriminator

    # this functions is used to generate an animated video of the distribuiton propagating throught the game 
    num_nets = discriminator.get_num_nets()
    rewards = np.zeros((horizon, size, size, nacs))
    output_rewards = [np.zeros((horizon, size, size, nacs)) for _ in range(num_nets)]

    for t in range(horizon):
        for x in range(size):
            for y in range(size):
                for a in range(nacs):
                    obs_input = inputs[f"{x}-{y}-{t}-{a}-m"]
                
                    reward, outputs = discriminator.get_reward(
                        obs_input,
                        None, None, None,
                        discrim_score=False,
                        only_rew=False,
                        weighted_rew=True) # For competitive tasks, log(D) - log(1-D) empirically works better (discrim_score=True)

                    rewards[t, y, x, a] = reward
                    for i in range(num_nets):
                        output_rewards[i][t, y, x, a] = outputs[i]
    return rewards, output_rewards

def multi_render_reward(size, nacs, horizon, inputs, discriminator, pop, single, notmu, save=False, filename="agent_dist"):
    from open_spiel.python.mfg.algorithms.discriminator import Discriminator

    # this functions is used to generate an animated video of the distribuiton propagating throught the game 
    rewards = np.zeros((horizon, size, size, nacs))
    dist_rewards = np.zeros((horizon, size, size, nacs))
    mu_rewards = np.zeros((horizon, size, size, nacs))

    for t in range(horizon):
        for x in range(size):
            for y in range(size):
                if single:
                    obs_input = inputs[f"{x}-{y}-{t}-m-{pop}"]
                    obs_input = np.array([obs_input for _ in range(nacs)])
                elif notmu:
                    obs_input = inputs[f"{x}-{y}-{t}"]
                    obs_input = np.array([obs_input for _ in range(nacs)])
                else:
                    obs_input = inputs[f"{x}-{y}-{t}-m"]
                    obs_input = np.array([obs_input for _ in range(nacs)])
                
                reward = discriminator.get_reward(
                    torch.from_numpy(obs_input).to(torch.float32),
                    torch.from_numpy(multionehot(np.arange(nacs), nacs)).to(torch.int64),
                    None, None,
                    discrim_score=False,
                    ) # For competitive tasks, log(D) - log(1-D) empirically works better (discrim_score=True)

                for a in range(nacs):
                    rewards[t, y, x, a] = reward[a]
    if save:
        datas = [rewards[:, :, :, a] for a in range(nacs)]
        action_str = ["stop", "right", "down", "up", "left"]
        path = filename + f'-all-action.gif' 
        print(np.array(datas).shape)
        multi_render(datas, path, action_str, use_kde=False)
        return rewards


def create_rew_input(obs_shape, nacs, horizon, mu_dists, single, notmu, state_only=False):
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
    parser.add_argument("--path", type=str, default="/mnt/shunsuke/result/0614/multi_maze2_airl_basicfuncs_episode1", help="file path")
    parser.add_argument("--update_eps", type=str, default=r"134_134", help="file path")
    parser.add_argument("--distance_filename", type=str, default="disc_distance", help="file path")
    parser.add_argument("--mu_filename", type=str, default="disc_mu", help="file path")
    parser.add_argument("--reward_filename", type=str, default="disc_reward", help="file path")
    parser.add_argument("--value_filename", type=str, default="disc_value", help="file path")
    parser.add_argument("--actor_filename", type=str, default="actor", help="file path")
    parser.add_argument("--filename", type=str, default="reward", help="file path")
    parser.add_argument("--single", action='store_true')
    parser.add_argument("--notmu", action='store_true')
    parser.add_argument("--basicfuncs", action='store_true')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    single = args.single
    notmu = args.notmu
    basicfuncs = args.basicfuncs

    if basicfuncs:
        from open_spiel.python.mfg.algorithms.discriminator_basicfuncs import Discriminator, divide_obs
    else:
        from open_spiel.python.mfg.algorithms.discriminator import Discriminator

    # Set the seed 
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

    update_eps_info = f'{args.update_eps}'
    device = torch.device("cpu")
    #distrib_path = os.path.join(args.path, args.distrib_filename+update_eps_info)
    #distrib = pkl.load(open(distrib_path, "rb"))
    #print("load actor model from", distrib_path)

    # Create the game instance 
    game = pyspiel.load_game('python_mfg_predator_prey')
    game.new_initial_state()

    num_agent = 3
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
    nacs = env.action_spec()['num_actions']
    nobs = env.observation_spec()['info_state'][0]

    agents = []
    actor_models = []
    ppo_policies = []
    mfg_dists = []
    discriminators = []
    for i in range(num_agent):
        agent = Agent(nobs, nacs).to(device)
        actor_model = agent.actor

        fname = copy.deepcopy(args.actor_filename+update_eps_info)
        fname = fname + f'-{i}.pth' 
        actor_path = osp.join(args.path, fname)
        actor_model.load_state_dict(torch.load(actor_path))
        actor_model.eval()
        print("load actor model from", actor_path)

        agents.append(agent)
        actor_models.append(actor_model)

        ppo_policies.append(PPOpolicy(game, agent, None, device))
        mfg_dist = distribution.DistributionPolicy(game, ppo_policies[-1])
        mfg_dists.append(mfg_dist)

        if single:
            discriminator = Discriminator(nobs+1, nacs, False, device)
        elif notmu:
            discriminator = Discriminator(nobs, nacs, False, device)
        elif basicfuncs_time:
            discriminator = Discriminator(num_agent, 2, nobs+num_agent, nacs, False, device)
        else:
            discriminator = Discriminator(nobs+num_agent, nacs, False, device)
        reward_path = osp.join(args.path, args.reward_filename+update_eps_info + f'-{i}.pth')
        value_path = osp.join(args.path, args.value_filename+update_eps_info + f'-{i}.pth')
        if basicfuncs:
            distance_path = osp.join(args.path, args.distance_filename+update_eps_info + f'-{i}.pth')
            mu_path = osp.join(args.path, args.mu_filename+update_eps_info + f'-{i}.pth')
            discriminator.load(distance_path, mu_path, reward_path, value_path, use_eval=True)
            discriminator.print_weights()
        else:
            distance_path = osp.join(args.path, args.distance_filename+update_eps_info + f'-{i}.pth')
            mu_path = osp.join(args.path, args.mu_filename+update_eps_info + f'-{i}.pth')
            discriminator.load(reward_path, value_path, use_eval=True)
            print(f'')
        discriminators.append(discriminator)

    merge_dist = distribution.MergeDistribution(game, mfg_dists)
    for env in envs:
      env.update_mfg_distribution(merge_dist)

    
    horizon = env.game.get_parameters()['horizon']
    size = env.game.get_parameters()['size']

    agent_dist = np.zeros((horizon,size,size))
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


    inputs = create_rew_input([size, size], nacs, horizon, mu_dists, single, notmu, state_only=False)
    save_path = os.path.join(args.path, args.filename+str(args.update_eps))
    datas = []
    dist_datas = []
    mu_datas = []
    for i in range(num_agent):
        if basicfuncs:
            rewards, dist_rew, mu_rew = multi_render_reward(size, nacs, horizon, inputs, discriminators[i], i, single, notmu, basicfuncs, basicfuncs_time, save=True, filename=save_path+f"-{i}")
            dist_datas.append(np.mean(dist_rew, axis=3))
            mu_datas.append(np.mean(mu_rew, axis=3))
        else:
            rewards = multi_render_reward(size, nacs, horizon, inputs, discriminators[i], i, single, notmu, basicfuncs, basicfuncs_time, save=True, filename=save_path+f"-{i}")
        datas.append(np.mean(rewards, axis=3))
    path = osp.join(save_path + f'-mean.gif')
    labels = [f'Group {i}' for i in range(num_agent)]
    print(np.array(datas).shape)
    multi_render(datas, path, labels, use_kde=False)
    if basicfuncs:
        path = osp.join(save_path + f'-mean-dist.gif')
        multi_render(dist_datas, path, labels, use_kde=False)

        path = osp.join(save_path + f'-mean-mu.gif')
        multi_render(mu_datas, path, labels, use_kde=False)

    for i in range(num_agent):
        plt.rcParams["font.size"] = 8 
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(1, 1, 1)
        points = datas[i]
        col = 1
        for s in range(len(points[0].shape)):
            col *= points[0].shape[s]
        points = points.reshape(len(points), col).T
        bp = ax.boxplot(points)
        plt.xlabel(r"$\mu_{time}$")
        save_path = os.path.join(args.path, args.filename+f'-mutime-box-{i}.png')
        plt.savefig(save_path)
        plt.close()
        print(f'saved {save_path} ')
        if basicfuncs:
            fig = plt.figure(figsize=(16, 12))
            ax = fig.add_subplot(1, 1, 1)
            points = dist_datas[i]
            col = 1
            for s in range(len(points[0].shape)):
                col *= points[0].shape[s]
            points = points.reshape(len(points), col).T
            bp = ax.boxplot(points)
            plt.xlabel(r"$\mu_{time}$")
            plt.ylabel(r"Distance Reward")
            save_path = os.path.join(args.path, args.filename+f'-mutime-box-dist-{i}.png')
            plt.savefig(save_path)
            plt.close()
            print(f'saved {save_path} ')

            fig = plt.figure(figsize=(16, 12))
            ax = fig.add_subplot(1, 1, 1)
            points = mu_datas[i]
            col = 1
            for s in range(len(points[0].shape)):
                col *= points[0].shape[s]
            points = points.reshape(len(points), col).T
            bp = ax.boxplot(points)
            plt.xlabel(r"$\mu_{time}$")
            plt.ylabel(r"Mu Reward")
            save_path = os.path.join(args.path, args.filename+f'-mutime-box-mu-{i}.png')
            plt.savefig(save_path)
            plt.close()
            print(f'saved {save_path} ')

        figsizes = [(16, 12), (64, 12)]
        fontsizes = [8, 24]
        for j in range(len(figsizes)):
            plt.rcParams["font.size"] = fontsizes[j]
            fig = plt.figure(figsize=figsizes[j])
            ax = fig.add_subplot(1, 1, 1)
            points = datas[i]
            col = 1
            for s in range(len(points[0].shape)):
                col *= points[0].shape[s]
            points = points.reshape(len(points), col)
            bp = ax.boxplot(points)
            plt.xlabel(r"State")
            save_path = os.path.join(args.path, args.filename+f'-box-{j}-{i}.png')
            plt.savefig(save_path)
            plt.close()
            print(f'saved {save_path} ')
            if basicfuncs:
                fig = plt.figure(figsize=figsizes[j])
                ax = fig.add_subplot(1, 1, 1)
                points = dist_datas[i]
                col = 1
                for s in range(len(points[0].shape)):
                    col *= points[0].shape[s]
                points = points.reshape(len(points), col)
                bp = ax.boxplot(points)
                plt.xlabel(r"State")
                plt.ylabel(r"Distance Reward")
                save_path = os.path.join(args.path, args.filename+f'-box-dist-{j}-{i}.png')
                plt.savefig(save_path)
                plt.close()
                print(f'saved {save_path} ')

                fig = plt.figure(figsize=figsizes[j])
                ax = fig.add_subplot(1, 1, 1)
                points = mu_datas[i]
                col = 1
                for s in range(len(points[0].shape)):
                    col *= points[0].shape[s]
                points = points.reshape(len(points), col)
                bp = ax.boxplot(points)
                plt.xlabel(r"State")
                plt.ylabel(r"Mu Reward")
                save_path = os.path.join(args.path, args.filename+f'-box-mu-{j}-{i}.png')
                plt.savefig(save_path)
                plt.close()
                print(f'saved {save_path} ')
