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
from open_spiel.python.mfg.algorithms.discriminator import Discriminator
from open_spiel.python.mfg.algorithms.mfg_ppo import Agent, PPOpolicy

plt.rcParams["animation.ffmpeg_path"] = "/usr/bin/ffmpeg"

def multi_render_reward(size, nacs, horizon, inputs, discriminator, pop, save=False, filename="agent_dist"):
    # this functions is used to generate an animated video of the distribuiton propagating throught the game 
    rewards = np.zeros((horizon, size, size, nacs))

    for t in range(horizon):
        for x in range(size):
            for y in range(size):
                obs_mu = inputs[f"{x}-{y}-{t}-m"]
                obs_mu = np.array([obs_mu for _ in range(nacs)])
                reward = discriminator.get_reward(
                    torch.from_numpy(obs_mu).to(torch.float32),
                    torch.from_numpy(multionehot(np.arange(nacs), nacs)).to(torch.int64),
                    None, None,
                    discrim_score=False) # For competitive tasks, log(D) - log(1-D) empirically works better (discrim_score=True)
                for a in range(nacs):
                    rewards[t, y, x, a] = reward[a]

    if save:
        for a in range(nacs):
            fig = plt.figure(figsize=(8,8))
            plt.axis("off")
            ims = [[plt.imshow(img, animated=True)] for img in rewards[:, :, :, a]]
            ani = animation.ArtistAnimation(fig, ims, blit=True, interval = 200)

            #  pos: [1, 1] = [right, down] (0,0 left up, size,size right down)
            #  0: np.array([0, 0]), stop
            #  1: np.array([1, 0]), right
            #  2: np.array([0, 1]), down
            #  3: np.array([0, -1]), up
            #  4: np.array([-1, 0]), left
            action_str = ["stop", "right", "down", "up", "left"]
            path = filename + f'-{action_str[a]}.mp4' 
            plt.title(f'The reward of Group {pop} ({action_str[a]})')
            ani.save(path, writer="ffmpeg", fps=5)
            plt.close()
            print(f"Save {path}")

        fig = plt.figure(figsize=(8,8))
        plt.axis("off")
        ims = [[plt.imshow(img, animated=True)] for img in np.mean(rewards, axis=3)]
        ani = animation.ArtistAnimation(fig, ims, blit=True, interval = 200)
        path = filename + f'-avg.mp4' 
        ani.save(path, writer="ffmpeg", fps=5)
        plt.close()
        print(f"Save {path}")

def create_rew_input(obs_shape, nacs, horizon, mu_dists, state_only=False):
    inputs = {}
    for x in range(obs_shape[1]):
        x_onehot = onehot(x, obs_shape[1]).tolist()
        for y in range(obs_shape[0]):
            xy_onehot = x_onehot + onehot(y, obs_shape[0]).tolist()
            for t in range(horizon):
                xytm_onehot = xy_onehot + onehot(t, horizon).tolist() + [0.0] + [mu_dists[i][t, y, x] for i in range(len(mu_dists))]
                inputs[f'{x}-{y}-{t}-m'] = xytm_onehot
    return inputs
    

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42, help="set a random seed")
    parser.add_argument("--path", type=str, default="/mnt/shunsuke/result/multi_type_maze_airl", help="file path")
    parser.add_argument("--reward_filename", type=str, default="disc_reward250_249", help="file path")
    parser.add_argument("--value_filename", type=str, default="disc_value250_249", help="file path")
    parser.add_argument("--actor_filename", type=str, default="actor250_249", help="file path")
    parser.add_argument("--filename", type=str, default="reward250", help="file path")
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    # Set the seed 
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


    device = torch.device("cpu")
    #distrib_path = os.path.join(args.path, args.distrib_filename)
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

        fname = copy.deepcopy(args.actor_filename)
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

        discriminator = Discriminator(nobs+num_agent, nacs, False, device)
        reward_path = osp.join(args.path, args.reward_filename + f'-{i}.pth')
        value_path = osp.join(args.path, args.value_filename + f'-{i}.pth')
        discriminator.load(reward_path, value_path, use_eval=True)
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


    inputs = create_rew_input([size, size], nacs, horizon, mu_dists, state_only=False)
    save_path = os.path.join(args.path, args.filename)
    for i in range(num_agent):
        multi_render_reward(size, nacs, horizon, inputs, discriminators[i], i, save=True, filename=save_path+f"-{i}")
