import os
# 
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=6 Mainly controlles the number of spawned threateds 
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=6

import argparse

import pickle as pkl
from distutils.util import strtobool
import time
import logging
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

def render_reward(size, nacs, horizon, inputs, discriminator, save=False, filename="agent_dist"):
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
            path = filename + f'-act{a}.mp4' 
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

def create_rew_input(obs_shape, nacs, horizon, mu_dist, state_only=False):
    inputs = {}
    for x in range(obs_shape[1]):
        x_onehot = onehot(x, obs_shape[1]).tolist()
        for y in range(obs_shape[0]):
            xy_onehot = x_onehot + onehot(y, obs_shape[0]).tolist()
            for t in range(horizon):
                xytm_onehot = xy_onehot + onehot(t, horizon).tolist() + [0.0] + [mu_dist[t, y, x]]
                inputs[f'{x}-{y}-{t}-m'] = xytm_onehot
    return inputs
    

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42, help="set a random seed")
    parser.add_argument("--path", type=str, default="/mnt/shunsuke/mfg_result/batch-test/batch400/", help="file path")
    parser.add_argument("--game-setting", type=str, default="crowd_modelling_2d_four_rooms", help="Set the game to benchmark options:(crowd_modelling_2d_four_rooms) and (crowd_modelling_2d_maze)")
    parser.add_argument("--reward_dir", type=str, default="/mnt/shunsuke/mfg_result/batch-test/batch400/disc_reward.pth", help="file path")
    parser.add_argument("--value_dir", type=str, default="/mnt/shunsuke/mfg_result/batch-test/batch400/disc_value.pth", help="file path")
    parser.add_argument("--actor_dir", type=str, default="/mnt/shunsuke/mfg_result/batch-test/batch400/actor.pth", help="file path")
    parser.add_argument("--critic_dir", type=str, default="/mnt/shunsuke/mfg_result/batch-test/batch400/critic.pth", help="file path")
    
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

    game_name = "mfg_crowd_modelling_2d"
    game = factory.create_game_with_setting(game_name, args.game_setting)

    uniform_policy = policy_std.UniformRandomPolicy(game)
    distrib = mfg_dist = distribution.DistributionPolicy(game, uniform_policy)
    env = rl_environment.Environment(game, mfg_distribution=distrib, mfg_population=0)

    nacs = env.action_spec()['num_actions']
    nobs = env.observation_spec()['info_state'][0]

    agent = Agent(nobs, nacs).to(device)
    actor_model = agent.actor
    actor_model.load_state_dict(torch.load(args.actor_dir))
    actor_model.eval()
    critic_model = agent.critic
    critic_model.load_state_dict(torch.load(args.critic_dir))
    critic_model.eval()
    ppo_policy = PPOpolicy(game, agent, None, device)
    distrib = distribution.DistributionPolicy(game, ppo_policy)

    env = rl_environment.Environment(game, mfg_distribution=distrib, mfg_population=0)
    env.seed(args.seed)

    discriminator = Discriminator(nobs+1, nacs, False, device)
    discriminator.load(args.reward_dir, args.value_dir, use_eval=True)
    
    horizon = env.game.get_parameters()['horizon']
    size = env.game.get_parameters()['size']

    mu_dist = np.zeros((horizon,size,size))
    for k,v in distrib.distribution.items():
        if "mu" in k:
            tt = k.split("_")[0].split(",")
            x = int(tt[0].split("(")[-1])
            y = int(tt[1].split()[-1])
            t = int(tt[2].split()[-1].split(")")[0])
            mu_dist[t,y,x] = v

    inputs = create_rew_input([size, size], nacs, horizon, mu_dist, state_only=False)

    save_path = os.path.join(args.path, "agent_dist")
    render_reward(size, nacs, horizon, inputs, discriminator, save=True, filename=save_path)
