import os
# 
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=6 Mainly controlles the number of spawned threateds 
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=6

import argparse
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
from open_spiel.python import rl_environment
from open_spiel.python import policy as policy_std
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms.nash_conv import NashConv
from open_spiel.python.mfg.algorithms import policy_value
from open_spiel.python.mfg.algorithms.mfg_ppo import *
from open_spiel.python.mfg.games import factory
from open_spiel.python.mfg import value

plt.rcParams["animation.ffmpeg_path"] = "/usr/bin/ffmpeg"

def render(env, game_name, distrib, info_state, save=False, filename="agent_dist.mp4"):
    # this functions is used to generate an animated video of the distribuiton propagating throught the game 
    horizon = env.game.get_parameters()['horizon']
    size = env.game.get_parameters()['size']
    if game_name == "maze":
        d_size = 21
    else:
        d_size = 13
    agent_dist = np.zeros((horizon,d_size,d_size))
    mu_dist = np.zeros((horizon,d_size,d_size))


    for k,v in distrib.distribution.items():
        if "mu" in k:
            tt = k.split("_")[0].split(",")
            x = int(tt[0].split("(")[-1])
            y = int(tt[1].split()[-1])
            t = int(tt[2].split()[-1].split(")")[0])
            mu_dist[t,y,x] = v

    for i in range(horizon):
        obs = info_state[i].tolist()
        obs_x = obs[:size].index(1)
        obs_y = obs[size:2*size].index(1)
        obs_t = obs[2*size:].index(1)
        agent_dist[obs_t,obs_y,obs_x] = 0.02

    final_dist = agent_dist + mu_dist

    if save:
        fig = plt.figure(figsize=(8,8))
        plt.axis("off")
        ims = [[plt.imshow(img, animated=True)] for img in final_dist]

        ani = animation.ArtistAnimation(fig, ims, blit=True, interval = 200)

        ani.save(filename, writer="ffmpeg", fps=5)

        plt.close()

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42, help="set a random seed")
    parser.add_argument("--path", type=str, default="/mnt/shunsuke/mfg_result", help="file path")
    parser.add_argument("--num_obs", type=int, default=67, help="set a random seed")
    parser.add_argument("--num_acs", type=int, default=5, help="set a random seed")
    parser.add_argument("--game-setting", type=str, default="crowd_modelling_2d_four_rooms", help="Set the game to benchmark options:(crowd_modelling_2d_four_rooms) and (crowd_modelling_2d_maze)")
    parser.add_argument("--distrib_filename", type=str, default="distrib.pkl", help="file path")
    parser.add_argument("--actor_filename", type=str, default="actor.pkl", help="file path")
    
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


    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    #distrib_path = os.path.join(args.path, args.distrib_filename)
    #distrib = pkl.load(open(distrib_path, "rb"))
    #print("load actor model from", distrib_path)

    agent = Agent(args.num_obs, args.num_acs).to(device)
    actor_model = agent.actor
    actor_path = os.path.join(args.path, args.actor_filename)
    actor_model.load_state_dict(torch.load(actor_path))
    print("load actor model from", actor_path)
    actor_model.eval()


    game_name = "mfg_crowd_modelling_2d"
    game = factory.create_game_with_setting(game_name, args.game_setting)

    ppo_policy = PPOpolicy(game, agent, None, device)
    mfg_dist = distribution.DistributionPolicy(game, ppo_policy)
    env = rl_environment.Environment(game, mfg_distribution=mfg_dist, mfg_population=0)

    env.seed(args.seed)

    # output = model(input_data)
    def get_action(x):
        logits = actor_model(x)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action
        

    info_state = []
    ep_ret = 0.0

    time_step = env.reset()
    while not time_step.last():
        obs = time_step.observations["info_state"][0]
        obs_pth = torch.Tensor(obs).to(device)
        action = get_action(obs_pth)
        time_step = env.step([action.item()])
        rewards = time_step.rewards[0]
        dist = env.mfg_distribution

        info_state.append(np.array(obs))
        ep_ret += rewards


    print(f"ep_ret: {ep_ret}")
    save_path = os.path.join(args.path, "agent_dist.mp4")
    render(env, game_name, mfg_dist, info_state, save=True, filename=save_path)
    print(f"Save {save_path}")
