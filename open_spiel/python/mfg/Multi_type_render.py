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
import pyspiel
import pickle as pkl

from open_spiel.python.mfg import utils
from open_spiel.python import rl_environment
from open_spiel.python import policy as policy_std
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms.nash_conv import NashConv
from open_spiel.python.mfg.algorithms import policy_value
from open_spiel.python.mfg.algorithms.mfg_ppo import *
from open_spiel.python.mfg.games import factory
from open_spiel.python.mfg import value
import copy
import os.path as osp
from gif_maker import *

plt.rcParams["animation.ffmpeg_path"] = r"/usr/bin/ffmpeg"

def calc_distribution(envs, merge_dist, info_states, save=False, filename="agent_distk"):
    # this functions is used to generate an animated video of the distribuiton propagating throught the game 
    num_agent = len(envs)
    horizon = envs[0].game.get_parameters()['horizon']
    d_size = size = envs[0].game.get_parameters()['size']

    final_dists = []
    final_dists_a = []
    final_dists_p = []
    for idx in range(num_agent):
        agent_dist = np.zeros((horizon,d_size,d_size))
        mu_dist = np.zeros((horizon,d_size,d_size))
        a_dist = np.zeros((horizon,d_size,d_size))
        p_dist = np.zeros((horizon,d_size,d_size))

        for k,v in merge_dist.distribution.items():
            if "mu" in k:
                tt = k.split(",")
                pop = int(tt[0][-1])
                if pop!=idx:
                    continue
                t = int(tt[1].split('=')[1].split('_')[0])
                xy = tt[2].split(" ")
                x = int(xy[1].split("[")[-1])
                y = int(xy[2].split("]")[0])
                mu_dist[t,y,x] = v
            elif "a" in k:
                tt = k.split(",")
                pop = int(tt[0][-1])
                if pop!=idx:
                    continue
                t = int(tt[1].split('=')[1].split('_')[0])
                if t>=40:
                    continue
                xy = tt[2].split(" ")
                x = int(xy[1].split("[")[-1])
                y = int(xy[2].split("]")[0])
                a_dist[t,y,x] = v
            elif "t=" in k:
                tt = k.split(",")
                pop = int(tt[0][-1])
                if pop!=idx:
                    continue
                t = int(tt[1].split('=')[1])
                if t>=40:
                    continue
                xy = tt[2].split(" ")
                x = int(xy[1].split("[")[-1])
                y = int(xy[2].split("]")[0])
                p_dist[t,y,x] = v

        for i in range(horizon):
            obs = info_states[idx][i].tolist()
            obs_x = obs[:size].index(1)
            obs_y = obs[size:2*size].index(1)
            obs_t = obs[2*size:].index(1)
            agent_dist[obs_t,obs_y,obs_x] = 0.02

        #final_dist = agent_dist + mu_dist
        final_dist = mu_dist
        final_dist_a = a_dist
        final_dist_p = p_dist

        final_dists.append(final_dist)
        final_dists_a.append(final_dist_a)
        final_dists_p.append(final_dist_p)

    final_dists = np.array(final_dists)
    final_dists_a = np.array(final_dists_a)
    final_dists_p = np.array(final_dists_p)
    if save:
        multi_render(final_dists[:, :, :], filename+'.gif', [f'Group{i}' for i in range(num_agent)], use_kde=True)
        #multi_render(final_dists_p[:, :, :], filename+'-p.gif', [f'Group{i}' for i in range(num_agent)], use_kde=True)
        #multi_render(final_dists_a[:, :, :], filename+'-a.gif', [f'Group{i}' for i in range(num_agent)], use_kde=True)
    return final_dists



def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42, help="set a random seed")
    parser.add_argument("--path", type=str, default="/mnt/shunsuke/result/multi_type_maze_airl", help="file path")
    parser.add_argument("--filename", type=str, default="actor", help="file path")
    parser.add_argument("--actor_filename", type=str, default="actor1100_1099", help="file path")
    
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
    for i in range(num_agent):
        agent = Agent(nobs, nacs).to(device)
        actor_model = agent.actor

        fname = copy.deepcopy(args.actor_filename)
        fname = fname + f'-{i}.pth' 
        actor_path = os.path.join(args.path, fname)
        actor_model.load_state_dict(torch.load(actor_path))
        actor_model.eval()
        print("load actor model from", actor_path)

        agents.append(agent)
        actor_models.append(actor_model)

        ppo_policies.append(PPOpolicy(game, agent, None, device))
        mfg_dist = distribution.DistributionPolicy(game, ppo_policies[-1])
        mfg_dists.append(mfg_dist)

    merge_dist = distribution.MergeDistribution(game, mfg_dists)
    for env in envs:
      env.update_mfg_distribution(merge_dist)

    # output = model(input_data)
    def get_action(x, actor_model):
        logits = actor_model(x)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action
        
    info_state = []
    ep_ret = 0.0

    steps = envs[0].max_game_length
    info_state = [torch.zeros((steps,agents[i].info_state_size), device=device) for i in range(num_agent)]
    actions = [torch.zeros((steps,), device=device) for _ in range(num_agent)]
    logprobs = [torch.zeros((steps,), device=device) for _ in range(num_agent)]
    rewards = [torch.zeros((steps,), device=device) for _ in range(num_agent)]
    dones = [torch.zeros((steps,), device=device) for _ in range(num_agent)]
    values = [torch.zeros((steps,), device=device) for _ in range(num_agent)]
    entropies = [torch.zeros((steps,), device=device) for _ in range(num_agent)]
    t_actions = [torch.zeros((steps,), device=device) for _ in range(num_agent)]
    t_logprobs = [torch.zeros((steps,), device=device) for _ in range(num_agent)]

    size = envs[0].game.get_parameters()['size']
    step = 0

    time_steps = [envs[i].reset() for i in range(num_agent)]
    while not time_steps[0].last():
        mu = []
        for i in range(num_agent):
            obs = time_steps[i].observations["info_state"][i]
            obs = torch.Tensor(obs).to(device)
            info_state[i][step] = obs
            with torch.no_grad():
                t_action, t_logprob, _, _ = agents[i].get_action_and_value(obs)
                action, logprob, entropy, value = agents[i].get_action_and_value(obs)

            # iteration policy data
            t_logprobs[i][step] = t_logprob
            t_actions[i][step] = t_action
            logprobs[i][step] = logprob
            entropies[i][step] = entropy
            values[i][step] = value
            actions[i][step] = action

            time_steps[i] = envs[i].step([action.item()])


        for i in range(num_agent):
            # episode policy data
            dones[i][step] = time_steps[i].last()
            rewards[i][step] = torch.Tensor(np.array(time_steps[i].rewards[i])).to(device)
        step += 1

    #for i in range(num_agent):
    #    print(f'reward{i}: {np.sum(rewards[i])}')
    save_path = os.path.join(args.path, f"reward.pkl")
    print(f'Saved as {save_path}')
    pkl.dump(rewards, open(save_path, 'wb'))
    print(rewards)
    for i in range(num_agent):
        reward_np = np.array(rewards[i])
        print(f'cumulative reward {i}: {np.sum(reward_np)}')
    save_path = os.path.join(args.path, f"{args.filename}")
    final_dists = calc_distribution(envs, merge_dist, info_state, save=True, filename=save_path)

