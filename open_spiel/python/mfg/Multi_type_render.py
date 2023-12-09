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

def render(envs, merge_dist, info_state, save=False, filename="agent_dist.mp4"):
    # this functions is used to generate an animated video of the distribuiton propagating throught the game 
    num_agent = len(envs)
    horizon = envs[i].game.get_parameters()['horizon']
    d_size = size = envs[i].game.get_parameters()['size']

    final_dists = []
    for idx in range(num_agent):
        agent_dist = np.zeros((horizon,d_size,d_size))
        mu_dist = np.zeros((horizon,d_size,d_size))

        for k,v in distrib.distribution.items():
            if "mu" in k:
                tt = k.split(",")
                pop = int(tt[0][-1])
                t = int(tt[1][3])
                xy = tt[2].split(" ")
                x = int(xy[1].split("[")[-1])
                y = int(xy[2].split("]")[0])
                mu_dist[t,y,x] = v

        for i in range(horizon):
            obs = info_states[idx][i].tolist()
            obs_x = obs[:size].index(1)
            obs_y = obs[size:2*size].index(1)
            obs_t = obs[2*size:].index(1)
            agent_dist[obs_t,obs_y,obs_x] = 0.02

        final_dist = agent_dist + mu_dist
        final_dists.append(final_dist)

        if save:
            fig = plt.figure(figsize=(8,8))
            plt.axis("off")
            ims = [[plt.imshow(img, animated=True)] for img in final_dist]
            ani = animation.ArtistAnimation(fig, ims, blit=True, interval = 200)
            fname = filename[0:-5] + str(idx) + filename[-4:-1]
            ani.save(fname, writer="ffmpeg", fps=5)
            plt.close()

    def parse_args():

        parser = argparse.ArgumentParser()

        parser.add_argument("--seed", type=int, default=42, help="set a random seed")
        parser.add_argument("--path", type=str, default="/mnt/shunsuke/mfg_result/batch-test/batch400", help="file path")
        parser.add_argument("--game-setting", type=str, default="crowd_modelling_2d_four_rooms", help="Set the game to benchmark options:(crowd_modelling_2d_four_rooms) and (crowd_modelling_2d_maze)")
        parser.add_argument("--distrib_filename", type=str, default="distrib.pth", help="file path")
        parser.add_argument("--actor_filename", type=str, default="actor.pth", help="file path")
        
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
        actor_filenames = [args.actor_filename for _ in range(num_agent)] #TODO
        for i in range(num_agent):J
            agent = Agent(nobs, nacs).to(device)
            actor_model = agent.actor
            actor_path = os.path.join(args.path, actor_filenames[i])
            actor_model.load_state_dict(torch.load(actor_path))
            actor_model.eval()
            print("load actor model from", actor_path)

            agents.append(agent)
            actor_models.append(actor_model)

            ppo_policies.append(PPOpolicy(game, agent, None, device))
            mfg_dist = distribution.DistributionPolicy(game, ppo_policies[-1])
            mfg_dists.append(mfg_dist)

        merge_dist = distribution.mergedistribution(game, mfg_dists)
        conv_dist = convert_distrib(envs, merge_dist)
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

        info_state = [torch.zeros((steps,iter_agents[i].info_state_size), device=device) for i in range(num_agent)]
        obs_mu = [torch.zeros((steps, iter_agents[i].info_state_size+3), device=device) for i in range(num_agent)]
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
                obs = time_steps[i].observations["info_state"][0]
                obs = torch.Tensor(obs).to(device)
                info_state[i][step] = obs
                with torch.no_grad():
                    t_action, t_logprob, _, _ = iter_agents[i].get_action_and_value(obs)
                    action, logprob, entropy, value = eps_agents[i].get_action_and_value(obs)

                # iteration policy data
                t_logprobs[i][step] = t_logprob
                t_actions[i][step] = t_action
                logprobs[i][step] = logprob
                entropies[i][step] = entropy
                values[i][step] = value
                actions[i][step] = action

                time_steps[i] = envs[i].step([action.item()])

                obs_list = list(obs.detach().numpy())
                obs_x = obs_list[:size].index(1)
                obs_y = obs_list[size:2*size].index(1)
                obs_t = obs_list[2*size:].index(1)
                mu.append(conv_dist[i][obs_t, obs_y, obs_x])

            for i in range(num_agent):
                # episode policy data
                dones[i][step] = time_steps[i].last()
                rewards[i][step] = torch.Tensor(np.array(time_steps[i].rewards[i])).to(device)
            ob_mu = obs_list 
            ob_mu += mu
            obs_mu[i][step] = torch.Tensor(ob_mu).to(device)
            step += 1

        for i in range(num_agent):
            print(f'Exp. Ret{i}: {rewards[i]}')
        save_path = os.path.join(args.path, "agent_dist.mp4")
        render(envs, merge_dist, info_state, save=True, filename=save_path)
        print(f"Saved in {args.path}")
