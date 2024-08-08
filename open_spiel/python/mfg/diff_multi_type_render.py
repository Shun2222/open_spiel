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
import pyspiel
import pickle as pkl

from open_spiel.python.mfg import utils
from open_spiel.python import rl_environment
from open_spiel.python import policy as policy_std
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms.nash_conv import NashConv
from open_spiel.python.mfg.algorithms import policy_value
from open_spiel.python.mfg.algorithms.multi_type_mfg_ppo import *
from open_spiel.python.mfg.games import factory
from open_spiel.python.mfg import value
import copy
from gif_maker import *
from scipy.spatial import distance
from scipy.stats import spearmanr
from scipy.special import kl_div
from diff_utils import *

plt.rcParams["animation.ffmpeg_path"] = r"/usr/bin/ffmpeg"
plt.rcParams["font.size"] = 20

def onehot(value, depth):
    a = np.zeros([depth])
    a[value] = 1
    return a


def multionehot(values, depth):
    a = np.zeros([values.shape[0], depth])
    for i in range(values.shape[0]):
        a[i, int(values[i])] = 1
    return a

def calc_distribution(envs, merge_dist, info_states, save=False, filename="agent_distk.mp4"):
    # this functions is used to generate an animated video of the distribuiton propagating throught the game 
    num_agent = len(envs)
    horizon = envs[0].game.get_parameters()['horizon']
    d_size = size = envs[0].game.get_parameters()['size']

    final_dists = []
    for idx in range(num_agent):
        agent_dist = np.zeros((horizon,d_size,d_size))
        mu_dist = np.zeros((horizon,d_size,d_size))

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

        for i in range(horizon):
            obs = info_states[idx][i].tolist()
            obs_x = obs[:size].index(1)
            obs_y = obs[size:2*size].index(1)
            obs_t = obs[2*size:].index(1)
            agent_dist[obs_t,obs_y,obs_x] = 0.02

        #final_dist = agent_dist + mu_dist
        final_dist = mu_dist
        final_dists.append(final_dist)

        if save:
            fig = plt.figure(figsize=(8,8))
            plt.axis("off")
            ims = [[plt.imshow(img, animated=True)] for img in final_dist]
            ani = animation.ArtistAnimation(fig, ims, blit=True, interval = 200)
            fname = filename[0:-5] + str(idx) + filename[-4:]
            ani.save(fname, writer="ffmpeg", fps=5)
            print(f"Saved as {fname}")
            plt.close()
    return final_dists



def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42, help="set a random seed")
    
    args = parser.parse_args()
    return args

# 4items: args actor_filename, filename, pathes, pathnames
filename = "actor"

pathes = [
            "/mnt/shunsuke/result/0726/multi_maze2_expert",
            "/mnt/shunsuke/result/0726/multi_maze2_dxy_mu_diversity3",
         ] 
           # "/mnt/shunsuke/result/0627/multi_maze2_ppo_s_mu_a",
           # "/mnt/shunsuke/result/0627/multi_maze2_ppo_s_mu_a_srew",
           # "/mnt/shunsuke/result/0627/multi_maze2_ppo_s_mu_a_murew",
           # "/mnt/shunsuke/result/0627/multi_maze2_ppo_s_mu_a_arew",
           # "/mnt/shunsuke/result/0627/multi_maze2_ppo_sa_mu",
           # "/mnt/shunsuke/result/0627/multi_maze2_ppo_sa_mu_sarew",
           # "/mnt/shunsuke/result/0627/multi_maze2_ppo_sa_mu_murew",
           # "/mnt/shunsuke/result/0627/multi_maze2_ppo_s_mua",
           # "/mnt/shunsuke/result/0627/multi_maze2_ppo_s_mua_srew",
           # "/mnt/shunsuke/result/0627/multi_maze2_ppo_s_mua_muarew",
           # "/mnt/shunsuke/result/0627/multi_maze2_ppo_dxy_mu_a",
           # "/mnt/shunsuke/result/0627/multi_maze2_ppo_dxy_mu_a_dxyrew",
           # "/mnt/shunsuke/result/0627/multi_maze2_ppo_dxy_mu_a_murew",
           # "/mnt/shunsuke/result/0627/multi_maze2_ppo_dxy_mu_a_arew",
           # "/mnt/shunsuke/result/0627/multi_maze2_ppo_dxya_mu",
           # "/mnt/shunsuke/result/0627/multi_maze2_ppo_dxya_mu_dxyarew",
           # "/mnt/shunsuke/result/0627/multi_maze2_ppo_dxya_mu_murew",
           # "/mnt/shunsuke/result/0627/multi_maze2_ppo_dxy_mua",
           # "/mnt/shunsuke/result/0627/multi_maze2_ppo_dxy_mua_dxyrew",
           # "/mnt/shunsuke/result/0627/multi_maze2_ppo_dxy_mua_muarew",
            #"/mnt/shunsuke/result/0627/multi_maze2_airl",
            #"/mnt/shunsuke/result/0627/multi_maze2_1hidden_mfairl",
            #"/mnt/shunsuke/result/0627/multi_maze2_2hidden_mfairl",
            # "/mnt/shunsuke/result/0627/multi_maze2_s_mu_a",
            # "/mnt/shunsuke/result/0627/multi_maze2_sa_mu",
            # "/mnt/shunsuke/result/0627/multi_maze2_s_mua",
            # "/mnt/shunsuke/result/0627/multi_maze2_dxy_mu_a",
            # "/mnt/shunsuke/result/0627/multi_maze2_dxya_mu",
            # "/mnt/shunsuke/result/0627/multi_maze2_dxy_mua",
            #"/mnt/shunsuke/result/0627/multi_maze2_mfairl_time",
pathnames = [
                "expert",
                "dxy_mu-diversity3",
            ] 
                #"ppo_s_mu_a",
                #"ppo_s_mu_a_srew",
                #"ppo_s_mu_a_murew",
                #"ppo_s_mu_a_arew",
                #"ppo_sa_mu",
                #"ppo_sa_mu_sarew",
                #"ppo_sa_mu_murew",
                #"ppo_s_mua",
                #"ppo_s_mua_srew",
                #"ppo_s_mua_muarew",
                #"ppo_dxy_mu_a",
                #"ppo_dxy_mu_a_dxyrew",
                #"ppo_dxy_mu_a_murew",
                #"ppo_dxy_mu_a_arew",
                #"ppo_dxya_mu",
                #"ppo_dxya_mu_dxyarew",
                #"ppo_dxya_mu_murew",
                #"ppo_dxy_mua",
                #"ppo_dxy_mua_dxyrew",
                #"ppo_dxy_mua_muarew",
                #"MF-AITL",
                #"MF-AITL_2hidden",
                #"MF-AITL_3hidden",
                #"MF-AITL_s_mu_a",
                #"MF-AITL_sa_mu",
                #"MF-AITL_s_mua",
                #"MF-AITL_dxy_mu_a",
                #"MF-AITL_dxya_mu",
                #"MF-AITL_dxy_mua",

filenames = [
                "50_19",
                "1115_9",
            ]
weights = [[1.0, 1.0]]

                    #"actor99_19",
                    #"actor200_2",

if __name__ == "__main__":
    args = parse_args()
    diversity_count = 0

    res_final_dists = []
    gifMaker = GifMaker()
    for ip, target_path in enumerate(pathes):
        for i in range(3):
            fname = copy.deepcopy('actor'+filenames[ip])
            fname = fname + f'-{i}.pth' 
            fpath = osp.join(target_path, fname)
            assert osp.isfile(fpath), f'isFileError: {fpath}'
    print(f'Checked pathed: OK')


    for ip, target_path in enumerate(pathes):
        from open_spiel.python.mfg.algorithms.multi_type_mfg_ppo_diversity import is_diversity
        is_diversity_ppo = is_diversity(target_path)
        if is_diversity_ppo:
            weight = weights[diversity_count]
            diversity_count += 1
            print('import from ppo weighted reward')
            from open_spiel.python.mfg.algorithms.multi_type_mfg_ppo_diversity import *

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
        horizon = env.game.get_parameters()['horizon']
        nacs = env.action_spec()['num_actions']
        nobs = env.observation_spec()['info_state'][0]


        agents = []
        actor_models = []
        ppo_policies = []
        mfg_dists = []
        for i in range(num_agent):
            if is_diversity_ppo:
                agent = Agent(nobs, len(weight), nacs).to(device)
            else:
                agent = Agent(nobs, nacs).to(device)
            actor_model = agent.actor
            critic_model = agent.critic

            fname = copy.deepcopy('actor'+filenames[ip])
            fname = fname + f'-{i}.pth' 
            actor_path = os.path.join(target_path, fname)
            actor_model.load_state_dict(torch.load(actor_path))
            actor_model.eval()

            fname = copy.deepcopy('critic'+filenames[ip])
            fname = fname + f'-{i}.pth' 
            critic_path = os.path.join(target_path, fname)
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

        inputs = [{} for _ in range(num_agent)]
        for idx in range(num_agent):
            for t in range(horizon):
                for x in range(size):
                    for y in range(size):
                        mu = [mu_dists[idx][t, y, x]]
                        x_onehot = onehot(x, size).tolist()
                        y_onehot = onehot(y, size).tolist()
                        t_onehot = onehot(t, horizon).tolist()
                        if is_diversity_ppo:
                            state = x_onehot + y_onehot
                            obs = torch.Tensor(state+mu+weight)
                        else:
                            state = x_onehot + y_onehot
                            obs = torch.Tensor(state+mu)
                        inputs[idx][f"obs-{x}-{y}-{t}-m"] = obs 

        values = np.zeros((horizon, size, size))
        for idx in range(num_agent):
            for t in range(horizon):
                for x in range(size):
                    for y in range(size):
                        obs_input = inputs[idx][f"obs-{x}-{y}-{t}-m"]
                        value = agents[idx].get_value(obs_input)
                        values[t, y, x] = value 
            
            value_filename = f'ppo-values{idx}' 
            save_path = os.path.join(target_path, f"{value_filename}.gif")
            multi_render([values], save_path, ['value'], use_kde=False)

            # output = model(input_data)
            def get_action(x, actor_model):
                logits = actor_model(x)
                probs = Categorical(logits=logits)
                action = probs.sample()
                return action
            
        info_state = []
        ep_ret = 0.0

        steps = envs[0].max_game_length
        info_state = [torch.zeros((steps,nobs), device=device) for i in range(num_agent)]
        actions = [torch.zeros((steps,), device=device) for _ in range(num_agent)]
        logprobs = [torch.zeros((steps,), device=device) for _ in range(num_agent)]
        rewards = [torch.zeros((steps,), device=device) for _ in range(num_agent)]
        dones = [torch.zeros((steps,), device=device) for _ in range(num_agent)]
        values = [torch.zeros((steps,), device=device) for _ in range(num_agent)]
        entropies = [torch.zeros((steps,), device=device) for _ in range(num_agent)]
        t_actions = [torch.zeros((steps,), device=device) for _ in range(num_agent)]
        t_logprobs = [torch.zeros((steps,), device=device) for _ in range(num_agent)]

        step = 0

        time_steps = [envs[i].reset() for i in range(num_agent)]
        while not time_steps[0].last():
            mu = []
            for i in range(num_agent):
                obs = time_steps[i].observations["info_state"][i]
                obs = torch.Tensor(obs).to(device)
                info_state[i][step] = obs
                obs_list = list(obs)
                if is_diversity_ppo:
                    obs_pth = torch.Tensor(obs_list[0:20] + [obs_list[-1]] + weight)
                else:
                    obs_pth = torch.Tensor(obs_list[0:20] + [obs_list[-1]])
                #obs_pth = torch.Tensor(obs).to(device)
                #obs = torch.Tensor(obs).to(device)
                with torch.no_grad():
                    t_action, t_logprob, _, _ = agents[i].get_action_and_value(obs_pth)
                    action, logprob, entropy, value = agents[i].get_action_and_value(obs_pth)

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
        save_path = os.path.join(target_path, f"reward.pkl")
        print(f'Saved as {save_path}')
        pkl.dump(rewards, open(save_path, 'wb'))
        print(rewards)
        for i in range(num_agent):
            reward_np = np.array(rewards[i])
            print(f'cumulative reward {i}: {np.sum(reward_np)}')
        save_path = os.path.join(target_path, f"{filename}k.mp4")
        final_dists = calc_distribution(envs, merge_dist, info_state, save=False, filename=save_path)

        final_dists = np.array(final_dists)
        save_path = os.path.join(target_path, f"{filename}.gif")
        print(np.array(final_dists).shape)
        multi_render(final_dists[:, :, :], save_path, [f'Group {i}' for i in range(num_agent)])

        #multi_render(final_dists[:, 6:20, :], save_path, ['Group1', 'Gorup2', 'Group3']) # render from 6step to 20step

        #labels = [[f"Group {n} ()" for n in range(num_agent)] for i in range(len(final_dists[0]))] 
        # save_path = os.path.join(target_path, f"test-{filename}.gif")
        # gifMaker.add_datas([[final_dists]])
        # gifMaker.make(save_path, titles)


        res_final_dists.append(final_dists)




    #save_path = os.path.join(pathes[0], f"test-diff-{filename}.gif")
    #titles = [[f"Group {n} (time={i})" for n in range(num_agent)] for i in range(len(final_dists[0]))] 
    #gifMaker.add_datas([res_final_dists])
    #gifMaker.make(save_path, titles, cmap='seismic', min_value=-1.0, max_value=1.0)

    labels = [f"Group {n}" for n in range(num_agent)] 
    diff_render_distance_plot(res_final_dists, pathes, pathnames, labels)
