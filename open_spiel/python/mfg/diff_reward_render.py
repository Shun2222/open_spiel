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
from open_spiel.python.mfg.multi_render_reward import multi_render_reward 
from open_spiel.python.mfg.games import factory
from open_spiel.python.mfg import value
from open_spiel.python.mfg.algorithms.mfg_ppo import Agent, PPOpolicy
from diff_utils import *
from gif_maker import *

plt.rcParams["font.size"] = 20
plt.rcParams["animation.ffmpeg_path"] = "/usr/bin/ffmpeg"


def create_rew_input(obs_shape, nacs, horizon, mu_dists, single, notmu, state_only=False):
    inputs = {}
    for x in range(obs_shape[1]):
        x_onehot = onehot(x, obs_shape[1]).tolist()
        for y in range(obs_shape[0]):
            for t in range(horizon):
                xy_onehot = x_onehot + onehot(y, obs_shape[0]).tolist()
                if single:
                    for i in range(len(mu_dists)):
                        xym_onehot = xy_onehot + [mu_dists[i][t, y, x]]
                        inputs[f'{x}-{y}-{t}-m-{i}'] = xym_onehot
                elif notmu:
                    inputs[f'{x}-{y}-{t}'] = xy_onehot
                else:
                    xym_onehot = xy_onehot + [mu_dists[i][t, y, x] for i in range(len(mu_dists))]
                    inputs[f'{x}-{y}-{t}-m'] = xym_onehot
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
    
    args = parser.parse_args()
    return args

filename = "actor"
pathes = [
            "/mnt/shunsuke/result/0627/multi_maze2_airl",
            "/mnt/shunsuke/result/0627/multi_maze2_1hidden_mfairl",
            "/mnt/shunsuke/result/0627/multi_maze2_2hidden_mfairl",
             "/mnt/shunsuke/result/0627/multi_maze2_airl_basicfuncs_s_mu_a",
             "/mnt/shunsuke/result/0627/multi_maze2_sa_mu",
             "/mnt/shunsuke/result/0627/multi_maze2_airl_basicfuncs_s_mua.py",
             "/mnt/shunsuke/result/0627/multi_maze2_dxy_mu_a",
             "/mnt/shunsuke/result/0627/multi_maze2_dxya_mu",
             "/mnt/shunsuke/result/0627/multi_maze2_dxy_mua",
         ] 
            #"/mnt/shunsuke/result/0627/multi_maze2_mfairl_time",
            #"/mnt/shunsuke/result/0614/multi_maze2_airl_basicfuncs",
            #"/mnt/shunsuke/result/0614/multi_maze2_airl_basicfuncs_time",
            #"/mnt/shunsuke/result/0614/185pc/multi_maze2_airl",
            #"/mnt/shunsuke/result/0614/multi_maze2_airl_basicfuncs_episode1",
            #"/mnt/shunsuke/result/0614/185pc/multi_maze2_airl_1episode",
           #"/mnt/shunsuke/result/0614/185pc/multi_maze1_airl_basicfuncs_time",
pathnames = [
                "MF-AITL",
                "MF-AITL_2hidden",
                "MF-AITL_3hidden",
                "MF-AITL_s_mu_a",
                "MF-AITL_sa_mu",
                "MF-AITL_s_mua",
                "MF-AITL_dxy_mu_a",
                "MF-AITL_dxya_mu",
                "MF-AITL_dxy_mua",
            ] 
update_infos = [
                "200_2",
                "200_2",
                "200_2",
                "200_2",
                "200_2",
                "200_2",
                "200_2",
                "200_2",
                "200_2",
                "200_2",
              ]

is_single = [False, False, False, False, False]
is_notmu = [False, False, False, False, False]
is_basicfuncs = [False, False, False]
is_basicfuncs_time = [False, False, False]
is_1hiddens = [False, True, False]


reward_filename = disc_filename = 'disc_reward'
value_filename = 'disc_value'
distance_filename = 'disc_distance'
mu_filename = 'disc_mu'
actor_filename = 'actor'
nets_dict = { 
                's_mu_a': ['state', 'mu', 'act'],
                'sa_mu': ['state_a', 'mu'],
                's_mua': ['state', 'mu_a'],
                'dxy_mu_a': ['dxy', 'mu', 'act'],
                'dxya_mu': ['dxy_a', 'mu'],
                'dxy_mua': ['dxy', 'mu_a'],
            }

if __name__ == "__main__":
    args = parse_args()

    for ip, target_path in enumerate(pathes):
        for i in range(3):
            fname = reward_filename
            fname = fname + f'{update_infos[ip]}-{i}.pth' 
            fpath = osp.join(target_path, fname)
            assert osp.isfile(fpath), f'isFileError: {fpath}'

            fname = value_filename
            fname = fname + f'{update_infos[ip]}-{i}.pth' 
            fpath = osp.join(target_path, fname)
            assert osp.isfile(fpath), f'isFileError: {fpath}'

            fname = actor_filename
            fname = fname + f'{update_infos[ip]}-{i}.pth' 
            fpath = osp.join(target_path, fname)
            assert osp.isfile(fpath), f'isFileError: {fpath}'

            for key in nets_dict.keys():
                if key in pathnames[ip]:
                    for label in nets_dict[key]:
                        fname = f'disc_{label}'
                        fname = fname + f'{update_infos[ip]}-{i}.pth' 
                        fpath = osp.join(target_path, fname)
                        print(f'checked {fpath}')
                        assert osp.isfile(fpath), f'isFileError: {fpath}'
    print(f'Checked path: OK')


    res = []
    outputs = []
    for p in range(len(pathes)):
        single = is_single[p]
        notmu = is_notmu[p]
        is_1hidden = is_1hiddens[p]

        is_nets = False
        net_input = None
        for key in nets_dict.keys():
            if key in pathnames[p]:
                is_nets = True
                net_input = key


        if is_1hidden:
            from open_spiel.python.mfg.algorithms.discriminator_1hidden import Discriminator
        elif is_nets:
            from open_spiel.python.mfg.algorithms.discriminator import Discriminator
        else:
            from open_spiel.python.mfg.algorithms.discriminator import Discriminator

        # Set the seed 
        seed = args.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        print(f"Random seed set as {seed}")

        update_info = update_eps_info = f'{update_infos[p]}'
        device = torch.device("cpu")
        #distrib_path = os.path.join(pathes[p], distrib_filename+update_eps_info)
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
        horizon = env.game.get_parameters()['horizon']

        nmu = num_agent
        size = env.game.get_parameters()['size']
        state_size = nobs -1 - horizon # nobs-1: obs size (exposed own mu), nmu: all agent mu size, horizon: horizon size
        obs_xym_size = nobs -1 - horizon + nmu # nobs-1: obs size (exposed own mu), nmu: all agent mu size, horizon: horizon size

        agents = []
        actor_models = []
        ppo_policies = []
        mfg_dists = []
        discriminators = []
        for i in range(num_agent):
            agent = Agent(nobs, nacs).to(device)
            actor_model = agent.actor

            fname = copy.deepcopy(actor_filename+update_eps_info)
            fname = fname + f'-{i}.pth' 
            actor_path = osp.join(pathes[p], fname)
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
            elif is_nets:
                if net_input=='s_mu_a':
                    inputs = [state_size, nmu, nacs]
                    labels = ['state', 'mu', 'act']
                elif net_input=='sa_mu':
                    inputs = [state_size+nacs, nmu]
                    labels = ['state_a', 'mu']
                elif net_input=='s_mua':
                    inputs = [state_size, nmu+nacs]
                    labels = ['state', 'mu_a']
                elif net_input=='dxy_mu_a':
                    inputs = [2, nmu, nacs]
                    labels = ['dxy', 'mu', 'act']
                elif net_input=='dxya_mu':
                    inputs = [2+nacs, nmu]
                    labels = ['dxy_a', 'mu']
                elif net_input=='dxy_mua':
                    inputs = [2, nmu+nacs]
                    labels = ['dxy', 'mu_a']
                else:
                    assert False, f'not matched disc type: {net_input}'
                discriminator = Discriminator(inputs, obs_xym_size, labels, device)
            else:
                discriminator = Discriminator(nobs+num_agent-horizon-1, nacs, False, device)
            reward_path = osp.join(pathes[p], reward_filename+update_eps_info + f'-{i}.pth')
            value_path = osp.join(pathes[p], value_filename+update_eps_info + f'-{i}.pth')

            if is_nets:
                discriminator.load(pathes[p], f'{update_eps_info}', use_eval=True)
                discriminator.print_weights()
            discriminators.append(discriminator)

        merge_dist = distribution.MergeDistribution(game, mfg_dists)
        for env in envs:
          env.update_mfg_distribution(merge_dist)

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


        if is_nets:
            inputs = discriminators[0]([size, size], nacs, horizon, mu_dists)
        else:
            inputs = create_rew_input([size, size], nacs, horizon, mu_dists, single, notmu, state_only=False)
        save_path = os.path.join(pathes[p], filename+str(update_info))
        datas = []
        outs = []
        if is_nets:
            n_nets = discriminators[0].get_num_nets()
            outs = [[] for _ in range(n_nets)]
        for i in range(num_agent):
            if is_nets:
                rewards, output = multi_render_reward(size, nacs, horizon, inputs, discriminators[i], i, single, notmu, is_nets=is_nets, net_input=net_input, save=True, filename=save_path+f"-{i}")
                for j in range(n_nets):
                    outs[j].append(np.mean(output[j], axis=3))
            else:
                rewards = multi_render_reward(size, nacs, horizon, inputs, discriminators[i], i, single, notmu, save=True, filename=save_path+f"-{i}")
            datas.append(np.mean(rewards, axis=3))

        res.append(datas)
        outputs.append(outs)
        path = osp.join(save_path + f'-mean.gif')
        labels = [f'Group {i}' for i in range(num_agent)]
        print(np.array(datas).shape)
        multi_render(datas, path, labels)
        if is_nets:
            labels = [f'Group {i}' for i in range(num_agent)]
            net_labels = discriminators[0].get_net_labels()
            for i in range(n_nets):
                path = osp.join(save_path + f'-mean-{net_labels[i]}.gif')
                multi_render(outputs[i], path, labels)

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
            save_path = os.path.join(pathes[p], filename+f'-mutime-box-{i}.png')
            plt.savefig(save_path)
            plt.close()
            print(f'saved {save_path} ')
            if is_nets:
                for j in range(n_nets):
                    fig = plt.figure(figsize=(16, 12))
                    ax = fig.add_subplot(1, 1, 1)
                    points = outputs[j][i]
                    col = 1
                    for s in range(len(points[0].shape)):
                        col *= points[0].shape[s]
                    points = points.reshape(len(points), col).T
                    bp = ax.boxplot(points)
                    plt.xlabel(r"$\mu_{time}$")
                    plt.ylabel(fr"{net_labels[j]} value")
                    save_path = os.path.join(pathes[p], filename+f'-mutime-box-{net_labels[j]}-{i}.png')
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
                save_path = os.path.join(pathes[p], filename+f'-box-{j}-{i}.png')
                plt.savefig(save_path)
                plt.close()
                print(f'saved {save_path} ')
                if is_nets:
                    for k in range(n_nets):
                        fig = plt.figure(figsize=figsizes[j])
                        ax = fig.add_subplot(1, 1, 1)
                        points = outputs[k][i]
                        col = 1
                        for s in range(len(points[0].shape)):
                            col *= points[0].shape[s]
                        points = points.reshape(len(points), col)
                        bp = ax.boxplot(points)
                        plt.xlabel(r"State")
                        plt.ylabel(fr"{net_labels[k]} value")
                        save_path = os.path.join(pathes[p], filename+f'-box-{net_labels[k]}-{i}.png')
                        plt.savefig(save_path)
                        plt.close()
                        print(f'saved {save_path} ')

    labels = [f"Group {n}" for n in range(num_agent)] 
    diff_render_distance_plot(np.array(res), pathes, pathnames, labels)
    if np.sum(is_basicfuncs)==len(is_basicfuncs):
        for j in range(n_nets):
            output_pathnames = [f'{net_labels[j]}-'+p for p in pathnames] 
            diff_render_distance_plot(np.array(outputs[j]), pathes, output_pathnames, labels)


