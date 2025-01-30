import os
import re
import argparse  # コマンドライン引数を処理するために使用
import numpy as np
from utils import *
from datetime import datetime  # 最終更新日時をフォーマットするために使用
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
#from open_spiel.python.mfg.algorithms.multi_type_mfg_ppo import *
from open_spiel.python.mfg.algorithms.multi_type_mfg_ppo_discrew import *
from open_spiel.python.mfg.multi_render_reward import * 
from open_spiel.python.mfg.games import factory
from games.predator_prey import *
from open_spiel.python.mfg import value
from diff_utils import *
from gif_maker import *
from open_spiel.python.mfg.algorithms.discriminator_networks_divided_value import * 

plt.rcParams["font.size"] = 10
plt.rcParams["animation.ffmpeg_path"] = "/usr/bin/ffmpeg"

num_agent = 3
use_horizon = False
device = torch.device("cpu")
filename = "actor"
reward_filename = disc_filename = 'disc_reward'
value_filename = 'disc_value'
distance_filename = 'disc_distance'
mu_filename = 'disc_mu'
actor_filename = 'actor'

_MODE = "4rooms"
if _MODE=="Predator_Prey":
    _DEFAULT_REWARD_MATRIX = np.array([[0, 100, 100], [-100, 0, 100], [-100, -100, 0]])
    _DEFAULT_FORBIDDEN_POSITION = np.array([])
    _DEFAULT_GOAL_POSITION = np.array([[5, 4], [4, 5], [5, 5]]) 
elif _MODE=="4rooms":
    _DEFAULT_REWARD_MATRIX = np.array([[0, -50, -50], [-50, 0, -50], [-50, -50, 0]])
    _DEFAULT_FORBIDDEN_POSITION = [[5, i] for i in [1, 3, 4, 5, 6, 8]]
    _DEFAULT_FORBIDDEN_POSITION += [[i, 4] for i in [0, 2, 4]]
    _DEFAULT_FORBIDDEN_POSITION += [[i, 5] for i in [6, 8]]
    _DEFAULT_FORBIDDEN_POSITION = np.array(_DEFAULT_FORBIDDEN_POSITION) 

    _DEFAULT_GOAL_POSITION = np.array([[8, 8], [1, 8], [8, 1]])
else:
    _DEFAULT_REWARD_MATRIX = np.array([[0, -50, -50], [-50, 0, -50], [-50, -50, 0]])
    _DEFAULT_FORBIDDEN_POSITION = np.array([[2, 4], [2, 5], [4, 2], [4, 7], [5, 2], [5, 7], [7, 4], [7, 5]])
    _DEFAULT_GOAL_POSITION = np.array([[5, 4], [4, 5], [5, 5]])

def render_sequence(datas, axshape, save_path, axtitles):
    fig, axes = plt.subplots(axshape[0], axshape[1], figsize=(axshape[1]*4+4, axshape[0]*4))
    for i in range(axshape[0]):
        for j in range(axshape[1]):
            axes[i][j].axis('off')

    for i in range(len(datas)):
        j = i%axshape[1]
        k = i//axshape[1]
        axes[k][j].imshow(datas[i])
        axes[k][j].set_title(axtitles[i])
    plt.savefig(save_path)


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
                    r, r_xy, r_mu = get_true_reward(pos, mu)
                    rew[idx][t, y, x] = r[idx]
                    rew_xy[idx][t, y, x] = r_xy[idx]
                    rew_mu[idx][t, y, x] = r_mu[idx]
    return rew, rew_xy, rew_mu 

def get_true_reward(pos, densities):
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
        r_xy = np.array([0, 0, 0])
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

def actor_render(game, envs, pathes, pathnames, update_infos):
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
        connected_data = []
        connected_label = []

        update_info = update_eps_info = f'{update_infos[p]}'
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
        critic_models = []
        ppo_policies = []
        mfg_dists = []
        for i in range(num_agent):
            agent = Agent(nobs, nacs).to(device)
            actor_model = agent.actor
            critic_model = agent.critic

            fname = copy.deepcopy(actor_filename+update_eps_info)
            fname = fname + f'-{i}.pth' 
            actor_path = osp.join(pathes[p], fname)
            actor_model.load_state_dict(torch.load(actor_path))
            actor_model.eval()
            print("load actor model from", actor_path)

            fname = copy.deepcopy('critic'+update_eps_info)
            fname = fname + f'-{i}.pth' 
            critic_path = osp.join(pathes[p], fname)
            critic_model.load_state_dict(torch.load(critic_path))
            critic_model.eval()
            print("load critic model from", critic_path)

            agents.append(agent)
            actor_models.append(actor_model)
            critic_models.append(critic_model)

            ppo_policies.append(PPOpolicy(game, agent, None, device))
            mfg_dist = distribution.DistributionPolicy(game, ppo_policies[-1])
            mfg_dists.append(mfg_dist)


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

        mu_dists = np.array(mu_dists)
        save_path = os.path.join(pathes[p], f"actor.gif")
        multi_render(mu_dists[:, :, :], save_path, [f'Group {i}' for i in range(num_agent)])

        fig = plt.figure()
        for i in range(num_agent):
            ax = fig.add_subplot(1, num_agent, i+1)
            ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
            ax.imshow(np.mean(mu_dists[i], axis=0))
            ax.set_title(f'Group{i}')
        save_path = os.path.join(pathes[p], f"mu_dists.png")
        plt.savefig(save_path)
        print(f"Saved in {save_path}")

        #connected_data.append(mu_dists[:, :, :])
        #connected_label.append([f'MF Group {i}' for i in range(num_agent)])
        #path = osp.join(pathes[p], f'connected_result.gif')
        #multi_render_set_pos(connected_data, connected_label, path)


def render(game, envs, pathes, pathnames, update_infos):
    print(f'len pathes = {len(pathes)}')
    print(f'len pathnames = {len(pathnames)}')
    for ip, target_path in enumerate(pathes):
        for i in range(3):
            fname = reward_filename
            fname = fname + f'{update_infos[ip]}-{i}.pth' 
            fpath = osp.join(target_path, fname)
            assert osp.isfile(fpath), f'isFileError: {fpath}'


            fname = actor_filename
            fname = fname + f'{update_infos[ip]}-{i}.pth' 
            fpath = osp.join(target_path, fname)
            assert osp.isfile(fpath), f'isFileError: {fpath}'

            net_input = get_net_input(pathnames[ip])
            if net_input:
                net_labels = get_net_labels(net_input)
                if is_divided_value(pathnames[ip]):
                    for label in net_labels:
                        fname = f'disc_{label}'
                        fname = fname + f'{update_infos[ip]}-{i}.pth' 
                        fpath = osp.join(target_path, fname)
                        print(f'checked {fpath}')
                        assert osp.isfile(fpath), f'isFileError: {fpath}'

                        fname = value_filename
                        fname = fname + f"_{label}" + f'{update_infos[ip]}-{i}.pth' 
                        fpath = osp.join(target_path, fname)
                        assert osp.isfile(fpath), f'isFileError: {fpath}'
                else:
                    fname = value_filename
                    fname = fname + f'{update_infos[ip]}-{i}.pth' 
                    fpath = osp.join(target_path, fname)
                    assert osp.isfile(fpath), f'isFileError: {fpath}'
                    for label in net_labels:
                        fname = f'disc_{label}'
                        fname = fname + f'{update_infos[ip]}-{i}.pth' 
                        fpath = osp.join(target_path, fname)
                        print(f'checked {fpath}')
                        assert osp.isfile(fpath), f'isFileError: {fpath}'
    print(f'Checked path: OK')


    res = []
    outputs = []
    for p in range(len(pathes)):
        connected_data = []
        connected_label = []
        is_nets = is_networks(pathnames[p]) 
        if is_nets:
            net_input = get_net_input(pathnames[p])
            net_labels = get_net_labels(net_input)
            is_divided = is_divided_value(pathnames[p])
            from open_spiel.python.mfg.algorithms.discriminator_networks_divided_value import Discriminator, Discriminator_2nets, Discriminator_3nets
            if not is_divided:
                assert False, "is_net is true but, is_divided is false"
        else:
            from open_spiel.python.mfg.algorithms.discriminator import Discriminator

        update_info = update_eps_info = f'{update_infos[p]}'

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
        critic_models = []
        ppo_policies = []
        mfg_dists = []
        discriminators = []
        for i in range(num_agent):
            agent = Agent(nobs, nacs).to(device)
            actor_model = agent.actor
            critic_model = agent.critic

            fname = copy.deepcopy(actor_filename+update_eps_info)
            fname = fname + f'-{i}.pth' 
            actor_path = osp.join(pathes[p], fname)
            actor_model.load_state_dict(torch.load(actor_path))
            actor_model.eval()
            print("load actor model from", actor_path)

            fname = copy.deepcopy('critic'+update_eps_info)
            fname = fname + f'-{i}.pth' 
            critic_path = osp.join(pathes[p], fname)
            critic_model.load_state_dict(torch.load(critic_path))
            critic_model.eval()
            print("load critic model from", critic_path)

            agents.append(agent)
            actor_models.append(actor_model)
            critic_models.append(critic_model)

            ppo_policies.append(PPOpolicy(game, agent, None, device))
            mfg_dist = distribution.DistributionPolicy(game, ppo_policies[-1])
            mfg_dists.append(mfg_dist)

            if is_nets:
                inputs = get_input_shape(net_input, env, num_agent)
                labels = get_net_labels(net_input)
                num_hidden = get_num_hidden(pathnames[p])
                if len(labels)==1:
                    discriminator = Discriminator(inputs, obs_xym_size, labels, device, num_hidden=num_hidden)
                if len(labels)==2:
                    discriminator = Discriminator_2nets(inputs, obs_xym_size, labels, device, num_hidden=num_hidden)
                if len(labels)==3:
                    discriminator = Discriminator_3nets(inputs, obs_xym_size, labels, device, kum_hidden=num_hidden)
            else:
                discriminator = Discriminator(3, nacs, True, device)
            reward_path = osp.join(pathes[p], reward_filename+update_eps_info + f'-{i}.pth')
            value_path = osp.join(pathes[p], value_filename+update_eps_info + f'-{i}.pth')

            if is_nets:
                discriminator.load(pathes[p], f'{update_eps_info}-{i}', use_eval=True)
                save_path = os.path.join(pathes[p], filename+str(update_info)+f'weights-{i}.png')
                discriminator.savefig_weights(save_path)
                discriminator.print_weights()
            else:
                discriminator.load(reward_path, value_path, use_eval=True)
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

        mu_dists = np.array(mu_dists)
        #save_path = os.path.join(target_path, f"actor.gif")
        #print(np.array(mu_dists).shape)
        #multi_render(mu_dists[:, :, :], save_path, [f'Group {i}' for i in range(num_agent)])
        connected_data.append(mu_dists[:, :, :])
        connected_label.append([f'MF Group {i}' for i in range(num_agent)])


        if is_nets:
            inputs = discriminators[0].create_inputs([size, size], nacs, horizon, mu_dists)
        else:
            inputs = create_rew_input([size, size], nacs, horizon, mu_dists, False, False, state_only=False)

        save_path = os.path.join(pathes[p], filename+str(update_info))

        true_reward, true_reward_xy, true_reward_mu = calc_true_reward([size, size], horizon, mu_dists)

        path = osp.join(save_path + f'-true_reward.gif')
        #labels = [f'Group {i}' for i in range(num_agent)]
        #multi_render(true_reward, path, labels, use_kde=False)
        #path = osp.join(save_path + f'-true_reward_xy.gif')
        #multi_render(true_reward_xy, path, labels, use_kde=False)
        #path = osp.join(save_path + f'-true_reward_mu.gif')
        #multi_render(true_reward_mu, path, labels, use_kde=False)
        connected_data.append(true_reward)
        connected_label.append([f'True Reward' for i in range(num_agent)])
        connected_data.append(true_reward_xy)
        connected_label.append([f'True Reward (xy)' for i in range(num_agent)])
        connected_data.append(true_reward_mu)
        connected_label.append([f'True Reward (mf)' for i in range(num_agent)])

        datas = []
        outs = []
        if is_nets:
            n_nets = discriminators[0].get_num_nets()
            outs = [[] for _ in range(n_nets)]
        for i in range(num_agent):
            if is_nets:
                if is_divided:
                    use_rate = False
                    if use_rate:
                        rewards, output = multi_render_weighted_reward_nets_divided_value(size, nacs, horizon, inputs[i], discriminators[i], rates[p], save=True, filename=save_path+f"-{i}")
                    else:
                        rewards, output = multi_render_reward_nets_divided_value(size, nacs, horizon, inputs[i], discriminators[i], save=True, filename=save_path+f"-{i}", mode=_MODE)
                else:
                    rewards, output = multi_render_reward_nets(size, nacs, horizon, inputs[i], discriminators[i], save=True, filename=save_path+f"-{i}")
                for j in range(n_nets):
                    outs[j].append(np.mean(output[j], axis=3))
            else:
                rewards = multi_render_reward(mu_dists, size, nacs, horizon, inputs[i], discriminators[i], i, False, False, False, False, dxyinput=True, save=True, filename=save_path+f"-{i}", mode=_MODE)
            datas.append(np.mean(rewards, axis=3))

        res.append(datas)
        outputs.append(outs)
        #path = osp.join(save_path + f'-mean.gif')
        #labels = [f'Group {i}' for i in range(num_agent)]
        #print(np.array(datas).shape)
        #multi_render(datas, path, labels, use_kde=False)
        connected_data.append(datas)
        connected_label.append([f'Est Reward' for i in range(num_agent)])
        if is_nets:
            labels = [f'Group {i}' for i in range(num_agent)]
            net_labels = get_net_labels(net_input)
            for i in range(n_nets):
                #path = osp.join(save_path + f'-mean-{net_labels[i]}.gif')
                output = np.array(outs[i])
                #print(output.shape)
                #multi_render(output, path, labels, use_kde=False)
                connected_data.append(output)
                connected_label.append([f'Est Reward ({net_labels[i]})' for j in range(num_agent)])
        path = osp.join(pathes[p], f'connected_result.gif')
        multi_render_set_pos(connected_data, connected_label, path)

        cds = np.array([])
        each = 5
        for cd in connected_data:
            if len(cds)==0:
                cdsi = np.concatenate([cd[::each], np.array([cd[-1]])])
                cds = cdsi
                cds_label = [f"t={i}" for i in range(0, len(cds)-1, each)]
                cds_label.append(f"t={len(cd)}")
            else:
                cdsi = np.concatenate([cd[::each], np.array([cd[-1]])])
                cds = np.concatenate([cds, cdsi])
                cds_label += [f"t={i}" for i in range(0, len(cds)-1, each)]
                cds_label.append(f"t={len(cd)}")
        
        axshape = [0, 9]
        axshape[0] = len(cds)//axshape[1] + 1
        path = osp.join(pathes[p], f'connected_sequence_result.png')
        render_sequence(cds, axshape, path, cds_label)


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
                    points = np.array(outs[j][i])
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
                        points = np.array(outs[k][i])
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
     

    #labels = [f"Group {n}" for n in range(num_agent)] 
    #if all_nets:
    #    for j in range(n_nets):
    #        output_pathnames = [f'{net_labels[j]}-'+p for p in pathnames] 
    #        diff_render_distance_plot(np.array(outputs[j]), pathes, output_pathnames, labels)




def find_max_number_in_filenames(base_dir, keywords, min_number, actor_only=False, exist_skip=False):
    # ディレクトリの確認
    if not os.path.isdir(base_dir):
        print(f"Error: {base_dir} is not a valid directory.")
        return

    # 各フォルダーの結果を格納
    results = []
    pathes = []
    filenames = []
    update_infos = []

    # 再帰的にディレクトリを探索
    for root, _, files in os.walk(base_dir):
        max_value = None
        max_file = None
        max_logname = None
        exist_keyword = True
        for keyword in keywords: 
            if not keyword in root:  # 特定の条件でフォルダをスキップ
                exist_keyword = False
                continue
        if not exist_keyword:
            continue
        
        fname = f'mu_dists.png' 
        fpath = osp.join(root, fname)
        is_exist = osp.isfile(fpath)
        if is_exist:
            print(f"Exist {fpath}")
            if exist_skip:
                print(f"Skip {root}")
                continue

        # 各ファイル名を処理
        for file in files:
            # ファイル名から数字を抽出
            numbers = map(int, re.findall(r'\d+', file))
            logname = re.findall(r'\d+_\d+', file)
            file_max = max(numbers, default=None)

            # 最大値を更新
            if file_max is not None and (max_value is None or file_max > max_value) and len(logname)>0:
                is_exist = True
                for i in range(3):

                    fname = actor_filename
                    fname = fname + f'{logname[0]}-{i}.pth' 
                    fpath = osp.join(root, fname)
                    is_exist = osp.isfile(fpath)
                    if not is_exist:
                        break

                    if not actor_only:
                        fname = reward_filename
                        fname = fname + f'{logname[0]}-{i}.pth' 
                        fpath = osp.join(root, fname)
                        is_exist = osp.isfile(fpath)
                        if not is_exist:
                            break

                        net_input = get_net_input(root.split("/")[-2], print_info=False)
                        if net_input:
                            net_labels = get_net_labels(net_input)
                            if is_divided_value(root.split("/")[-2]):
                                for label in net_labels:
                                    fname = f'disc_{label}'
                                    fname = fname + f'{logname[0]}-{i}.pth' 
                                    fpath = osp.join(root, fname)
                                    is_exist = osp.isfile(fpath)
                                    if not is_exist:
                                        break

                                    fname = value_filename
                                    fname = fname + f"_{label}" + f'{logname[0]}-{i}.pth' 
                                    fpath = osp.join(root, fname)
                                    is_exist = osp.isfile(fpath)
                                    if not is_exist:
                                        break
                            else:
                                fname = value_filename
                                fname = fname + f'{logname[0]}-{i}.pth' 
                                fpath = osp.join(root, fname)
                                is_exist = osp.isfile(fpath)
                                if not is_exist:
                                    break
                                for label in net_labels:
                                    fname = f'disc_{label}'
                                    fname = fname + f'{logname[0]}-{i}.pth' 
                                    fpath = osp.join(root, fname)
                                    is_exist = osp.isfile(fpath)
                                    if not is_exist:
                                        break
                if not is_exist:
                    print(f'Checked path: NG')
                    continue

                print(f'Checked path: OK')
                max_value = file_max
                max_file = os.path.join(root, file)
                max_logname = logname[0]



        # 最大値を記録
        if max_file is not None:
            if max_value<min_number:
                continue
            # ファイルの最終更新日時を取得
            last_modified_timestamp = os.path.getmtime(max_file)
            last_modified_time = datetime.fromtimestamp(last_modified_timestamp).strftime('%Y-%m-%d %H:%M:%S')


            results.append((root, max_file, max_value, last_modified_time, max_logname))
            pathes.append(root)
            if "seed" in root:
                filenames.append(root.split("/")[-2])
            else:
                filenames.append(root.split("/")[-1])
            update_infos.append(max_logname)

            print(f"----------------------------------------------------------------")
            print(f"  Folder: {root}({filenames[-1]})")
            #print(f"  File with max number in name: {max_file}")
            print(f"  Max number: {max_value}")
            print(f"  Last modified: {last_modified_time}")

    # Create the game instance 
    game, envs = create_env(seed)

    env = envs[0]
    horizon = env.game.get_parameters()['horizon']
    nacs = env.action_spec()['num_actions']
    nobs = env.observation_spec()['info_state'][0]
    
    if actor_only:
        actor_render(game, envs, pathes, filenames, update_infos)
    else:
        render(game, envs, pathes, filenames, update_infos)
        actor_render(game, envs, pathes, filenames, update_infos)
    return results

if __name__ == "__main__":
    # コマンドライン引数を処理
    parser = argparse.ArgumentParser(description="Find the file with the largest number in its name.")
    parser.add_argument(
        "-d", "--directory", 
        type=str, 
        default="./",
        help="The target directory to search."
    )
    parser.add_argument(
        "-s", "--seed", 
        type=int, 
        default=0,
    )
    parser.add_argument(
        "-a","--actor_only", 
        action='store_true'
    )
    parser.add_argument(
        "-e", "--exist_skip", 
        action='store_true'
    )
    parser.add_argument(
        "-k", "--keyword", 
        nargs='+', 
        type=str, 
        default=["seed-42"])
    parser.add_argument(
        "-m", "--min_number", 
        type=int, 
        default=0)
    args = parser.parse_args()

    # Set the seed 
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

    # ターゲットディレクトリの取得
    target_directory = args.directory
    keyword = args.keyword
    print(f"Keyword: \"{keyword}\"")
    results = find_max_number_in_filenames(target_directory, keyword, args.min_number, args.actor_only, args.exist_skip)
