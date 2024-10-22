import os
import os.path as osp
import click
import time
import numpy as np
import pickle as pkl

import argparse
import torch
import pyspiel
import distribution

import pickle as pkl
from scipy.spatial import distance
from open_spiel.python.mfg.games.predator_prey import calc_reward 
import matplotlib.pyplot as plt

from gif_maker import *

path = r"/mnt/shunsuke/result/0726/multi_maze2_expert"

expert_mu_pkl = rf"expert-conv_dists.pkl"
expert_mu_pkl = osp.join(path, expert_mu_pkl)


sampled_expert_pkl = [rf"expert-1000tra-{i}.pkl" for i in range(3)]
sampled_expert_pkl = [osp.join(path, sampled_expert_pkl[i]) for i in range(3)]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_trajs", type=int, default=1000, help="set a random seed")
    args = parser.parse_args()
    return args

def load_pkl(path_pkl):
    return pkl.load(open(path_pkl, 'rb'))

def state_visition_flequency(trajs, horizon=40, size=10, num_trajs=1000):
    state_visitation_count = np.zeros((horizon, size, size))
    num_trajs = num_trajs if len(trajs) > num_trajs else len(trajs)
    for i in range(num_trajs):
        obs_mu = trajs[i]["ob"]
        for t in range(len(obs_mu)):
            obs_x = np.argmax(obs_mu[t][:size])
            obs_y = np.argmax(obs_mu[t][size:size*2])
            state_visitation_count[t][obs_y][obs_x] += 1
    svf = state_visitation_count/num_trajs
    return svf 

def get_expected_return(mus, trajs, size=10, idx=0, num_trajs=1000):
    num_trajs = num_trajs if len(trajs) > num_trajs else len(trajs)
    expected_return = [] 
    for i in range(num_trajs):
        obs_mu = trajs[i]["ob"]
        rews = [] 
        for t in range(len(obs_mu)):
            x = np.argmax(obs_mu[t][:size])
            y = np.argmax(obs_mu[t][size:size*2])
            pos = [x, y]
            densities = np.array([
                mus[i][t][y][x] for i in range(len(mus))
            ], dtype=np.float64)
            rew = calc_reward(pos, densities)[idx]
            rews.append(rew)
            #print(f"{rew}, {densities}, {pos}")
            #input()
        expected_return.append(np.sum(rews))
        r = trajs[i]["ep_ret"]
    return np.mean(expected_return)

def cos_sim(a, b):
    return 1-distance.cosine(a, b)

def eval_sampled_expert(num_trajs):
    expert_mu = load_pkl(expert_mu_pkl)
    sampled_expert = [load_pkl(sampled_expert_pkl[i]) for i in range(3)]

    num_agent = len(expert_mu)
    sampled_svf = [state_visition_flequency(sampled_expert[i], num_trajs=num_trajs) for i in range(num_agent)]
    similarities = []
    returns = []
    for i in range(len(sampled_expert)):
        print(f"--------------------Agent{i}--------------------")


        similarity = [cos_sim(expert_mu[i][t].flatten(), sampled_svf[i][t].flatten()) for t in range(len(sampled_svf[i]))]
        similarity = np.mean(similarity)
        similarities.append(similarity)
        print(f"cos_sim={np.round(similarity, 2)} where, num_trajs={num_trajs}")

        expert_mu_returns = get_expected_return(expert_mu, sampled_expert[i], idx=i, num_trajs=num_trajs)
        sampled_svf_returns = get_expected_return(sampled_svf, sampled_expert[i], idx=i, num_trajs=num_trajs)
        returns.append([expert_mu_returns, sampled_svf_returns])
        print(f"Expected returns={expert_mu_returns} under expert mean-field")
        print(f"Expected returns={sampled_svf_returns} under svf calced by sampled trajs")
    print(f"----------------------------------------------")

    return similarities, returns

def analysis_sampled_expert(num_trajs_list):
    all_similarities = [[] for _ in range(3)]
    all_returns = [[] for _ in range(3)]
    for num_trajs in num_trajs_list:
        sim, ret = eval_sampled_expert(num_trajs)
        for i in range(3):
            all_similarities[i].append(sim[i])
            all_returns[i].append(ret[i])
    for i in range(3):
        similarities = np.array(all_similarities[i])
        returns = np.array(all_returns[i])
        plt.figure()
        plt.plot(num_trajs_list, similarities)
        plt.xlabel("Num of sampled trajectories")
        plt.ylabel("Similarity")
        savepath = osp.join(path, f"similarity-num_traj-{i}.png")
        plt.savefig(savepath)
        print(f"saved in {savepath}")

        plt.figure()
        plt.plot(num_trajs_list, returns.T[0], label="Expert")
        plt.plot(num_trajs_list, returns.T[1], label="Sampled Expert")
        plt.xlabel("Num of sampled trajectories")
        plt.ylabel("Expected return")
        savepath = osp.join(path, f"ret_sampled_expert-num_traj-{i}.png")
        plt.legend()
        plt.savefig(savepath)
        print(f"saved in {savepath}")
    return
    
def save_svf(num_trajs):
    expert_mu = load_pkl(expert_mu_pkl)
    sampled_expert = [load_pkl(sampled_expert_pkl[i]) for i in range(3)]

    num_agent = len(expert_mu)
    sampled_svf = [state_visition_flequency(sampled_expert[i], num_trajs=num_trajs) for i in range(num_agent)]

    plt.figure()
    n_datas = len(sampled_svf)
    fig, axes = plt.subplots(1, n_datas, figsize = (4*n_datas, 4))
    for i in range(len(sampled_svf)):
        axes[i].axis('off')
        svf = np.mean(sampled_svf[i], axis=0)
        axes[i].imshow(svf)
    savepath = osp.join(path, f"svf-{num_trajs}trajs.png")
    plt.savefig(savepath)
    print(f"Saved as {savepath}")

    savepath = osp.join(path, f"svf-{num_trajs}trajs.gif")
    multi_render(sampled_svf, savepath, ["" for _ in range(len(sampled_svf))], use_kde=False)

def save_dataset(num_trajs):
    from dataset import MFGDataSet
    mfgds = MFGDataSet(sampled_expert_pkl[0], traj_limitation=num_trajs)
    trajs = mfgds.get_trajs()
    sampled_svf = [state_visition_flequency(trajs, num_trajs=num_trajs)]

    plt.figure()
    n_datas = len(sampled_svf)
    fig, ax = plt.subplots(1, 1, figsize = (4, 4))
    ax.axis('off')
    svf = np.mean(sampled_svf[0], axis=0)
    ax.imshow(svf)
    savepath = osp.join(path, f"dataset-svf-{num_trajs}trajs.png")
    plt.savefig(savepath)
    print(f"Saved as {savepath}")

    savepath = osp.join(path, f"dataset-svf-{num_trajs}trajs.gif")
    multi_render(sampled_svf, savepath, ["" for _ in range(len(sampled_svf))], use_kde=False)



if __name__ == '__main__':
    args = parse_args()
    # maximum exp ret under sampled x trajs
    #eval_sampled_expert(args.num_trajs)
    
    # analysis relationship max exp ret and num trajs 
    #num_trajs_list = np.arange(0, 205, 5)
    #num_trajs_list[0] = 1
    #analysis_sampled_expert(num_trajs_list)

    # save svf
    num_trajs_list = [1, 15, 50, 100]
    for num_trajs in num_trajs_list:
        save_svf(num_trajs)

    # save dataset 
    #save_dataset(1)
