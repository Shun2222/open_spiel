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


path = r"/mnt/shunsuke/result/0726/multi_maze2_expert"

expert_mu_pkl = r"expert-{num_trajs}traj.pkl"
expert_mu_pkl = osp.join(path, expert_mu_pkl)


sampled_expert_pkl = r"expert-conv_dists.pkl"
sampled_expert_pkl = osp.join(path, sampled_expert_pkl)

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
        obs_mu = trajs[i]["all_ob"]
        for t in len(obs_mu):
            obs_x = obs_mu[t][:size]
            obs_y = obs_mu[t][size:size*2]
            state_visitation_count[t][obs_y][obs_x] += 1
    svf = state_visitation_count/num_trajs
    return svf 

def get_expected_return(trajs, mus, size=10):
    num_trajs = len(trajs)
    expected_return = [] 
    for i in range(num_trajs):
        obs_mu = trajs[i]["all_ob"]
        rews = [] 
        for t in len(obs_mu):
            x = np.argmax(obs_mu[t][:size])
            y = np.argmax(obs_mu[t][size:size*2])
            pos = [x, y]
            densities = np.array([
                mus[i][t][y][x] for i in range(len(mus))
            ], dtype=np.float64)
            rew = calc_reward(pos, densities)
            rews.appedn(rew)
        expected_return.append(np.mean(rews))
    return np.mean(expected_return)

def cos_sim(a, b):
    return 1-distance.cosine(a, b)


if __name__ == '__main__':
    args = parse_args()

    expert_mu = load_pkl(expert_mu_pkl)
    sampled_expert = load_pkl(sampled_expert_pkl)
    sampled_svf = []
    for i in range(len(sampled_expert)):
        print(f"--------------------Agent{i}--------------------")

        svf = state_visition_flequency(sampled_expert[i], num_trajs=args.num_trajs)
        sampled_svf.append(svf)

        similarity = [cos_sim(expert_mu[i][t], svf[t]) for t in range(len(svf))]
        similarity = np.mean(similarity)
        print(f"cos_sim={np.round(similarity, 2)} where, num_trajs={args.num_trajs}")

        expert_mu_returns = get_expected_return(expert_mu, sampled_expert[i])
        sampled_svf_returns = get_expected_return(svf, sampled_expert[i])
        print(f"Expected returns={expert_mu_returns} under expert mean-field")
        print(f"Expected returns={sampled_svf_returns} under svf calced by sampled trajs")
    print(f"----------------------------------------------")

