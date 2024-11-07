import os
import os.path as osp
import pyspiel
from utils import onehot, multionehot
# 
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=6 Mainly controlles the number of spawned threateds 
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=6

import argparse
from tqdm import tqdm
from distutils.util import strtobool
import time
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import copy
from matplotlib import animation

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import logger
from open_spiel.python.mfg import utils
from open_spiel.python import rl_environment
from open_spiel.python import policy as policy_std
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms.nash_conv import NashConv
from open_spiel.python.mfg.algorithms import policy_value
from open_spiel.python.mfg.games import factory
from open_spiel.python.mfg import value
from open_spiel.python.mfg.algorithms import best_response_value
from open_spiel.python.mfg.algorithms.multi_type_mfg_ppo_discrew import * 
from games.predator_prey import goal_distance, divide_obs

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=4, help="set a random seed")
    parser.add_argument("--num_seed", type=int, default=10, help="set a random seed")
    parser.add_argument("--game-setting", type=str, default="crowd_modelling_2d_four_rooms", help="Set the game to benchmark options:(crowd_modelling_2d_four_rooms) and (crowd_modelling_2d_maze)")
    
    parser.add_argument("--batch_step", type=int, default=200, help="set the number of episodes of to collect per rollout")
    parser.add_argument("--num_episodes", type=int, default=20, help="set the number of episodes of the inner loop")
    parser.add_argument("--num_iterations", type=int, default=50, help="Set the number of global update steps of the outer loop")
    
    #parser.add_argument("--path", type=str, default="/mnt/shunsuke/result/10xx/seed-42/multi_maze2_airl_1trajs", help="file path")
    #parser.add_argument('--logdir', type=str, default="/mnt/shunsuke/result/1112/multi_maze2_ppo_eval_airl_1trajs", help="logdir")

    parser.add_argument("--rew_index", type=int, default=-1, help="-1 is reward, 0 or more are output")
    parser.add_argument("--update_eps", type=str, default=r"200_1", help="file path")

    parser.add_argument("--single", action='store_true')
    parser.add_argument("--notmu", action='store_true')

    parser.add_argument("--reward_filename", type=str, default="disc_reward", help="file path")
    parser.add_argument("--value_filename", type=str, default="disc_value", help="file path")
    parser.add_argument("--actor_filename", type=str, default="actor", help="file path")

    args = parser.parse_args()
    return args
pathes = ["/mnt/shunsuke/result/10xx/multi_maze2_airl_1trajs/seed-42",
          "/mnt/shunsuke/result/10xx/multi_maze2_airl_15trajs/seed-42",
          "/mnt/shunsuke/result/10xx/multi_maze2_airl_100trajs/seed-42",
          "/mnt/shunsuke/result/10xx/multi_maze2_airl_1000trajs/seed-42",
          ]

logdirs = ["/mnt/shunsuke/result/10xx/multi_maze2_ppo_eval_airl_1trajs",
          "/mnt/shunsuke/result/10xx/multi_maze2_ppo_eval_airl_15trajs",
          "/mnt/shunsuke/result/10xx/multi_maze2_ppo_eval_airl_100trajs",
          "/mnt/shunsuke/result/10xx/multi_maze2_ppo_eval_airl_1000trajs",
          ]


if __name__ == "__main__":
    args = parse_args()
    seeds = np.arange(args.seed, args.seed+args.num_seed)

    for disc_idx, disc_path in enumerate(pathes):
        for seed in seeds:
            # Set the seed 
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            os.environ["PYTHONHASHSEED"] = str(seed)
            print(f"Random seed set as {seed}")

            single = args.single
            notmu = args.notmu

            update_eps_info = f'{args.update_eps}'
            logdir = osp.join(logdirs[disc_idx], f"seed-{seed}")
            logger.configure(logdir, format_strs=['stdout', 'log', 'json'])

            from open_spiel.python.mfg.algorithms.discriminator_networks_divided_value import * 
            is_nets = is_networks(disc_path)
            print(f'Is networks: {is_nets}')
            if not is_nets:
                from open_spiel.python.mfg.algorithms.discriminator import Discriminator
                rew_index = -1
                net_input = None
            else:
                net_input = get_net_input(disc_path)
                net_label = get_net_labels(net_input)
                is_divided = is_divided_value(disc_path)
                if not is_divided:
                    from open_spiel.python.mfg.algorithms.discriminator_networks import * 
                assert len(net_label)>=args.rew_index, 'rew_index is wrong'
                rew_index = args.rew_index


            # Create the game instance 
            game = pyspiel.load_game('python_mfg_predator_prey')
            states = game.new_initial_state()

            num_agent = game.num_players() 

            mfg_dists = []
            for i in range(num_agent):
                uniform_policy = policy_std.UniformRandomPolicy(game)
                start = time.time()

                mfg_dist = distribution.DistributionPolicy(game, uniform_policy)

                end = time.time()
                print(f'time: {end - start}s')

                mfg_dists.append(mfg_dist)
            merge_dist = distribution.MergeDistribution(game, mfg_dists)

            envs = []
            for i in range(num_agent):
                envs.append(rl_environment.Environment(game, mfg_distribution=merge_dist, mfg_population=i))
                envs[-1].seed(seed)
            
            conv_dist = convert_distrib(envs, merge_dist)
            device = torch.device("cpu")

            env = envs[0]
            nacs = env.action_spec()['num_actions']
            nobs = env.observation_spec()['info_state'][0]
            horizon = env.game.get_parameters()['horizon']

            nmu = num_agent
            size = env.game.get_parameters()['size']
            state_size = nobs -1 - horizon # nobs-1: obs size (exposed own mu), nmu: all agent mu size, horizon: horizon size
            obs_xym_size = nobs -1 - horizon + nmu # nobs-1: obs size (exposed own mu), nmu: all agent mu size, horizon: horizon size
            discriminators = []
            for i in range(num_agent):
                if single:
                    discriminator = Discriminator(nobs+1, nacs, False, device)
                elif notmu:
                    discriminator = Discriminator(nobs, nacs, False, device)
                elif is_nets:
                    inputs = get_input_shape(net_input, env, num_agent)
                    labels = get_net_labels(net_input)
                    num_hidden = get_num_hidden(disc_path)
                    print(num_hidden)
                    if len(labels)==2:
                        discriminator = Discriminator_2nets(inputs, obs_xym_size, labels, device, num_hidden=num_hidden)
                    if len(labels)==3:
                        discriminator = Discriminator_3nets(inputs, obs_xym_size, labels, device, num_hidden=num_hidden)
                else:
                    #discriminator = Discriminator(nobs-1+num_agent-horizon, nacs, False, device)
                    discriminator = Discriminator(3, nacs, True, device)

                if is_nets:
                    discriminator.load(disc_path, f'{update_eps_info}-{i}', use_eval=True)
                    discriminator.print_weights()
                else:
                    reward_path = osp.join(disc_path, args.reward_filename+update_eps_info + f'-{i}.pth')
                    value_path = osp.join(disc_path, args.value_filename+update_eps_info + f'-{i}.pth')
                    discriminator.load(reward_path, value_path, use_eval=True)
                    print(f'')
                discriminators.append(discriminator)
            
            """
            from multi_render_reward import multi_render_reward_nets_divided_value
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
            inputs = discriminators[0].create_inputs([size, size], nacs, horizon, mu_dists)
            disc_rewards, disc_outputs = multi_render_reward_nets_divided_value(size, nacs, horizon, inputs[0], discriminators[0], save=False, filename='test_disc_reward')
            """

            mfgppo = [MultiTypeMFGPPO(game, envs[i], merge_dist, conv_dist, discriminators[i], device, player_id=i, is_nets=is_nets, net_input=net_input, rew_index=rew_index) for i in range(num_agent)]

            batch_step = args.batch_step
            for niter in tqdm(range(args.num_iterations)):
                exp_ret = [[] for _ in range(num_agent)]
                for neps in range(args.num_episodes):
                    logger.record_tabular(f"num_iteration", niter)
                    logger.record_tabular(f"num_episodes", neps)
                    for i in range(num_agent):
                        obs_pth, actions_pth, logprobs_pth, rewards, true_rewards_pth, dones_pth, values_pth, entropies_pth, t_actions_pth, t_logprobs_pth, mu, ret \
                            = mfgppo[i].rollout(envs[i], args.batch_step)
                        adv_pth, returns = mfgppo[i].cal_Adv(rewards, values_pth, dones_pth)
                        v_loss = mfgppo[i].update_eps(obs_pth, logprobs_pth, actions_pth, adv_pth, returns, t_actions_pth, t_logprobs_pth) 
                        logger.record_tabular(f"total_loss {i}", v_loss.item())
                        exp_ret[i].append(np.mean(ret))
                        #print(f'Exp. ret{i} {np.mean(ret)}')

                mfg_dists = []
                for i in range(num_agent):
                    policy = mfgppo[i]._ppo_policy
                    start = time.time()
                    mfg_dist = distribution.DistributionPolicy(game, policy)
                    end = time.time()
                    print(f'time: {end - start}s')
                    mfg_dists.append(mfg_dist)
                
                merge_dist = distribution.MergeDistribution(game, mfg_dists)
                conv_dist = convert_distrib(envs, merge_dist)
                for i in range(num_agent):
                    print(f'update iter {i}')
                    nashc_ppo = mfgppo[i].update_iter(game, envs[i], merge_dist, conv_dist, nashc=True, population=i)
                    logger.record_tabular(f'NashC ppo{i}', nashc_ppo)
                    logger.record_tabular(f'Exp. Ret{i}', np.mean(exp_ret[i]))

                    fname = f'{niter}_{neps}-{i}'
                    mfgppo[i].save(game, fname)
                logger.dump_tabular()
            logger.reset()
                
