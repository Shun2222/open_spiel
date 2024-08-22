import os
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
import pyspiel

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import logger
from dataset import MFGDataSet
from open_spiel.python.mfg import utils
from open_spiel.python import rl_environment
from open_spiel.python import policy as policy_std
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms.nash_conv import NashConv
from open_spiel.python.mfg.algorithms import policy_value
from open_spiel.python.mfg.games import factory
from open_spiel.python.mfg import value
from open_spiel.python.mfg.algorithms import best_response_value

from open_spiel.python.mfg.algorithms.multi_type_mfg_ppo import convert_distrib, Agent, PPOpolicy
from open_spiel.python.mfg.algorithms.discriminator_networks_divided_value import * 

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_path", type=str, default="/mnt/shunsuke/result/0726/multi_maze2_expert/expert-1000---", help="expert path")
    parser.add_argument("--expert_actor_path", type=str, default="/mnt/shunsuke/result/0726/multi_maze2_expert/actor50_19", help="expert actor path")
    parser.add_argument("--logdir", type=str, default="/mnt/shunsuke/result/0726/multi_maze2_dxy_mu-divided_value_common_skip_defagent_1traj", help="log path")
    parser.add_argument("--net_input", type=str, default="dxy_mu", help="log path")
    parser.add_argument("--num_hidden", type=int, default=1, help="log path")
    parser.add_argument("--use_ppo_value", action='store_true', help="cpu or cuda")

    parser.add_argument('--select_common', nargs='*', type=int, default=[0, 1, 2])
    parser.add_argument('--skip_train', nargs='*', default=["false", "false", "false"])
    parser.add_argument("--differ_expert", action='store_true', help="commonalize reward")

    parser.add_argument("--exp-name", type=str, default=".py", help="Set the name of this experiment")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate of the optimizer")
    parser.add_argument('--torch-deterministic', 
        type=lambda x:bool(strtobool(x)), default=True, nargs="?", 
        const=True, help="Use to repreduce experiment results")
    parser.add_argument("--game-setting", type=str, default="crowd_modelling_2d_four_rooms", help="Set the game to benchmark options:(crowd_modelling_2d_four_rooms) and (crowd_modelling_2d_maze)")
    parser.add_argument("--cuda", action='store_true', help="cpu or cuda")
    #parser.add_argument("--cpu", action='store_true', help="cpu or cuda")
    parser.add_argument("--seed", type=int, default=42, help="set a random seed")
    parser.add_argument("--batch_step", type=int, default=1200, help="set a step batch size")
    parser.add_argument("--traj_limitation", type=int, default=1000, help="set a traj limitation")
    parser.add_argument("--total_step", type=int, default=1.6e5, help="set a total step")
    parser.add_argument("--num_episode", type=int, default=100, help="")
    parser.add_argument("--save_interval", type=float, default=10, help="save models  per save_interval")
    args = parser.parse_args()
    return args

differ_expert_path = [
                        "/mnt/shunsuke/result/0726/multi_maze2_expert/expert-1tra",
                        "/mnt/shunsuke/result/0726/multi_maze2_expert/expert-1000tra",
                        "/mnt/shunsuke/result/0726/multi_maze2_expert/expert-1000tra",
                     ]
skip_agent_actor = [ 
                        {"folder" : "/mnt/shunsuke/result/master_middle/multi_maze2_ppo_dxy_mu_fixmu_1traj-dxyrew",
                         "update_info": "49_19" },
                        {},
                        {},
                   ]

if __name__ == "__main__":
    args = parse_args()

    is_common = np.max(args.select_common)+1!=3
    print(f'select common mode? {is_common}, {args.select_common}')
    if is_common:
        from open_spiel.python.mfg.algorithms.multi_type_adversarial_inverse_rl_networks_divided_value_selectable_common import MultiTypeAIRL
    else:
        from open_spiel.python.mfg.algorithms.multi_type_adversarial_inverse_rl_networks_divided_value import MultiTypeAIRL

    # Set the seed 
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

    #device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f'device: {device}')
    #batch_step = args.batch_step
    #update_generator_until = batch_step * 10
    expert_path = args.expert_path
    expert_actor_path = args.expert_actor_path
    traj_limitation = args.traj_limitation

    logger.configure(args.logdir, format_strs=['stdout', 'log', 'json'])
    assert get_net_input(args.logdir)==args.net_input
    print(f'Checked net input: OK')

    # Create the game instance 
    game = pyspiel.load_game('python_mfg_predator_prey')
    states = game.new_initial_state()

    num_agent = game.num_players() 

    assert len(args.skip_train)==num_agent, f"not match size of skip train and num agent: {len(args.skip_train)}, {num_agent}"
    skip_train = [args.skip_train[idx]=="true" for idx in range(num_agent)]
    print(f'skip train: {skip_train}')

    mfg_dists = []
    skip_agents = [None for _ in range(num_agent)]
    for i in range(num_agent):
        uniform_policy = policy_std.UniformRandomPolicy(game)
        mfg_dist = distribution.DistributionPolicy(game, uniform_policy)
        if skip_train[i]:
            agent =  Agent(61, 5).to(device)
            folder = skip_agent_actor[i]["folder"]
            fileupd = skip_agent_actor[i]["update_info"]
            filepath = os.path.join(folder, f"actor{fileupd}-{i}.pth")
            print("load actor model from", filepath)
            agent.actor.load_state_dict(torch.load(filepath))
            filepath = os.path.join(folder, f"critic{fileupd}-{i}.pth")
            agent.critic.load_state_dict(torch.load(filepath))
            ppo_policy = PPOpolicy(game, agent, None, device)
            mfg_dist = distribution.DistributionPolicy(game, ppo_policy)
            skip_agents[i] = agent
        mfg_dists.append(mfg_dist)
    merge_dist = distribution.MergeDistribution(game, mfg_dists)

    envs = []
    for i in range(num_agent):
        envs.append(rl_environment.Environment(game, mfg_distribution=merge_dist, mfg_population=i))
        envs[-1].seed(args.seed)

    conv_dist = convert_distrib(envs, merge_dist)
    device = torch.device("cpu")

    num_obs = envs[0].observation_spec()['info_state'][0]
    num_acs = envs[0].action_spec()['num_actions']

    # expertと比較用にモデルを読み込む
    expert_actor_pathes = [expert_actor_path + f'-{i}.pth' for i in range(num_agent)]
    ppo_policies = []
    for i in range(num_agent):
        agent =  Agent(num_obs, num_acs).to(device)
        filepath = os.path.join(expert_actor_pathes[i])
        actor_model = agent.actor
        print("load actor model from", filepath)
        actor_model.load_state_dict(torch.load(filepath))

        # Set the initial policy to uniform and generate the distribution 
        ppo_policies.append(PPOpolicy(game, agent, None, device))

    experts = []
    for i in range(num_agent):
        if args.differ_expert:
            fname = differ_expert_path[i] + f'-{i}.pkl'
        else:
            fname = expert_path + f'-{i}.pkl'
        expert = MFGDataSet(fname, traj_limitation=traj_limitation, nobs_flag=True)
        experts.append(expert)
        print(f'expert load from {fname}')
    airl = MultiTypeAIRL(game, envs, merge_dist, conv_dist, device, experts, ppo_policies, disc_type=args.net_input, disc_num_hidden=args.num_hidden, use_ppo_value=args.use_ppo_value, skip_train=skip_train, skip_agents=skip_agents, disc_index=args.select_common)
    airl.run(args.total_step, None, \
        args.num_episode, args.batch_step, args.save_interval)


