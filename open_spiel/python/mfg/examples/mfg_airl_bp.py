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

from open_spiel.python.mfg.algorithms.adversarial_inverse_rl import AIRL


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=".py", help="Set the name of this experiment")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate of the optimizer")
    parser.add_argument('--torch-deterministic', 
        type=lambda x:bool(strtobool(x)), default=True, nargs="?", 
        const=True, help="Use to repreduce experiment results")


    parser.add_argument("--game-setting", type=str, default="crowd_modelling_2d_four_rooms", help="Set the game to benchmark options:(crowd_modelling_2d_four_rooms) and (crowd_modelling_2d_maze)")
    parser.add_argument("--expert_path", type=str, default="/mnt/shunsuke/result/mfgPPO-dist1.0/expert-1000tra.pkl", help="expert path")
    parser.add_argument("--logdir", type=str, default="/mnt/shunsuke/mfg_result/episode-test/episode1", help="log path")
    parser.add_argument("--cuda", action='store_true', help="cpu or cuda")
    #parser.add_argument("--cpu", action='store_true', help="cpu or cuda")
    parser.add_argument("--seed", type=int, default=42, help="set a random seed")
    parser.add_argument("--batch_step", type=int, default=1200, help="set a step batch size")
    parser.add_argument("--traj_limitation", type=int, default=1000, help="set a traj limitation")
    parser.add_argument("--total_step", type=int, default=2.5e7, help="set a total step")
    parser.add_argument("--num_episode", type=int, default=1, help="")
    parser.add_argument("--save_interval", type=float, default=10, help="save models  per save_interval")
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

    #device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f'device: {device}')
    #batch_step = args.batch_step
    #update_generator_until = batch_step * 10
    expert_path = args.expert_path
    traj_limitation = args.traj_limitation

    logger.configure(args.logdir, format_strs=['stdout', 'log', 'json'])

    # Create the game instance 
    game = factory.create_game_with_setting("mfg_crowd_modelling_2d", args.game_setting)

    # Set the initial policy to uniform and generate the distribution 
    uniform_policy = policy_std.UniformRandomPolicy(game)
    mfg_dist = distribution.DistributionPolicy(game, uniform_policy)
    env = rl_environment.Environment(game, mfg_distribution=mfg_dist, mfg_population=0)

    # Set the environment seed for reproduciblility 
    env.seed(args.seed)

    expert = MFGDataSet(expert_path, traj_limitation=traj_limitation, nobs_flag=True)
    airl = AIRL(game, env, device, expert)
    airl.run(args.total_step, None, \
        args.num_episode, args.batch_step, args.save_interval)

