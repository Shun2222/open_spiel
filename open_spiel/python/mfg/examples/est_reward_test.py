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
from open_spiel.python.mfg import utils
from open_spiel.python import rl_environment
from open_spiel.python import policy as policy_std
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms.nash_conv import NashConv
from open_spiel.python.mfg.algorithms import policy_value
from open_spiel.python.mfg.games import factory
from open_spiel.python.mfg import value
from open_spiel.python.mfg.algorithms import best_response_value

import os.path as osp
import random
#import joblib
from utils import onehot, multionehot
from scipy.stats import pearsonr, spearmanr

from open_spiel.python.mfg.algorithms.mfg_ppo import MFGPPO
from open_spiel.python.mfg.algorithms.discriminator import Discriminator

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=".py", help="Set the name of this experiment")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate of the optimizer")
    parser.add_argument('--torch-deterministic', 
        type=lambda x:bool(strtobool(x)), default=True, nargs="?", 
        const=True, help="Use to repreduce experiment results")


    parser.add_argument("--game-setting", type=str, default="crowd_modelling_2d_four_rooms", help="Set the game to benchmark options:(crowd_modelling_2d_four_rooms) and (crowd_modelling_2d_maze)")
    parser.add_argument("--cuda", action='store_true', help="cpu or cuda")
    parser.add_argument("--logdir", type=str, default="/mnt/shunsuke/mfg_result/batch-test/batch1200/learn_est_reward_best", help="reward path")
    parser.add_argument("--reward_dir", type=str, default="/mnt/shunsuke/mfg_result/batch-test/batch1200/disc_reward650_129.pth", help="reward path")
    parser.add_argument("--value_dir", type=str, default="/mnt/shunsuke/mfg_result/batch-test/batch1200/disc_value650_129.pth", help="value path")
    #parser.add_argument("--cpu", action='store_true', help="cpu or cuda")
    parser.add_argument("--seed", type=int, default=42, help="set a random seed")
    parser.add_argument("--batch_step", type=int, default=200, help="set a step batch size")
    parser.add_argument("--total_step", type=int, default=4e5, help="set a total step")
    parser.add_argument("--num_episode", type=int, default=20, help="")
    parser.add_argument("--save_interval", type=float, default=100, help="save models  per save_interval")
    args = parser.parse_args()
    return args




class AIRL(object):
    def __init__(self, game, env, device, reward_path, value_path):
        self._game = game
        self._env = env
        self._device = device

        self._nacs = env.action_spec()['num_actions']
        self._nobs = env.observation_spec()['info_state'][0]
        self._nmu  = 1

        self._generator = MFGPPO(game, env, device)
        self._discriminator = Discriminator(self._nobs+self._nmu, self._nacs, False, device)
        self._discriminator.load(reward_path, value_path, use_eval=True)


    def run(self, total_step, num_episodes, batch_step, save_interval=1000):
        logger.record_tabular("total_step", total_step)
        logger.record_tabular("num_episodes", num_episodes)
        logger.record_tabular("batch_step", batch_step)
        logger.dump_tabular()

        t_step = 0
        num_update_eps = 0
        num_update_iter = 0
        batch_step = (batch_step//self._env.max_game_length) * self._env.max_game_length
        buffer = None
        while(t_step < total_step):
            for neps in range(num_episodes): 
                obs_pth, actions_pth, logprobs_pth, true_rewards_pth, dones_pth, values_pth, entropies_pth, t_actions_pth, t_logprobs_pth, obs_mu_pth, ret \
                    = self._generator.rollout(self._env, batch_step)

                obs = obs_pth.cpu().detach().numpy()
                nobs = obs.copy()
                nobs[:-1] = obs[1:]
                nobs[-1] = obs[0]
                obs_next = nobs
                obs_next_pth = torch.from_numpy(obs_next).to(self._device)
                actions = actions_pth.cpu().detach().numpy()
                logprobs = logprobs_pth.cpu().detach().numpy()
                true_rewards = true_rewards_pth.cpu().detach().numpy()
                dones = dones_pth.cpu().detach().numpy()
                values = values_pth.cpu().detach().numpy()
                entropies = entropies_pth.cpu().detach().numpy()
                t_actions = t_actions_pth.cpu().detach().numpy()
                t_logprobs = t_logprobs_pth.cpu().detach().numpy()
                obs_mu = obs_mu_pth.cpu().detach().numpy()
                nobs = obs_mu.copy()
                nobs[:-1] = obs_mu[1:]
                nobs[-1] = obs_mu[0]
                obs_next_mu = nobs
                obs_next_mu_pth = torch.from_numpy(obs_next_mu).to(self._device)

                
                disc_rewards_pth = self._discriminator.get_reward( 
                    torch.from_numpy(obs_mu).to(self._device),
                    torch.from_numpy(multionehot(actions, self._nacs)).to(self._device),
                    torch.from_numpy(obs_next).to(self._device),
                    torch.from_numpy(logprobs).to(self._device),
                    discrim_score=False) # For competitive tasks, log(D) - log(1-D) empirically works better (discrim_score=True)

                disc_rewards = disc_rewards_pth.cpu().detach().numpy().reshape(batch_step)
                disc_rewards_pth = torch.from_numpy(disc_rewards).to(self._device)

                #if t_step < total_step_gen:
                adv_pth, returns = self._generator.cal_Adv(disc_rewards_pth, values_pth, dones_pth)
                v_loss = self._generator.update_eps(obs_pth, logprobs_pth, actions_pth, adv_pth, returns, t_actions_pth, t_logprobs_pth) 


                pear = ""
                spear = ""
                try:
                    pear = pearsonr(disc_rewards.flatten(), true_rewards.flatten())[0]
                    spear = spearmanr(disc_rewards.flatten(), true_rewards.flatten())[0]
                except:
                    pass
                logger.record_tabular("timestep", t_step)
                logger.record_tabular("generator_loss", v_loss.item())
                logger.record_tabular("mean_ret", np.mean(ret))
                logger.record_tabular("pearsonr", pear)
                logger.record_tabular("spearson", spear)
                logger.dump_tabular()

                t_step += batch_step 
                num_update_eps += 1
                if(num_update_eps%save_interval==0):
                    fname = f"{num_update_eps}_{num_update_iter}"
                    self._generator.save(self._game, filename=fname)

            #if t_step < total_step_gen:
            nashc = self._generator.update_iter(self._game, self._env, nashc=True)
            logger.record_tabular("nashc", nashc)
            logger.dump_tabular()
            num_update_iter += 1
        self._generator.save(self._game)


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

    logger.configure(args.logdir, format_strs=['stdout', 'log', 'json'])

    # Create the game instance 
    game = factory.create_game_with_setting("mfg_crowd_modelling_2d", args.game_setting)

    # Set the initial policy to uniform and generate the distribution 
    uniform_policy = policy_std.UniformRandomPolicy(game)
    mfg_dist = distribution.DistributionPolicy(game, uniform_policy)
    env = rl_environment.Environment(game, mfg_distribution=mfg_dist, mfg_population=0)

    # Set the environment seed for reproduciblility 
    env.seed(args.seed)

    airl = AIRL(game, env, device, args.reward_dir, args.value_dir)
    airl.run(args.total_step, \
        args.num_episode, args.batch_step, args.save_interval)

