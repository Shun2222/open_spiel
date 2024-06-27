import logger
import argparse
import pyspiel
from open_spiel.python.mfg.games import predator_prey
from render import render
from open_spiel.python.mfg import utils
from open_spiel.python import rl_environment
from open_spiel.python import policy as policy_std
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms.nash_conv import NashConv
from open_spiel.python.mfg.algorithms import policy_value
from open_spiel.python.mfg.games import factory
from open_spiel.python.mfg import value
from open_spiel.python.mfg.algorithms import best_response_value
import numpy as np

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="/mnt/shunsuke/mfg_result", help="save dir")
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    param = {'geometry': predator_prey.Geometry.SQUARE}

    game = pyspiel.load_game('python_mfg_predator_prey')
    game.new_initial_state()

    num_agent = 3
    envs = []
    mfg_dists = []
    for i in range(num_agent):
        uniform_policy = policy_std.UniformRandomPolicy(game)
        mfg_dists.append(distribution.DistributionPolicy(game, uniform_policy))
    merge_dist = distribution.MergeDistribution(game, mfg_dists)

    envs = [rl_environment.Environment(game, mfg_distribution=merge_dist, mfg_population=i) for i in range(num_agent)]

    for i in range(num_agent):
        print(f'--------{i}--------')
        time_step = envs[i].reset()
        obs = time_step.observations["info_state"][0]
        rew = time_step.rewards
        print(f'obs: {obs}')
        print(f'obs shape: {np.array(obs).shape}')
        print(f'rew: {rew}')
    print(f'-----------------')

    step = 0
    env = envs[0]
    print(f'num_players: {env.num_players}')
    time_step = env.reset()
    while not time_step.last():
        obs = time_step.observations["info_state"][0]
        rew = time_step.rewards
        print(f'obs: {obs}')
        print(f'obs shape: {np.array(obs).shape}')
        print(f'rew: {rew}')
        action = 0
        time_step = env.step([action])

        uniform_policy = policy_std.UniformRandomPolicy(game)
        distrib = distribution.DistributionPolicy(game, uniform_policy)
        #env.update_mfg_distribution(distrib)
        step += 1
