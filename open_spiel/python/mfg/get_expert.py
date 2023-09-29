import os
import click
import time
import numpy as np
import pickle as pkl

import torch
from open_spiel.python.mfg.games import factory
from open_spiel.python import rl_environment
from torch.distributions.categorical import Categorical
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms.mfg_ppo import Agent, PPOpolicy
from open_spiel.python import policy as policy_std
from utils import onehot, multionehot


@click.command()
@click.option('--path', type=click.STRING, default="result")
@click.option('--game_setting', type=click.STRING, default="crowd_modelling_2d_four_rooms")
@click.option('--distrib_filename', type=click.STRING, default="distrib.pth")
@click.option('--actor_filename', type=click.STRING, default="actor.pth")
@click.option('--num_trajs', type=click.INT, default=1000)
@click.option('--seed', type=click.INT, default=0)
@click.option('--num_acs', type=click.INT, default=5)
@click.option('--num_obs', type=click.INT, default=67)


def expert_generator(path, distrib_filename, actor_filename, num_trajs, game_setting, seed, num_acs, num_obs):
    device = torch.device("cpu")
    agent = Agent(num_obs, num_acs).to(device)
    actor_model = agent.actor

    game = factory.create_game_with_setting("mfg_crowd_modelling_2d", game_setting)

    # Set the initial policy to uniform and generate the distribution 
    #distrib_path = os.path.join(path, distrib_filename)
    #distrib = pkl.load(open(distrib_path, "rb"))
    ppo_policy = PPOpolicy(game, agent, None, device)
    mfg_dist = distribution.DistributionPolicy(game, ppo_policy)
    env = rl_environment.Environment(game, mfg_distribution=mfg_dist, mfg_population=0)

    # Set the environment seed for reproduciblility 
    env.seed(seed)

    n_agents = 1
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    num_players = env.num_players
    print(f"num players: {num_players}")

    filepath = os.path.join(path, actor_filename)
    print("load model from", filepath)
    actor_model.load_state_dict(torch.load(filepath))

    actor_model.eval()

    # output = model(input_data)
    def get_action(x):
        logits = actor_model(x)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action
        

    sample_trajs = []
    avg_ret = []

    for i in range(num_trajs):
        all_ob, all_ac, all_dist, all_rew = [], [], [], []
        ep_ret = 0.0

        time_step = env.reset()
        while not time_step.last():
            obs = time_step.observations["info_state"][0]
            obs_pth = torch.Tensor(obs).to(device)
            action = get_action(obs_pth)
            time_step = env.step([action.item()])
            rewards = time_step.rewards[0]
            dist = env.mfg_distribution

            all_ob.append(obs)
            all_ac.append(onehot(action.item(), num_actions))
            # all_dist.append(dist)
            all_rew.append(rewards)
            ep_ret += rewards

        avg_ret.append(ep_ret)
        traj_data = {
            "ob": all_ob, "ac": all_ac, "rew": all_rew, 
            "ep_ret": ep_ret
        }

        sample_trajs.append(traj_data)
        print(f'traj_num:{i}/{num_trajs}, expected_return:{ep_ret}')

    print(f'agent ret:{np.mean(avg_ret)}, std:{np.std(avg_ret)}')

    pkl.dump(sample_trajs, open(path + '/expert-%dtra.pkl' % num_trajs, 'wb'))

if __name__ == '__main__':
    expert_generator()
     
    # from dataset import Dset
    # sample_trajs = pkl.load(open(path + 'expert-%dtra.pkl' % num_trajs, 'rb'))
    # buffer = Dset(obs, actions, obs_next, all_obs, values,
    #                           randomize=True, num_agents=1, nobs_flag=True)
