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
from open_spiel.python.mfg.algorithms.mfg_ppo import Agent
from open_spiel.python import policy as policy_std


@click.command()
@click.option('--env', type=click.STRING)
@click.option('--path', type=click.STRING, default="result")
@click.option('--game_setting', type=click.STRING, default="crowd_modelling_2d_four_rooms")
@click.option('--filename', type=click.STRING, default="actor.pth")
@click.option('--num_trajs', type=click.INT, default=100)
@click.option('--seed', type=click.INT, default=0)

def expert_generator(env, path, filename, num_trajs, game_setting, seed):
    game = factory.create_game_with_setting("mfg_crowd_modelling_2d", game_setting)

    # Set the initial policy to uniform and generate the distribution 
    uniform_policy = policy_std.UniformRandomPolicy(game)
    mfg_dist = distribution.DistributionPolicy(game, uniform_policy)
    env = rl_environment.Environment(game, mfg_distribution=mfg_dist, mfg_population=0)

    # Set the environment seed for reproduciblility 
    env.seed(seed)

    n_agents = 1
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]
    device = torch.device("cpu")
    agent = Agent(info_state_size, num_actions).to(device)
    actor_model = agent.actor

    print("load model from", path)
    filepath = os.path.join(path, filename)
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
        all_ob, all_ac, all_rew = [], [], []
        ep_ret = 0.0

        time_step = env.reset()
        while not time_step.last():
            obs = time_step.observations["info_state"][0]
            obs_pth = torch.Tensor(obs).to(device)
            action = get_action(obs_pth)
            time_step = env.step([action.item()])
            rewards = time_step.rewards[0]

            all_ob.append(obs)
            all_ac.append(action)
            all_rew.append(rewards)
            ep_ret += rewards

        avg_ret.append(ep_ret)
        traj_data = {
            "ob": all_ob, "ac": all_ac, "rew": all_rew, "ep_ret": ep_ret
        }

        sample_trajs.append(traj_data)
        print(f'traj_num:{i}/{num_trajs}, expected_return:{ep_ret}')

    print(path)
    print(f'agent ret:{np.mean(avg_ret)}, std:{np.std(avg_ret)}')

    pkl.dump(sample_trajs, open(path + 'expert-%dtra.pkl' % num_trajs, 'wb'))

if __name__ == '__main__':
    expert_generator()
