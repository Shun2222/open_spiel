import os
import click
import time
import numpy as np
import pickle as pkl

import torch
import pyspiel
import distribution
from open_spiel.python.mfg.games import factory
from open_spiel.python import rl_environment
from torch.distributions.categorical import Categorical
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms.mfg_ppo import Agent, PPOpolicy
from open_spiel.python.mfg.algorithms.multi_type_mfg_ppo import MultiTypeMFGPPO, convert_distrib
from open_spiel.python import policy as policy_std
from utils import onehot, multionehot
#from render import render
from Multi_type_render import multi_type_render


@click.command()
@click.option('--path', type=click.STRING, default="/mnt/shunsuke/result/mtmfgppo")
@click.option('--game_setting', type=click.STRING, default="crowd_modelling_2d_four_rooms")
@click.option('--distrib_filename', type=click.STRING, default="distrib.pkl")
@click.option('--actor_filename', type=click.STRING, default="actor.pth")
@click.option('--critic_filename', type=click.STRING, default="critic.pth")
@click.option('--num_trajs', type=click.INT, default=1000)
@click.option('--seed', type=click.INT, default=0)
@click.option('--num_acs', type=click.INT, default=5)
@click.option('--num_obs', type=click.INT, default=67)


def expert_generator(path, distrib_filename, actor_filename, critic_filename, num_trajs, game_setting, seed, num_acs, num_obs):
    device = torch.device("cpu")
    agent = Agent(num_obs, num_acs).to(device)
    actor_model = agent.actor
    filepath = os.path.join(path, actor_filename)
    print("load actor model from", filepath)
    actor_model.load_state_dict(torch.load(filepath))
    filepath = os.path.join(path, critic_filename)
    print("load critic model from", filepath)
    agent.critic.load_state_dict(torch.load(filepath))

    game_name = "mfg_crowd_modelling_2d"
    game = factory.create_game_with_setting(game_name, game_setting)

    # Set the initial policy to uniform and generate the distribution 
    distrib_path = os.path.join(path, distrib_filename)
    distrib = pkl.load(open(distrib_path, "rb"))
    ppo_policy = PPOpolicy(game, agent, None, device)
    mfg_dist = distribution.DistributionPolicy(game, ppo_policy)
    mfg_dist.set_params(distrib)
    # uniform_policy = policy_std.UniformRandomPolicy(game)
    # mfg_dist = distribution.DistributionPolicy(game, uniform_policy)

    env = rl_environment.Environment(game, mfg_distribution=mfg_dist, mfg_population=0)

    # Set the environment seed for reproduciblility 
    env.seed(seed)

    n_agents = 1
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    num_players = env.num_players
    print(f"num players: {num_players}")

    d_size = 13
    size = env.game.get_parameters()['size']
    horizon = env.game.get_parameters()['horizon']
    mu_dist = np.zeros((horizon,d_size,d_size))
    for k,v in mfg_dist.distribution.items():
        if "mu" in k:
            tt = k.split("_")[0].split(",")
            x = int(tt[0].split("(")[-1])
            y = int(tt[1].split()[-1])
            t = int(tt[2].split()[-1].split(")")[0])
            mu_dist[t,y,x] = v

    actor_model.eval()

    # output = model(input_data)
    def get_action(x):
        logits = actor_model(x)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action
        

    sample_trajs = []
    avg_ret = []
    best_traj = None

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

            obs_x = obs[:size].index(1)
            obs_y = obs[size:2*size].index(1)
            obs_t = obs[2*size:].index(1)
            obs_mu = obs.copy()
            obs_mu.append(mu_dist[obs_t, obs_y, obs_x])

            all_ob.append(obs_mu)
            all_ac.append(onehot(action.item(), num_actions))
            # all_dist.append(dist)
            all_rew.append(rewards)
            ep_ret += rewards

        avg_ret.append(ep_ret)
        traj_data = {
            "ob": all_ob, "ac": all_ac, "rew": all_rew, 
            "ep_ret": ep_ret
        }
        if best_traj==None:
            best_traj = traj_data
        elif best_traj["ep_ret"] < ep_ret:
            best_traj = traj_data

        sample_trajs.append(traj_data)
        print(f'traj_num:{i}/{num_trajs}, expected_return:{ep_ret}')

    print(f'expert avg ret:{np.mean(avg_ret)}, std:{np.std(avg_ret)}')
    print(f'best traj ret: {best_traj["ep_ret"]}')

    save_path = os.path.join(path, "agent_dist.mp4")
    pkl.dump(sample_trajs, open(path + '/expert-%dtra.pkl' % num_trajs, 'wb'))

    render(env, mfg_dist, np.array(best_traj["ob"]), save=True, filename=path+"/expert_best.mp4")
    print(f"Saved expert trajs and best expert mp4 in {path}")


@click.command()
@click.option('--path', type=click.STRING, default="/mnt/shunsuke/result/multi_type_maze_test")
@click.option('--game_setting', type=click.STRING, default="crowd_modelling_2d_four_rooms")
@click.option('--distrib_filename', type=click.STRING, default="distrib50_19")
@click.option('--actor_filename', type=click.STRING, default="actor50_19")
@click.option('--critic_filename', type=click.STRING, default="critic50_19")
@click.option('--num_trajs', type=click.INT, default=1000)
@click.option('--seed', type=click.INT, default=0)
@click.option('--num_acs', type=click.INT, default=5)
@click.option('--num_obs', type=click.INT, default=61)
def multi_type_expert_generator(path, distrib_filename, actor_filename, critic_filename, num_trajs, game_setting, seed, num_acs, num_obs):
    device = torch.device("cpu")
    game = pyspiel.load_game('python_mfg_predator_prey')
    states = game.new_initial_state()
    n_agents = num_agent = game.num_players()
    print(f"num players: {num_agent}")

    agents = []
    actor_models = []
    ppo_policies = []
    distribs = []
    mfg_dists = []
    for i in range(num_agent):
        agents.append(Agent(num_obs, num_acs).to(device))
        actor_model = agents[-1].actor
        filepath = os.path.join(path, actor_filename + f"-{i}.pth")
        print("load actor model from", filepath)
        actor_model.load_state_dict(torch.load(filepath))
        actor_models.append(actor_model)
        filepath = os.path.join(path, critic_filename + f"-{i}.pth")
        print("load critic model from", filepath)
        agents[-1].critic.load_state_dict(torch.load(filepath))

        # Set the initial policy to uniform and generate the distribution 
        distrib_path = os.path.join(path, distrib_filename + f"-{i}.pth")
        distribs.append(pkl.load(open(distrib_path, "rb")))
        ppo_policies.append(PPOpolicy(game, agents[-1], None, device))
        mfg_dist = distribution.DistributionPolicy(game, ppo_policies[-1])
        mfg_dist.set_params(distribs[-1])
        mfg_dists.append(mfg_dist)

    
    merge_dist = distribution.MergeDistribution(game, mfg_dists)

    envs = []
    for i in range(num_agent):
        envs.append(rl_environment.Environment(game, mfg_distribution=merge_dist, mfg_population=i))
        envs[-1].seed(seed)


    env = envs[0]
    info_state_size = env.observation_spec()["info_state"][0]
    horizon = env.game.get_parameters()['horizon']
    num_actions = env.action_spec()["num_actions"]
    size = env.game.get_parameters()['size']
    print(horizon)


    conv_dist = convert_distrib(envs, merge_dist)
    actor_model.eval()

    # output = model(input_data)
    def get_action(x, i):
        logits = actor_models[i](x)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action
        

    sample_trajs = [[] for _ in range(num_agent)]
    avg_ret = [[] for _ in range(num_agent)]
    best_traj = [None for _ in range(num_agent)]
    info_states = [[] for _ in range(num_agent)]

    for idx in range(num_agent):
        for i in range(num_trajs):
            all_ob, all_ac, all_dist, all_rew = [], [], [], []
            ep_ret = 0.0

            time_step = envs[idx].reset()
            while not time_step.last():
                obs = time_step.observations["info_state"][idx]
                obs_pth = torch.Tensor(obs).to(device)
                action = get_action(obs_pth, idx)
                time_step = envs[idx].step([action.item()])
                rewards = time_step.rewards[idx]
                dist = envs[idx].mfg_distribution

                obs_x = obs[:size].index(1)
                obs_y = obs[size:2*size].index(1)
                obs_t = obs[2*size:].index(1)
                obs_mu = obs.copy()
                for k in range(num_agent):
                    obs_mu.append(conv_dist[k][obs_t, obs_y, obs_x])

                all_ob.append(obs_mu)
                all_ac.append(onehot(action.item(), num_actions))
                all_rew.append(rewards)
                ep_ret += rewards
                if len(info_states[idx])<horizon:
                    info_states[idx].append(obs)

            traj_data = {
                "ob": all_ob, "ac": all_ac, "rew": all_rew, 
                "ep_ret": ep_ret
            }
            if best_traj[idx]==None:
                best_traj[idx] = traj_data
            elif best_traj[idx]["ep_ret"] < ep_ret:
                best_traj[idx] = traj_data

            sample_trajs[idx].append(traj_data)
            print(f'traj_num:{i}/{num_trajs}, expected_return:{ep_ret}')
            avg_ret[idx].append(ep_ret)

        print(f'expert avg ret{idx}:{np.mean(avg_ret[idx])}, std:{np.std(avg_ret[idx])}')
        print(f'best traj ret{idx}: {best_traj[idx]["ep_ret"]}')

    for i in range(len(sample_trajs)):
        fname = path + f'/expert-{num_trajs}tra-{i}.pkl'
        pkl.dump(sample_trajs[i], open(fname,  'wb'))
        print(f'Saved {fname}')

    for i in range(num_agent):
        info_states[i] = np.array(info_states[i])
    multi_type_render(envs, merge_dist, info_states, save=True, filename=path+f"/expert_best{idx}.mp4")
    #print(f"Saved expert trajs and best expert mp4 in {path}")


if __name__ == '__main__':
    #expert_generator()
    #multi_type_expert_generator(path, distrib_filename, actor_filename, critic_filename, num_trajs, game_setting, seed, num_acs, num_obs)
    multi_type_expert_generator()
