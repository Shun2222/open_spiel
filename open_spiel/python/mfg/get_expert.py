import os
import click
import time
import numpy as np
import pickle as pkl

from torch.distributions.categorical import Categorical
from open_spiel.python.mfg.algorithms.mfg_ppo import Agent


@click.command()
@click.option('--env', type=click.STRING)
@click.option('--path', type=click.STRING, default="")
@click.option('--filename', type=click.STRING, default="")
@click.option('--num_trajs', type=click.INT, default=100)

def expert_generator(env, image, all, path, filename):
    game = factory.create_game_with_setting("mfg_crowd_modelling_2d", args.game_setting)

    # Set the initial policy to uniform and generate the distribution 
    uniform_policy = policy_std.UniformRandomPolicy(game)
    mfg_dist = distribution.DistributionPolicy(game, uniform_policy)
    env = rl_environment.Environment(game, mfg_distribution=mfg_dist, mfg_population=0)

    # Set the environment seed for reproduciblility 
    env.seed(args.seed)

    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]
    device = torch.device("cpu")
    agent = Agent(info_state_size, num_actions).to(device)
    actor_model = agent.actor

    print("load model from", path)
    # 保存された.pthファイルからモデルの重みを読み込む
    filepath = os.path.join(path, filename)
    actor_model.load_state_dict(torch.load(filepath))

    # モデルを評価モードに設定（推論用）
    actor_model.eval()

    # モデルを使用して予測などを行う
    # output = model(input_data)
    def get_action(x):
        logits = actor_model(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action
        

    images = []
    sample_trajs = []
    avg_ret = [[] for _ in range(n_agents)]

    n_agent = 1
    for i in range(num_trajs):
        all_ob, all_agent_ob, all_ac, all_rew, ep_ret = [], [], [], [], [0 for k in range(n_agents)]
        for k in range(n_agents):
            all_ob.append([])
            all_ac.append([])
            all_rew.append([])
        obs = env.reset()
        obs = [ob[None, :] for ob in obs]
        action = [np.zeros([1]) for _ in range(n_agents)]

        for _ in range(num_trajs):
            time_step = env.reset()
            while not time_step.last():
                obs = time_step.observations["info_state"][0]
                obs_pth = torch.Tensor(obs).to(device)
                action = get_action(obs_pth)
                time_step = env.step([action.item()])

                all_ob[0].append([obs])
                all_ac[0].append(action)
                all_rew.append(rewards)
                ep_ret += rewards

            all_ob[0] = np.squeeze(all_ob[0])

            all_agent_ob = np.squeeze(all_agent_ob)
            traj_data = {
                "ob": all_ob, "ac": all_ac, "rew": all_rew,
                "ep_ret": ep_ret, "all_ob": all_agent_ob
            }

            sample_trajs.append(traj_data)
            # print('traj_num', i, 'expected_return', ep_ret)

            avg_ret[0].append(ep_ret[0])

    print(path)
    print(f'agent ret:{np.mean(avg_ret[0])}, std:{np.std(avg_ret[0])}')

    pkl.dump(sample_trajs, open(path + '-%dtra.pkl' % num_trajs, 'wb'))

if __name__ == '__main__':
    expert_generator()
