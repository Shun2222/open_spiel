import os
import os.path as osp
import pyspiel
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


class NashC(NashConv):
    # Mainly used to calculate the exploitability 
    def __init__(self, game,distrib,pi_value, root_state=None):
        self._game = game
        if root_state is None:
            self._root_states = game.new_initial_states()
        else:
            self._root_states = [root_state]
            
        self._distrib = distrib

        self._pi_value = pi_value

        self._br_value = best_response_value.BestResponse(
            self._game,
            self._distrib,
            value.TabularValueFunction(self._game),
            root_state=root_state)

class Agent(nn.Module):
    def __init__(self, info_state_size, num_actions):
        super(Agent, self).__init__()
        self.num_actions = num_actions
        self.info_state_size = info_state_size
        self.critic = nn.Sequential(
            layer_init(nn.Linear(info_state_size, 128)), 
            nn.Tanh(),
            layer_init(nn.Linear(128,128)),
            nn.Tanh(),
            layer_init(nn.Linear(128,1))
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(info_state_size, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128,128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, num_actions))
        )
        

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def get_action(self, x):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        return action, probs.log_prob(action)

    def get_log_action_prob(self, states, actions):
        # print(f"obs: {states}")
        # print(f"acs: {actions}")
        logits = self.actor(states)
        logpac = -F.cross_entropy(logits, actions)
        # print(f"logpac: {logpac}")
        return logpac

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def layer_init(layer, bias_const=0.0):
    # used to initalize layers
    nn.init.xavier_normal_(layer.weight)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOpolicy(policy_std.Policy):
    # required obeject to work with openspiel 
    # used in updating the distribution using the policy 
    # and in calculating the nash-convergance 

    def __init__(self, game, agent, player_ids, device):
        super().__init__(game, player_ids)
        self.agent = agent
        self.device = device

    def action_probabilities(self, state, player_id=None):
        # main method that is called to update the population states distribution
        obs = torch.Tensor(state.observation_tensor()).to(self.device)
        legal_actions = state.legal_actions()
        logits = self.agent.actor(obs).detach().cpu()
        legat_logits = np.array([logits[action] for action in legal_actions])
        probs = np.exp(legat_logits -legat_logits.max())
        probs /= probs.sum(axis=0)
        
        # returns a dictionary with actions as keys and their probabilities as values
        return {action:probs[legal_actions.index(action)] for action in legal_actions}

def convert_distrib(envs, distrib):
    env = envs[0]
    num_agent = len(envs)
    horizon = env.game.get_parameters()['horizon']
    size = env.game.get_parameters()['size']
    mu_dist = [np.zeros((horizon, size, size)) for _ in range(num_agent)]

    for k,v in distrib.distribution.items():
        if "mu" in k:
            tt = k.split(",")
            pop = int(tt[0][-1])
            t = int(tt[1].split('=')[1].split('_')[0])
            xy = tt[2].split(" ")
            x = int(xy[1].split("[")[-1])
            y = int(xy[2].split("]")[0])
            mu_dist[pop][t,y,x] = v
    return mu_dist

class MultiTypeMFGPPO(object):
    def __init__(self, game, envs, device):
        self._device = device
        self._num_agent = len(envs)
        env = envs[0]

        info_state_size = env.observation_spec()["info_state"][0]
        num_actions = env.action_spec()["num_actions"]
        self._size = env.game.get_parameters()['size']

        self._eps_agents = []
        self._ppo_policies = []
        self._iter_agents = []
        self._optimizer_actors = []
        self._optimizer_critics = []
        mfg_dists = []
        for i in range(num_agent):
            # Creat the agent and population policies 
            self._eps_agents.append(Agent(info_state_size, num_actions).to(device))
            self._ppo_policies.append(PPOpolicy(game, self._eps_agents[-1], None, device))
            self._iter_agents.append(Agent(info_state_size, num_actions).to(device))

            self._optimizer_actors.append(optim.Adam(self._eps_agents[-1].actor.parameters(),lr=1e-3,eps=1e-5))
            self._optimizer_critics.append(optim.Adam(self._eps_agents[-1].critic.parameters(), lr=1e-3, eps=1e-5))
            #self._optimizer_actors.append(optim.SGD(self._eps_agents[-1].actor.parameters(), lr=1e-3, momentum=0.9))
            #self._optimizer_critics.append(optim.SGD(self._eps_agents[-1].critic.parameters(), lr=1e-3, momentum=0.9))

            mfg_dist = distribution.DistributionPolicy(game, self._ppo_policies[-1])
            mfg_dists.append(mfg_dist)
        
        self._merge_dist = distribution.MergeDistribution(game, mfg_dists)
        self._conv_dist = convert_distrib(envs, merge_dist)
        for env in envs:
          env.update_mfg_distribution(self._merge_dist)
        

    def rollout(self, envs, num_episodes, steps):
        device = self._device
        # generates num_episodes rollouts
        num_agent = self._num_agent

        info_state = [torch.zeros((steps,self._iter_agents[i].info_state_size), device=device) for i in range(num_agent)]
        obs_mu = [torch.zeros((steps, self._iter_agents[i].info_state_size+3), device=device) for i in range(num_agent)]
        actions = [torch.zeros((steps,), device=device) for _ in range(num_agent)]
        logprobs = [torch.zeros((steps,), device=device) for _ in range(num_agent)]
        rewards = [torch.zeros((steps,), device=device) for _ in range(num_agent)]
        dones = [torch.zeros((steps,), device=device) for _ in range(num_agent)]
        values = [torch.zeros((steps,), device=device) for _ in range(num_agent)]
        entropies = [torch.zeros((steps,), device=device) for _ in range(num_agent)]
        t_actions = [torch.zeros((steps,), device=device) for _ in range(num_agent)]
        t_logprobs = [torch.zeros((steps,), device=device) for _ in range(num_agent)]

        size = envs[0].game.get_parameters()['size']
        step = 0
        ret = np.zeros(num_agent) 
        for _ in range(num_episodes):
            time_steps = [envs[i].reset() for i in range(num_agent)]
            while not time_steps[0].last():
                mu = []
                for i in range(num_agent):
                    obs = time_steps[i].observations["info_state"][0]
                    obs = torch.Tensor(obs).to(device)
                    info_state[i][step] = obs
                    with torch.no_grad():
                        t_action, t_logprob, _, _ = self._iter_agents[i].get_action_and_value(obs)
                        action, logprob, entropy, value = self._eps_agents[i].get_action_and_value(obs)

                    # iteration policy data
                    t_logprobs[i][step] = t_logprob
                    t_actions[i][step] = t_action
                    logprobs[i][step] = logprob
                    entropies[i][step] = entropy
                    values[i][step] = value
                    actions[i][step] = action

                    time_steps[i] = envs[i].step([action.item()])

                    obs_list = list(obs.detach().numpy())
                    obs_x = obs_list[:size].index(1)
                    obs_y = obs_list[size:2*size].index(1)
                    obs_t = obs_list[2*size:].index(1)
                    mu.append(self._conv_dist[i][obs_t, obs_y, obs_x])

                for i in range(num_agent):
                    # episode policy data
                    dones[i][step] = time_steps[i].last()
                    rewards[i][step] = torch.Tensor(np.array(time_steps[i].rewards[i])).to(device)
                    ret[i] += time_steps[i].rewards[i]
                for i in range(num_agent):
                    ob_mu = list(info_state[i][step])
                    ob_mu += mu
                    obs_mu[i][step] = torch.Tensor(ob_mu).to(device)
                step += 1

        ret /= num_episodes
        return info_state, actions, logprobs, rewards, dones, values, entropies,t_actions,t_logprobs, obs_mu, ret

    def cal_Adv(self, rewards, values, dones, gamma=0.99, norm=True):
        # function used to calculate the Generalized Advantage estimate
        # using the exact method in stable-baseline3
        with torch.no_grad():
            next_done = dones[-1]
            next_value = values[-1] 
            nsteps = len(values)
            returns = torch.zeros_like(rewards).to(self._device)
            for t in reversed(range(nsteps)):
                if t == nsteps - 1:
                    nextnonterminal = 1.0 - next_done
                    next_return = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    next_return = returns[t + 1]
                returns[t] = rewards[t] + gamma * nextnonterminal * next_return

            advantages = returns - values

        if norm:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns


    def update_eps(self, idx, obs, logprobs, actions, advantages, returns, t_actions, t_logprobs, 
                    update_epochs=5, num_minibatch=5, alpha = 0.5, t_eps = 0.2, eps = 0.2,
                    ent_coef=0.01, max_grad_norm=5):
        # update the agent network (actor and critic)
        batch_size = actions.shape[0]
        b_inds = np.arange(batch_size)
        mini_batch_size = batch_size // num_minibatch
        # get batch indices
        np.random.shuffle(b_inds)
        for _ in range(update_epochs):
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                mb_inds = b_inds[start:end]
                # generate the new logprobs, entropy and value then calculate the ratio
                b_obs = obs[mb_inds]
                b_advantages = advantages[mb_inds]

                # Get the data under the episode policy (representative agent current policy)
                _, newlogprob, entropy, new_value = self._eps_agents[idx].get_action_and_value(b_obs, actions[mb_inds])
                logratio = newlogprob - logprobs[mb_inds]
                ratio = torch.exp(logratio)

                # Get the data under the iteration policy (the population policy)
                _, t_newlogprob, _, _ = self._eps_agents[idx].get_action_and_value(b_obs, t_actions[mb_inds])
                t_logratio = t_newlogprob - t_logprobs[mb_inds]
                t_ratio = torch.exp(t_logratio)

                # iteration update PPO
                t_pg_loss1 = b_advantages * t_ratio
                t_pg_loss2 = b_advantages * torch.clamp(t_ratio, 1 - t_eps, 1 + t_eps)
                
                # episodic update PPO 
                pg_loss1 = b_advantages * ratio
                pg_loss2 = b_advantages * torch.clamp(ratio, 1 - eps, 1 + eps)

                # Calculate the loss using our loss function 
                pg_loss = - alpha * torch.min(pg_loss1, pg_loss2).mean() - (1-alpha) * torch.min(t_pg_loss1, t_pg_loss2).mean()
                v_loss = F.smooth_l1_loss(new_value.reshape(-1), returns[mb_inds]).mean()
                entropy_loss = entropy.mean()

                loss = pg_loss - ent_coef * entropy_loss 
                
                # Actor update 
                self._optimizer_actors[idx].zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._eps_agents[idx].actor.parameters(), max_grad_norm)
                self._optimizer_actors[idx].step()
                
                # Critic update 
                self._optimizer_critics[idx].zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self._eps_agents[idx].critic.parameters(), max_grad_norm)
                self._optimizer_critics[idx].step()

        return v_loss

    def update_iter(self, game, envs):
        mfg_dists = []
        for i in range(self._num_agent):
            # Update the iteration policy with the new policy 
            self._iter_agents[i].load_state_dict(self._eps_agents[i].state_dict())
            
            # update the environment distribution 
            mfg_dist = distribution.DistributionPolicy(game, self._ppo_policies[-1])
            mfg_dists.append(mfg_dist)
        
        self._merge_dist = distribution.MergeDistribution(game, mfg_dists)
        self._conv_dist = convert_distrib(envs, self._merge_dist)
        for env in envs:
          env.update_mfg_distribution(self._merge_dist)
        return self._merge_dist, self._conv_dist

    def log_metrics(self, game):
        nash_conv = []
        for i in range(self._num_agent):
            policy = self._ppo_policies[i]
            distrib = self._merge_dist

            # this function is used to log the results to tensor board
            initial_states = game.new_initial_states()
            pi_value = policy_value.PolicyValue(game, distrib, policy, value.TabularValueFunction(game))
            nash_conv.append(NashC(game, distrib, pi_value).nash_conv())
        return nash_conv 

    def get_value(self, obs):
        values = []
        for i in range(self._num_agent):
            with torch.no_grad():
                value = self._eps_agents[i].get_value(obs)
                values.append(value)
        return values

    def get_action(self, obs):
        actions = []
        probs = []
        for i in range(self._num_agent):
            with torch.no_grad():
                action, prob = self._eps_agents[i].get_action(obs)
                actions.append(action)
                probs.append(prob)
        return actions, probs

    def get_log_action_prob(self, states, actions):
        logpacs = []
        for i in range(self._num_agent):
            with torch.no_grad():
                logpac = self._eps_agents[i].get_log_action_prob(states, actions)
                logpacs.append(logpac)
        return logpacs

    def save(self, game, filename=""):
        for i in range(num_agent):
            fname = osp.join(logger.get_dir(), f'actor_{filename}-{i}.pth')
            torch.save(self.eps__agents[i].actor.state_dict(), fname)

            fname = osp.join(logger.get_dir(), f'critic_{filename}-{i}.pth')
            torch.save(self._eps_agents[i].critic.state_dict(), fname)

            distrib = distribution.DistributionPolicy(game, ppo_policies[i])
            fname = osp.join(logger.get_dir(), f'merge_distrib_{filename}-{i}.pth')
            utils.save_parametric_distribution(self._merge_dist, fname)   

            fname = osp.join(logger.get_dir(), f'conv_distrib_{filename}-{i}.pkl')
            pkl.dump(self._conv_dist, open(fname, 'wb'))

    def load(self):
        return None
        

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="set a random seed")
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"), help="Set the name of this experiment")
    parser.add_argument("--game-name", type=str, default="python_mfg_predator_prey", help="Set the game name")
    parser.add_argument("--game-setting", type=str, default="", help="")
    parser.add_argument('--logdir', type=str, default="/mnt/shunsuke/result/test", help="logdir")
    parser.add_argument("--num_episodes", type=int, default=20, help="set the number of episodes of the inner loop")
    parser.add_argument("--num_iterations", type=int, default=100, help="Set the number of global update steps of the outer loop")
    parser.add_argument("--batch_step", type=int, default=200, help="set the number of episodes of to collect per rollout")
    
    args = parser.parse_args()

    return args



if __name__ == "__main__":
    args = parse_args()

    logger.configure(args.logdir, format_strs=['stdout', 'log', 'json'])

    # Set the seed 
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")
    args = parse_args()

    # Create the game instance 
    game = pyspiel.load_game('python_mfg_predator_prey')
    game.new_initial_state()

    num_agent = 3

    mfg_dists = []
    for i in range(num_agent):
        uniform_policy = policy_std.UniformRandomPolicy(game)
        mfg_dist = distribution.DistributionPolicy(game, uniform_policy)
        mfg_dists.append(mfg_dist)
    merge_dist = distribution.MergeDistribution(game, mfg_dists)
    
    envs = []
    for i in range(num_agent):
        envs.append(rl_environment.Environment(game, mfg_distribution=merge_dist, mfg_population=i))
        envs[-1].seed(args.seed)

    device = torch.device("cpu")
    mtmfgppo = MultiTypeMFGPPO(game, envs, device, logger)

    steps = args.num_episodes * envs[0].max_game_length
    episode_entropy = [[] for _ in range(num_agent)]
    total_entropy = [[] for _ in range(num_agent)]
    Nash_con_vect = [[] for _ in range(num_agent)]

    eps_reward = [[] for _ in range(num_agent)]
    total_reward = [[] for _ in range(num_agent)]
    for _ in tqdm(range(args.num_iterations)):
        for _ in range(args.num_episodes):
            obs_pth, actions_pth, logprobs_pth, true_rewards_pth, dones_pth, values_pth, entropies_pth, t_actions_pth, t_logprobs_pth, mu, ret \
                = mtmfgppo.rollout(envs, args.num_episodes, steps)
            v_loss = []
            for i in range(num_agent):
                episode_entropy[i].append(entropies_pth[i].mean().item())
                eps_reward[i].append(true_rewards_pth[i].sum().item()/args.num_episodes)

                # Calculate the advantage function 
                adv_pth, returns_pth = mtmfgppo.cal_Adv(true_rewards_pth[i], values_pth[i], dones_pth[i])

                # Update the learned policy and report loss for debugging
                v_loss.append(mtmfgppo.update_eps(i, obs_pth[i], logprobs_pth[i], actions_pth[i], adv_pth, returns_pth, t_actions_pth[i], t_logprobs_pth[i]))
            
                #collect and print the metrics
                total_reward[i].append(np.mean(eps_reward[i]))
                total_entropy[i].append(np.mean(episode_entropy[i]))

        merge_dist, conv_dist = mtmfgppo.update_iter(game, envs)
        nash_conv = mtmfgppo.log_metrics(game) 
        for i in range(num_agent):
            Nash_con_vect[i].append(nash_conv[i])
            logger.record_tabular(f"total_loss {i}", v_loss[i].item())
            logger.record_tabular(f"nash_conv {i}", Nash_con_vect[i][-1])
            logger.record_tabular(f"mean_reward {i}", total_reward[i][-1])
        logger.dump_tabular()
        ret = np.array(ret) 
