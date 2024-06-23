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
import time
import logger
import logging
import pickle as pkl
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from distutils.util import strtobool
from open_spiel.python.mfg import utils
from open_spiel.python import rl_environment
from open_spiel.python import policy as policy_std
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms.nash_conv import NashConv
from open_spiel.python.mfg.algorithms import policy_value
from open_spiel.python.mfg.games import factory
from open_spiel.python.mfg import value
from open_spiel.python.mfg.algorithms import best_response_value
from games.predator_prey import goal_distance, divide_obs

from scipy.spatial import distance
from scipy.stats import spearmanr
from scipy.special import kl_div

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

class NashC(NashConv):
    # Mainly used to calculate the exploitability 
    def __init__(self, game, distrib, pi_value, root_state=None):
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

class MultiTypeMFGPPO(object):
    def __init__(self, game, env, merge_dist, conv_dist, discriminator, device, player_id=0, expert_policy=None, is_nets=True, net_input=None, rew_index=-1):
        self._device = device
        self._rew_index = rew_index

        info_state_size = env.observation_spec()["info_state"][0]
        self._nacs = num_actions = env.action_spec()["num_actions"]
        self._num_agent = game.num_players()
        self._player_id = player_id

        self._eps_agent = Agent(info_state_size,num_actions).to(self._device) 
        self._ppo_policy = PPOpolicy(game, self._eps_agent, None, self._device) 
        self._iter_agent = Agent(info_state_size,num_actions).to(self._device)
        self._optimizer_actor = optim.Adam(self._eps_agent.actor.parameters(), lr=1e-3,eps=1e-5)
        self._optimizer_critic = optim.Adam(self._eps_agent.critic.parameters(), lr=1e-3,eps=1e-5)
        self._discriminator = discriminator
        self._is_nets = is_nets
        self._net_input = net_input

        self._horizon = env.game.get_parameters()['horizon']
        self._size = env.game.get_parameters()['size']

        env.update_mfg_distribution(merge_dist)
        self._mu_dist = conv_dist 

        self._expert_policy = expert_policy

    def rollout(self, env, nsteps):
        n_nets = self._discriminator.get_num_nets()
        vs = np.arange(-0.3, 0.31, 0.01)
        grids = np.meshgrid(*[vs] * n_nets)
        combinations = np.vstack([grid.ravel() for grid in grids]).T
        all_p_tau = {}
        all_p_tau2 = {}
        all_rew = []
        all_rew2 = {}
        inputs = self._discriminator.create_inputs([self._size, self._size], self._nacs, self._horizon, self._mu_dist)
        for rate in combinations:
            all_rew2[rate_str] = []
        for key, v in inputs[self._player_id].items():
            v = [vi.reshape(1, len(vi)) for vi in v]
            for rate in combinations:
                print(f'key:{key}, value:{v}')
                rew, rew2, _, _ = self._discriminator.get_reward_weighted(
                    v,
                    None, None, None, 
                    rate=rate,
                    expert_prob=False) # For competitive tasks, log(D) - log(1-D) empirically works better (discrim_score=True)
                rate_str = f'{rate}'
                all_rew2[rate_str].append(rew2)
            all_rew.append(rew)

        cos_sims = {} 
        spearmanrs = {}
        kl_divs = {}
        cos_sims_rews = {} 
        spearmanrs_rews = {}
        kl_divs_rews = {}
        #euclids = {}
        rew = np.array(all_rew).flatten()
        for rate_str in all_p_tau.keys():
            rew2 = np.array(all_rew2[rate_str])

            cos_sim_rew = 1-distance.cosine(rew, rew2)
            sp_rew, p_value = spearmanr(rew, rew2)
            kl_div_rew = np.sum([ai * np.log(ai / bi) for ai, bi in zip(rew, rew2)]) 
            #euclid = np.sqrt(np.sum((p_tau-p_tau2)**2))

            cos_sims_rews[rate_str] = cos_sim_rew
            spearmanrs_rews[rate_str] = sp_rew
            kl_divs_rews[rate_str] = kl_div_rew

        pkl.dump(rew, open('rew_all_state.pkl', 'wb'))
        pkl.dump(rew2, open('rew2_all_state.pkl', 'wb'))
        pkl.dump(cos_sims_rews, open('rew_cos_sims_all_state.pkl', 'wb'))
        pkl.dump(spearmanrs_rews, open('rew_spearmanrs_all_state.pkl', 'wb'))
        pkl.dump(kl_divs_rews, open('rew_kl_div_all_state.pkl', 'wb'))
        print(f'dumped all state pkl')

        num_agent = self._num_agent
        info_state = torch.zeros((nsteps,self._iter_agent.info_state_size), device=self._device)
        actions = torch.zeros((nsteps,), device=self._device)
        logprobs = torch.zeros((nsteps,), device=self._device)
        rewards = torch.zeros((nsteps,), device=self._device)
        true_rewards = torch.zeros((nsteps,), device=self._device)
        dones = torch.zeros((nsteps,), device=self._device)
        values = torch.zeros((nsteps,), device=self._device)
        entropies = torch.zeros((nsteps,), device=self._device)
        t_actions = torch.zeros((nsteps,), device=self._device)
        t_logprobs = torch.zeros((nsteps,), device=self._device)
        all_mu = [] 
        all_p_tau = {}
        all_p_tau2 = {}
        all_rew = [] 
        all_rew2 = {} 
        ret = []
        if self._is_nets:
            n_nets = self._discriminator.get_num_nets()
            vs = np.arange(-0.3, 0.31, 0.01)
            grids = np.meshgrid(*[vs] * n_nets)
            combinations = np.vstack([grid.ravel() for grid in grids]).T
            for rate in combinations:
                rate_str = f'{rate}'
                all_p_tau[rate_str] = []
                all_p_tau2[rate_str] = []
                all_rew2[rate_str] = []

        size = self._size
        step = 0
        while step!=nsteps:
            time_step = env.reset()
            rew = 0
            while not time_step.last():
                obs = time_step.observations["info_state"][self._player_id]
                obs_pth = torch.Tensor(obs).to(self._device)
                info_state[step] = obs_pth
                with torch.no_grad():
                    t_action, t_logprob, _, _ = self._iter_agent.get_action_and_value(obs_pth)
                    action, logprob, entropy, value = self._eps_agent.get_action_and_value(obs_pth)

                time_step = env.step([action.item()])

                obs_list = obs[:-1]
                obs_x = obs_list[:size].index(1)
                obs_y = obs_list[size:2*size].index(1)
                obs_t = obs_list[2*size:].index(1)
                mus = [self._mu_dist[n][obs_t, obs_y, obs_x] for n in range(num_agent)]
                all_mu.append(mus[self._player_id])
                obs_mu = np.array(obs_list+mus)

                idx = self._player_id
                acs = onehot(action, self._nacs).reshape(1, self._nacs)
                if self._net_input=='s_mu_a':
                    x, y, t, mu = divide_obs([obs_mu], self._size, use_argmax=False)
                    state = np.concatenate([x, y], axis=1)

                    inputs = [torch.from_numpy(state), 
                                torch.from_numpy(mu), 
                                torch.from_numpy(acs)]
                elif self._net_input=='sa_mu':
                    x, y, t, mu = divide_obs([obs_mu], self._size, use_argmax=False)
                    state_a = np.concatenate([x, y, acs], axis=1)

                    inputs = [torch.from_numpy(state_a), 
                                torch.from_numpy(mu)]
                elif self._net_input=='s_mua':
                    x, y, t, mu = divide_obs([obs_mu], self._size, use_argmax=False)
                    state = np.concatenate([x, y], axis=1)
                    mua = np.concatenate([mu, acs], axis=1)

                    inputs = [torch.from_numpy(state), 
                                torch.from_numpy(mua)]
                elif self._net_input=='dxy_mu_a':
                    x, y, t, mu = divide_obs([obs_mu], self._size, use_argmax=True)
                    dx, dy = goal_distance(x, y, idx)
                    dxy = np.concatenate([dx, dy], axis=1)

                    inputs = [torch.from_numpy(dxy), 
                                torch.from_numpy(mu),
                                torch.from_numpy(acs),]
                elif self._net_input=='dxya_mu':
                    x, y, t, mu = divide_obs([obs_mu], self._size, use_argmax=True)
                    dx, dy = goal_distance(x, y, idx)
                    dxy_a = np.concatenate([dx, dy, acs], axis=1)

                    inputs = [torch.from_numpy(dxy_a),
                                torch.from_numpy(mu),]
                elif self._net_input=='dxy_mua':
                    x, y, t, mu = divide_obs([obs_mu], self._size, use_argmax=True)
                    dx, dy = goal_distance(x, y, idx)
                    dxy = np.concatenate([dx, dy], axis=1)
                    mua = np.concatenate([mu, acs], axis=1)

                    inputs = [torch.from_numpy(dxy),
                                torch.from_numpy(mua),]

                if self._is_nets:
                    x, y, t, mu = divide_obs([obs_mu], self._size, use_argmax=False)
                    obs_xym = np.concatenate([x, y, mu], axis=1)
                    nobs = obs_xym.copy()
                    nobs[:-1] = obs_xym[1:]
                    nobs[-1] = obs_xym[0]
                    obs_next_xym = nobs
                    reward, outputs = self._discriminator.get_reward(
                        inputs,
                        torch.from_numpy(obs_xym).to(self._device),
                        torch.from_numpy(obs_next_xym).to(self._device),
                        None,
                        discrim_score=False,
                        weighted_rew=True) # For competitive tasks, log(D) - log(1-D) empirically works better (discrim_score=True)
                    for rate in combinations:
                        rew, rew2, p_tau, p_tau2 = self._discriminator.get_reward_weighted(
                            inputs,
                            torch.from_numpy(obs_xym).to(self._device),
                            torch.from_numpy(obs_next_xym).to(self._device),
                            None,
                            rate=rate) # For competitive tasks, log(D) - log(1-D) empirically works better (discrim_score=True)
                        rate_str = f'{rate}'
                        all_p_tau[rate_str].append(p_tau[0])
                        all_p_tau2[rate_str].append(p_tau2[0])
                        all_rew2[rate_str].append(rew2[0])
                    all_rew.append(rew)
                else:
                    reward = discriminator.get_reward(
                        torch.from_numpy(obs_mu).to(torch.float32),
                        torch.from_numpy(onehot(action, nacs)).to(torch.int64),
                        None, None,
                        discrim_score=False,
                        ) # For competitive tasks, log(D) - log(1-D) empirically works better (discrim_score=True)

                true_rewards[step] = torch.Tensor([time_step.rewards[self._player_id]]).to(self._device)
                # iteration policy data
                t_logprobs[step] = t_logprob
                t_actions[step] = t_action

                # episode policy data
                logprobs[step] = logprob
                dones[step] = time_step.last()
                entropies[step] = entropy
                values[step] = value
                actions[step] = action
                #rewards[step] = reward
                if self._rew_index>=0:
                    rewards[step] = outputs[self._rew_index] 
                else:
                    rewards[step] = reward
                    
                rew += time_step.rewards[self._player_id]

                #print(f'xyt: {obs_x},{obs_y},{obs_t}')
                #print(f'mu{self._player_id}: {[self._mu_dist[idx][obs_t, obs_y, obs_x] for idx in range(3)]}')
                #print(f'rew: {time_step.rewards}')
                step += 1
                if step==nsteps:
                    break
            ret.append(rew)
        ret = np.array(ret)
        assert step==nsteps

        cos_sims = {} 
        spearmanrs = {}
        kl_divs = {}
        #euclids = {}
        cos_sims_rews = {} 
        spearmanrs_rews = {}
        kl_divs_rews = {}
        rew = np.array(all_rew).flatten()
        for rate_str in all_p_tau.keys():
            p_tau = np.array(all_p_tau[rate_str])
            p_tau2 = np.array(all_p_tau2[rate_str])
            rew2 = np.array(all_rew2[rate_str])

            cos_sim = 1-distance.cosine(p_tau, p_tau2)
            sp, p_value = spearmanr(p_tau, p_tau2)
            kl_div = np.sum([ai * np.log(ai / bi) for ai, bi in zip(p_tau, p_tau2)]) 

            cos_sim_rew = 1-distance.cosine(rew, rew2)
            sp_rew, p_value = spearmanr(rew, rew2)
            kl_div_rew = np.sum([ai * np.log(ai / bi) for ai, bi in zip(rew, rew2)]) 
            #euclid = np.sqrt(np.sum((p_tau-p_tau2)**2))

            cos_sims[rate_str] = cos_sim
            spearmanrs[rate_str] = sp 
            kl_divs[rate_str] = kl_div 

            cos_sims_rews[rate_str] = cos_sim_rew
            spearmanrs_rews[rate_str] = sp_rew
            kl_divs_rews[rate_str] = kl_div_rew

            print(f'----------------------')
            print(f'rate: {rate_str}')
            print(f'log p tau: {np.mean(p_tau)}')
            print(f'log p tau2: {np.mean(p_tau2)}')
            print(f'cos_sim(p,p2): {np.mean(cos_sim)}')
            print(f'spearmanr(p,p2): {np.mean(sp)}')
            print(f'kl_div(p,p2): {np.mean(kl_div)}')
            #print(f'euclid(p,p2): {np.mean(euclid)}')
        pkl.dump(p_tau, open('p_tau_sampled.pkl', 'wb'))
        pkl.dump(p_tau2, open('p_tau2_sampled.pkl', 'wb'))
        pkl.dump(cos_sims, open('cos_sims_sampled.pkl', 'wb'))
        pkl.dump(spearmanrs, open('spearmanrs_sampled.pkl', 'wb'))
        pkl.dump(kl_divs, open('kl_div_sampled.pkl', 'wb'))

        pkl.dump(rew, open('rew_sampled.pkl', 'wb'))
        pkl.dump(rew2, open('rew2_sampled.pkl', 'wb'))
        pkl.dump(cos_sims_rews, open('rew_cos_sims_sampled.pkl', 'wb'))
        pkl.dump(spearmanrs_rews, open('rew_spearmanrs_sampled.pkl', 'wb'))
        pkl.dump(kl_divs_rews, open('rew_kl_div_sampled.pkl', 'wb'))
        print(f'dumped sampled state pkl')



        return info_state, actions, logprobs, rewards, true_rewards, dones, values, entropies,t_actions,t_logprobs, all_mu, ret


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


    def update_eps(self, obs, logprobs, actions, advantages, returns, t_actions, t_logprobs, 
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
                _, newlogprob, entropy, new_value = self._eps_agent.get_action_and_value(b_obs, actions[mb_inds])
                logratio = newlogprob - logprobs[mb_inds]
                ratio = torch.exp(logratio)

                # Get the data under the iteration policy (the population policy)
                _, t_newlogprob, _, _ = self._eps_agent.get_action_and_value(b_obs, t_actions[mb_inds])
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
                self._optimizer_actor.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._eps_agent.actor.parameters(), max_grad_norm)
                self._optimizer_actor.step()
                
                # Critic update 
                self._optimizer_critic.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self._eps_agent.critic.parameters(), max_grad_norm)
                self._optimizer_critic.step()

        return v_loss

    def update_iter(self, game, env, merge_dist, conv_dist, nashc=False, population=0):
        # iter agent の更新
        self._iter_agent.load_state_dict(self._eps_agent.state_dict())

        self._mu_dist = conv_dist
        env.update_mfg_distribution(merge_dist)

        nashc_ppo = None
        root_state = game.new_initial_state_for_population(population)
        if nashc:
            pi_value = policy_value.PolicyValue(game, merge_dist, self._ppo_policy, value.TabularValueFunction(game))
            nashc_ppo = NashC(game, merge_dist, pi_value, root_state=root_state).nash_conv()
            #nashc_ppo = NashC(game, merge_dist, pi_value).nash_conv()
        return nashc_ppo

    def calc_nashc(self, game, merge_dist, use_expert_policy=False, population=0):

        root_state = game.new_initial_state_for_population(population)
        if use_expert_policy:
            pi_value = policy_value.PolicyValue(game, merge_dist, self._expert_policy, value.TabularValueFunction(game))
        else:
            pi_value = policy_value.PolicyValue(game, merge_dist, self._ppo_policy, value.TabularValueFunction(game))
        nashc_ppo = NashC(game, merge_dist, pi_value, root_state=root_state).nash_conv()

        return nashc_ppo

    def log_metrics(self, it, distrib, policy, writer, reward, entropy):
        # this function is used to log the results to tensor board
        initial_states = game.new_initial_states()
        pi_value = policy_value.PolicyValue(game, distrib, policy, value.TabularValueFunction(game))
        m = {
            f"ppo_br/{state}": pi_value.eval_state(state)
            for state in initial_states
        }
        m["nash_conv_ppo"] = NashC(game, distrib, pi_value).nash_conv()
        writer.add_scalar("initial_state_value", m['ppo_br/initial'], it)
        # debug
        writer.add_scalar("rewards", reward, it)
        writer.add_scalar("entorpy", entropy, it)

        writer.add_scalar("nash_conv_ppo", m['nash_conv_ppo'], it)
        logger.debug(f"ppo_br: {m['ppo_br/initial']}, and nash_conv: {m['nash_conv_ppo']}, reward: {reward}, entropy: {entropy}")
        print(f"ppo_br: {m['ppo_br/initial']}, and nash_conv: {m['nash_conv_ppo']}, reward: {reward}, entropy: {entropy}")
        return m["nash_conv_ppo"]

    def get_value(self, obs):
        with torch.no_grad():
            value = self._eps_agent.get_value(obs)
        return value

    def get_action(self, obs):
        with torch.no_grad():
            action, prob = self._eps_agent.get_action(obs)
        return action, prob

    def get_log_action_prob(self, states, actions):
        with torch.no_grad():
            logpac = self._eps_agent.get_log_action_prob(states, actions)
        return logpac

    def save(self, game, filename=""):
        fname = osp.join(logger.get_dir(), 'actor'+filename+".pth")
        torch.save(self._eps_agent.actor.state_dict(), fname)

        fname = osp.join(logger.get_dir(), 'critic'+filename+".pth")
        torch.save(self._eps_agent.critic.state_dict(), fname)

        distrib = distribution.DistributionPolicy(game, self._ppo_policy)
        fname = osp.join(logger.get_dir(), 'distrib'+filename+".pth")
        utils.save_parametric_distribution(distrib, fname)   
        print(f'Saved generator param (actor, critic, distrib -{filename})')

    def load(self):
        return None
        

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="set a random seed")
    parser.add_argument("--game-setting", type=str, default="crowd_modelling_2d_four_rooms", help="Set the game to benchmark options:(crowd_modelling_2d_four_rooms) and (crowd_modelling_2d_maze)")
    
    parser.add_argument("--batch_step", type=int, default=200, help="set the number of episodes of to collect per rollout")
    parser.add_argument("--num_episodes", type=int, default=20, help="set the number of episodes of the inner loop")
    parser.add_argument("--num_iterations", type=int, default=100, help="Set the number of global update steps of the outer loop")
    
    parser.add_argument("--path", type=str, default="/mnt/shunsuke/result/0627/multi_maze2_dxy_mua", help="file path")
    parser.add_argument('--logdir', type=str, default="/mnt/shunsuke/result/0627/multi_maze2_ppo_dxy_mua_diversity", help="logdir")
    parser.add_argument("--rew_index", type=int, default=-1, help="-1 is reward, 0 or more are output")
    parser.add_argument("--update_eps", type=str, default=r"200_2", help="file path")

    parser.add_argument("--single", action='store_true')
    parser.add_argument("--notmu", action='store_true')

    parser.add_argument("--reward_filename", type=str, default="disc_reward", help="file path")
    parser.add_argument("--value_filename", type=str, default="disc_value", help="file path")
    parser.add_argument("--actor_filename", type=str, default="actor", help="file path")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    single = args.single
    notmu = args.notmu

    update_eps_info = f'{args.update_eps}'
    logger.configure(args.logdir, format_strs=['stdout', 'log', 'json'])

    from open_spiel.python.mfg.algorithms.discriminator_networks import * 
    is_nets = is_networks(args.path)
    print(f'Is networks: {is_nets}')
    if not is_nets:
        from open_spiel.python.mfg.algorithms.discriminator import Discriminator
        rew_index = -1
    else:
        net_input = get_net_input(args.path)
        label = get_net_label(net_input)
        assert len(label)>=args.rew_index, 'rew_index is wrong'
        rew_index = args.rew_index

    # Set the seed 
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

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
        envs[-1].seed(args.seed)
    
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
            labels = get_net_label(net_input)
            discriminator = Discriminator(inputs, obs_xym_size, labels, device)
        else:
            discriminator = Discriminator(nobs+num_agent, nacs, False, device)
        reward_path = osp.join(args.path, args.reward_filename+update_eps_info + f'-{i}.pth')
        value_path = osp.join(args.path, args.value_filename+update_eps_info + f'-{i}.pth')
        if is_nets:
            discriminator.load(args.path, f'{update_eps_info}-{i}', use_eval=True)
            discriminator.print_weights()
        else:
            distance_path = osp.join(args.path, args.distance_filename+update_eps_info + f'-{i}.pth')
            mu_path = osp.join(args.path, args.mu_filename+update_eps_info + f'-{i}.pth')
            discriminator.load(reward_path, value_path, use_eval=True)
            print(f'')
        discriminators.append(discriminator)

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
        

