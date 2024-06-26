import os
import os.path as osp
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

class MFGPPO(object):
    def __init__(self, game, env, device):
        self._device = device

        info_state_size = env.observation_spec()["info_state"][0]
        num_actions = env.action_spec()["num_actions"]

        self._eps_agent = Agent(info_state_size,num_actions).to(self._device) 
        self._ppo_policy = PPOpolicy(game, self._eps_agent, None, self._device) 
        self._iter_agent = Agent(info_state_size,num_actions).to(self._device)
        self._optimizer_actor = optim.Adam(self._eps_agent.actor.parameters(), lr=1e-3,eps=1e-5)
        self._optimizer_critic = optim.Adam(self._eps_agent.critic.parameters(), lr=1e-3,eps=1e-5)

        distrib = distribution.DistributionPolicy(game, self._ppo_policy)

        self._size = env.game.get_parameters()['size']
        horizon = env.game.get_parameters()['horizon']
        mu_dist = np.zeros((horizon, self._size, self._size))
        for k,v in distrib.distribution.items():
            if "mu" in k:
                tt = k.split("_")[0].split(",")
                x = int(tt[0].split("(")[-1])
                y = int(tt[1].split()[-1])
                t = int(tt[2].split()[-1].split(")")[0])
                mu_dist[t,y,x] = v
        self._mu_dist = mu_dist

    def rollout(self, env, nsteps):
        info_state = torch.zeros((nsteps,self._iter_agent.info_state_size), device=self._device)
        actions = torch.zeros((nsteps,), device=self._device)
        logprobs = torch.zeros((nsteps,), device=self._device)
        rewards = torch.zeros((nsteps,), device=self._device)
        dones = torch.zeros((nsteps,), device=self._device)
        values = torch.zeros((nsteps,), device=self._device)
        entropies = torch.zeros((nsteps,), device=self._device)
        t_actions = torch.zeros((nsteps,), device=self._device)
        t_logprobs = torch.zeros((nsteps,), device=self._device)
        obs_mu = torch.zeros((nsteps,self._iter_agent.info_state_size+1), device=self._device)
        ret = []

        size = self._size
        step = 0
        while step!=nsteps-1:
            time_step = env.reset()
            rew = 0
            while not time_step.last():
                obs = time_step.observations["info_state"][0]
                obs_pth = torch.Tensor(obs).to(self._device)
                info_state[step] = obs_pth
                with torch.no_grad():
                    t_action, t_logprob, _, _ = self._iter_agent.get_action_and_value(obs_pth)
                    action, logprob, entropy, value = self._eps_agent.get_action_and_value(obs_pth)

                time_step = env.step([action.item()])

                obs_x = obs[:size].index(1)
                obs_y = obs[size:2*size].index(1)
                obs_t = obs[2*size:].index(1)
                mu = obs.copy()
                mu.append(self._mu_dist[obs_t, obs_y, obs_x])
                obs_mu[step] = torch.Tensor(mu).to(self._device)

                # iteration policy data
                t_logprobs[step] = t_logprob
                t_actions[step] = t_action

                # episode policy data
                logprobs[step] = logprob
                dones[step] = time_step.last()
                entropies[step] = entropy
                values[step] = value
                actions[step] = action
                rewards[step] = torch.Tensor(time_step.rewards).to(self._device)
                rew += time_step.rewards[0]
                step += 1
                if step==nsteps-1:
                    break
            ret.append(rew)
        ret = np.array(ret)
        assert step==nsteps-1
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

    def update_iter(self, game, env, nashc=False):
        # iter agent の更新
        self._iter_agent.load_state_dict(self._eps_agent.state_dict())

        # mf policyの更新
        distrib = distribution.DistributionPolicy(game, self._ppo_policy)
        env.update_mfg_distribution(distrib)

        horizon = env.game.get_parameters()['horizon']
        mu_dist = np.zeros((horizon, self._size, self._size))
        for k,v in distrib.distribution.items():
            if "mu" in k:
                tt = k.split("_")[0].split(",")
                x = int(tt[0].split("(")[-1])
                y = int(tt[1].split()[-1])
                t = int(tt[2].split()[-1].split(")")[0])
                mu_dist[t,y,x] = v
        self._mu_dist = mu_dist

        nashc_ppo = None
        if nashc:
            pi_value = policy_value.PolicyValue(game, distrib, self._ppo_policy, value.TabularValueFunction(game))
            nashc_ppo = NashC(game, distrib, pi_value).nash_conv()
        return nashc_ppo

    def calc_nashc(self, game, env):
        # mf policyの更新
        distrib = distribution.DistributionPolicy(game, self._ppo_policy)
        env.update_mfg_distribution(distrib)

        pi_value = policy_value.PolicyValue(game, distrib, self._ppo_policy, value.TabularValueFunction(game))
        nashc_ppo = NashC(game, distrib, pi_value).nash_conv()

        return nashc_ppo


    def plot_dist(self, env, game_name, distrib, info_state, save=False, filename="agent_dist.mp4"):
        # this functions is used to generate an animated video of the distribuiton propagating throught the game 
        horizon = env.game.get_parameters()['horizon']
        size = env.game.get_parameters()['size']
        if game_name == "maze":
            d_size = 21
        else:
            d_size = 13
        agent_dist = np.zeros((horizon,d_size,d_size))
        mu_dist = np.zeros((horizon,d_size,d_size))


        for k,v in distrib.distribution.items():
            if "mu" in k:
                tt = k.split("_")[0].split(",")
                x = int(tt[0].split("(")[-1])
                y = int(tt[1].split()[-1])
                t = int(tt[2].split()[-1].split(")")[0])
                mu_dist[t,y,x] = v

        for i in range(horizon):
            obs = info_state[i].tolist()
            obs_x = obs[:size].index(1)
            obs_y = obs[size:2*size].index(1)
            obs_t = obs[2*size:].index(1)
            agent_dist[obs_t,obs_y,obs_x] = 0.02

        final_dist = agent_dist + mu_dist

        if save:
            fig = plt.figure(figsize=(8,8))
            plt.axis("off")
            ims = [[plt.imshow(img, animated=True)] for img in final_dist]
            ani = animation.ArtistAnimation(fig, ims, blit=True, interval = 200)

            ani.save(filename, fps=5)

            plt.close()

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

    # Create the game instance 
    game = factory.create_game_with_setting("mfg_crowd_modelling_2d", args.game_setting)

    # Set the initial policy to uniform and generate the distribution 
    uniform_policy = policy_std.UniformRandomPolicy(game)
    mfg_dist = distribution.DistributionPolicy(game, uniform_policy)
    env = rl_environment.Environment(game, mfg_distribution=mfg_dist, mfg_population=0)

    # Set the environment seed for reproduciblility 
    env.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    mfgppo = MFGPPO(game, env, device)

    batch_step = args.batch_step
    for _ in tqdm(range(args.num_iterations)):
        for _ in range(args.num_episodes):
            obs_pth, actions_pth, logprobs_pth, true_rewards_pth, dones_pth, values_pth, entropies_pth, t_actions_pth, t_logprobs_pth, mu, ret \
                = mfgppo.rollout(env, args.batch_step)
            adv_pth, returns = mfgppo.cal_Adv(true_rewards_pth, values_pth, dones_pth)
            v_loss = mfgppo.update_eps(obs_pth, logprobs_pth, actions_pth, adv_pth, returns, t_actions_pth, t_logprobs_pth) 

        nashc_ppo = mfgppo.update_iter(game, env, nashc=True)
        ret = np.array(ret)
        print("Value_loss", v_loss.item())
        print('reward', ret)    
        print('Mean reward', np.mean(ret))    
        print('NashC ppo', nashc_ppo)    
