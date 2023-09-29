import os.path as osp
import random
import time

#import joblib
import numpy as np
import torch
import logger
from dataset import Dset
from utils import onehot, multionehot
from scipy.stats import pearsonr, spearmanr

import torch.optim as optim
from open_spiel.python.mfg.algorithms.mfg_ppo import MFGPPO
from open_spiel.python.mfg.algorithms.discriminator import Discriminator


class AIRL(object):
    def __init__(self, game, env, device, expert):
        self._game = game
        self._env = env
        self._device = device

        self._expert = expert
        self._nacs = env.action_spec()['num_actions']
        self._nobs = env.observation_spec()['info_state'][0]

        self._generator = MFGPPO(game, env, device)
        self._discriminator = Discriminator(self._nobs, self._nacs, True, device)
        self._optimizer = optim.Adam(self._discriminator.parameters(), lr=0.01)


    def run(self, total_step, total_step_gen, num_episodes, log_interval_rate=0.1):
        log_interval = total_step * log_interval_rate

        t_step = 0
        batch_step = num_episodes * self._env.max_game_length
        buffer = None
        while(t_step < total_step):
            for neps in range(num_episodes): 
                obs_pth, actions_pth, logprobs_pth, true_rewards_pth, dones_pth, values_pth, entropies_pth, t_actions_pth, t_logprobs_pth, ret \
                    = self._generator.rollout(self._env, batch_step)

                obs = obs_pth.to(self._device).detach().numpy()
                nobs = obs.copy()
                nobs[:-1] = obs[1:]
                nobs[-1] = obs[0]
                obs_next = nobs
                obs_next_pth = torch.from_numpy(obs_next)
                actions = actions_pth.to(self._device).detach().numpy()
                logprobs = logprobs_pth.to(self._device).detach().numpy()
                true_rewards = true_rewards_pth.to(self._device).detach().numpy()
                dones = dones_pth.to(self._device).detach().numpy()
                values = values_pth.to(self._device).detach().numpy()
                entropies = entropies_pth.to(self._device).detach().numpy()
                t_actions = t_actions_pth.to(self._device).detach().numpy()
                t_logprobs = t_logprobs_pth.to(self._device).detach().numpy()

                disc_rewards_pth = self._discriminator.get_reward( obs_pth,
                                                                   torch.from_numpy(multionehot(actions, self._nacs)),
                                                                   obs_next_pth,
                                                                   logprobs_pth,
                                                                   discrim_score=False) # For competitive tasks, log(D) - log(1-D) empirically works better (discrim_score=True)

                disc_rewards = disc_rewards_pth.detach().numpy().reshape(batch_step)
                disc_rewards_pth = torch.from_numpy(disc_rewards)

                mh_obs = [np.array(obs)]
                mh_actions = [np.array(multionehot(actions, self._nacs))]
                mh_obs_next = [np.array(obs_next)]
                all_obs = np.array(obs)
                mh_values = [np.array(values)]

                if buffer:
                    buffer.update(mh_obs, mh_actions, mh_obs_next, all_obs, mh_values)
                else:
                    buffer = Dset(mh_obs, mh_actions, mh_obs_next, all_obs, mh_values, 
                                  randomize=True, num_agents=1, nobs_flag=True)

                e_obs, e_actions, e_nobs, e_all_obs, _ = self._expert.get_next_batch(batch_step)
                g_obs, g_actions, g_nobs, g_all_obs, _ = buffer.get_next_batch(batch_step)

                e_a = [np.argmax(e_actions[k], axis=1) for k in range(len(e_actions))]
                g_a = [np.argmax(g_actions[k], axis=1) for k in range(len(g_actions))]

                e_log_prob = [] 
                g_log_prob = [] 
                # e_log_prob.append(self._generator.get_log_action_prob(torch.from_numpy(e_obs[0]).to(torch.float32), torch.from_numpy(e_a[0]).to(torch.int64)))
                for i in range(len(e_obs[0])):
                    e_log_prob.append(self._generator.get_log_action_prob(
                        torch.from_numpy(np.array([e_obs[0][i]])).to(torch.float32), 
                        torch.from_numpy(np.array([e_a[0][i]])).to(torch.int64)))
                    g_log_prob.append(self._generator.get_log_action_prob(
                        torch.from_numpy(np.array([g_obs[0][i]])).to(torch.float32), 
                        torch.from_numpy(np.array([g_a[0][i]])).to(torch.int64)))
                e_log_prob = np.array([e_log_prob])
                g_log_prob = np.array([g_log_prob])
            
                d_obs = np.concatenate([g_obs[0], e_obs[0]], axis=0)
                d_acs = np.concatenate([g_actions[0], e_actions[0]], axis=0)
                d_nobs = np.concatenate([g_nobs[0], e_nobs[0]], axis=0)
                d_lprobs = np.concatenate([g_log_prob.reshape([-1, 1]), e_log_prob.reshape([-1, 1])], axis=0)
                d_labels = np.concatenate([np.zeros([g_obs[0].shape[0], 1]), np.ones([e_obs[0].shape[0], 1])], axis=0)

                total_loss = self._discriminator.train(
                    self._optimizer,
                    torch.from_numpy(d_obs).to(torch.float32),
                    torch.from_numpy(d_acs).to(torch.int64),
                    torch.from_numpy(d_nobs).to(torch.float32),
                    torch.from_numpy(d_lprobs).to(torch.float32),
                    torch.from_numpy(d_labels).to(torch.int64),
                )

                if t_step < total_step_gen:
                    adv_pth, returns = self._generator.cal_Adv(disc_rewards_pth, values_pth, dones_pth)
                    v_loss = self._generator.update_eps(obs_pth, logprobs_pth, actions_pth, adv_pth, returns, t_actions_pth, t_logprobs_pth) 


                pear = ""
                spear = ""
                try:
                    pear = pearsonr(disc_rewards.flatten(), true_rewards.flatten())[0]
                    spear = spearson(disc_rewards.flatten(), true_rewards.flatten())[0]
                except:
                    pass
                logger.record_tabular("timestep", t_step)
                logger.record_tabular("ppo_loss", v_loss.item())
                logger.record_tabular("mean_ret", np.mean(ret))
                logger.record_tabular("pearsonr", pear)
                logger.record_tabular("spearson", spear)
                logger.dump_tabular()

                t_step += batch_step 

            if t_step < total_step_gen:
                self._generator.update_iter(self._game, self._env)
        self._generator.save()
        self._discriminator.save()
