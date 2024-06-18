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
        self._nmu  = 1

        self._generator = MFGPPO(game, env, device)
        self._discriminator = Discriminator(self._nobs+self._nmu, self._nacs, False, device)
        self._optimizer = optim.Adam(self._discriminator.parameters(), lr=0.01)


    def run(self, total_step, total_step_gen, num_episodes, batch_step, save_interval=1000):
        logger.record_tabular("total_step", total_step)
        logger.record_tabular("total_step_gen(=total_step)", total_step_gen)
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

                mh_obs = [np.array(obs)]
                mh_actions = [np.array(multionehot(actions, self._nacs))]
                mh_obs_next = [np.array(obs_next)]
                all_obs = np.array(obs)
                mh_values = [np.array(values)]
                mh_obs_mu = [np.array(obs_mu)]
                mh_obs_next_mu = [np.array(obs_next_mu)]

                if buffer:
                    buffer.update(mh_obs_mu, mh_actions, mh_obs_next_mu, all_obs, mh_values)
                else:
                    buffer = Dset(mh_obs_mu, mh_actions, mh_obs_next_mu, all_obs, mh_values, 
                                  randomize=True, num_agents=1, nobs_flag=True)

                e_obs_mu, e_actions, e_nobs, e_all_obs, _ = self._expert.get_next_batch(batch_step)
                g_obs_mu, g_actions, g_nobs, g_all_obs, _ = buffer.get_next_batch(batch_step)

                e_a = [np.argmax(e_actions[k], axis=1) for k in range(len(e_actions))]
                g_a = [np.argmax(g_actions[k], axis=1) for k in range(len(g_actions))]

                e_log_prob = [] 
                g_log_prob = [] 
                # e_log_prob.append(self._generator.get_log_action_prob(torch.from_numpy(e_obs_mu[0]).to(torch.float32), torch.from_numpy(e_a[0]).to(torch.int64)))
                for i in range(len(e_obs_mu[0])):
                    e_log_prob.append(self._generator.get_log_action_prob(
                        torch.from_numpy(np.array([e_obs_mu[0][i][:self._nobs]])).to(torch.float32).to(self._device), 
                        torch.from_numpy(np.array([e_a[0][i]])).to(torch.int64).to(self._device)).cpu().detach().numpy())

                    g_log_prob.append(self._generator.get_log_action_prob(
                        torch.from_numpy(np.array([g_obs_mu[0][i][:self._nobs]])).to(torch.float32).to(self._device), 
                        torch.from_numpy(np.array([g_a[0][i]])).to(torch.int64).to(self._device)).cpu().detach().numpy())

                e_log_prob = np.array([e_log_prob])
                g_log_prob = np.array([g_log_prob])
            
                d_obs_mu = np.concatenate([g_obs_mu[0], e_obs_mu[0]], axis=0)
                d_acs = np.concatenate([g_actions[0], e_actions[0]], axis=0)
                #d_nobs = np.concatenate([np.array(g_nobs[0])[:, :self._nobs], np.array(e_nobs[0])[:, :self._nobs]], axis=0)
                d_nobs = np.concatenate([np.array(g_nobs[0])[:, :self._nobs+1], np.array(e_nobs[0])[:, :self._nobs+1]], axis=0)
                d_lprobs = np.concatenate([g_log_prob.reshape([-1, 1]), e_log_prob.reshape([-1, 1])], axis=0)
                d_labels = np.concatenate([np.zeros([g_obs_mu[0].shape[0], 1]), np.ones([e_obs_mu[0].shape[0], 1])], axis=0)

                total_loss = self._discriminator.train(
                    self._optimizer,
                    torch.from_numpy(d_obs_mu).to(torch.float32).to(self._device),
                    torch.from_numpy(d_acs).to(torch.int64).to(self._device),
                    torch.from_numpy(d_nobs).to(torch.float32).to(self._device),
                    torch.from_numpy(d_lprobs).to(torch.float32).to(self._device),
                    torch.from_numpy(d_labels).to(torch.int64).to(self._device),
                )



                pear = ""
                spear = ""
                try:
                    pear = pearsonr(disc_rewards.flatten(), true_rewards.flatten())[0]
                    spear = spearmanr(disc_rewards.flatten(), true_rewards.flatten())[0]
                except:
                    pass
                logger.record_tabular("timestep", t_step)
                logger.record_tabular("generator_loss", v_loss.item())
                logger.record_tabular("discriminator_loss", total_loss)
                logger.record_tabular("mean_ret", np.mean(ret))
                logger.record_tabular("pearsonr", pear)
                logger.record_tabular("spearson", spear)
                logger.dump_tabular()

                t_step += batch_step 
                num_update_eps += 1
                if(num_update_eps%save_interval==0):
                    fname = f"{num_update_eps}_{num_update_iter}"
                    self._generator.save(self._game, filename=fname)
                    self._discriminator.save(filename=fname)


            #if t_step < total_step_gen:
            nashc = self._generator.update_iter(self._game, self._env, nashc=True)
            logger.record_tabular("nashc", nashc)
            logger.dump_tabular()
            num_update_iter += 1
        self._generator.save(self._game)
        self._discriminator.save()
