from open_spiel.python.mfg.algorithms.discriminator import Discriminator
from open_spiel.python.mfg.examples.mfg_Proximal_policy_optimization_pytorch import MFGPPO

import os.path as osp
import random
import time

import joblib
import numpy as np
import tensorflow as tf
from scipy.stats import pearsonr, spearmanr
from dataset import Dset

from open_spiel.python.mfg.algorithms.mfg_ppo import MFGPPO
from open_spiel.python.mfg.algorithms.discriminator import Discriminator


class AIRL(object):
    def __init__(self, game, env, expert):
        self._game = game
        self._env = env
        self._generator = MFGPPO()
        self._discriminator = Discriminator()
        self._expert = expert

    def run(self, total_step):

        t_step = 0
        while(total_step < t_step):
            obs, actions, logprobs, true_rewards, dones, values, entropies, t_actions, t_logprobs 
                = self._generator.rollout(env, batch_step)

            disc_rewards = np.squeeze(self.discriminator.get_reward(re_obs[k],
                                                               multionehot(re_actions[k], self.n_actions[k]),
                                                               re_obs_next[k],
                                                               re_path_prob[k],
                                                               discrim_score=False)) # For competitive tasks, log(D) - log(1-D) empirically works better (discrim_score=True)


            if buffer:
                buffer.update(mh_obs, mh_actions, mh_obs_next, all_obs, mh_values)
            else:
                buffer = Dset(mh_obs, mh_actions, mh_obs_next, all_obs, mh_values, 
                              randomize=True, num_agents=1, nobs_flag=True)

            e_obs, e_actions, e_nobs, e_all_obs, _ = expert.get_next_batch(batch_step)
            g_obs, g_actions, g_nobs, g_all_obs, _ = buffer.get_next_batch(batch_size=batch_step)

            e_a = [np.argmax(e_actions[k], axis=1) for k in range(len(e_actions))]
            g_a = [np.argmax(g_actions[k], axis=1) for k in range(len(g_actions))]

            g_log_prob = generator.get_log_action_prob(g_obs, g_a)
            e_log_prob = generator.get_log_action_prob(e_obs, e_a)

            k = 0 # num agent = 0
            total_loss[d_iter] = self._discriminator.train(
                g_obs[k],
                g_actions[k],
                g_nobs[k],
                g_log_prob[k].reshape([-1, 1]),
                e_obs[k],
                e_actions[k],
                e_nobs[k],
                e_log_prob[k].reshape([-1, 1])
            )

            if t_iter > update_eps_generator_until:
                #store rewards and entropy for debugging
                adv, returns = cal_Adv(disc_rewards, values, dones)
                # Update the learned policy and report loss for debugging
                v_loss = self._genertorupdate_eps(obs, logprobs, actions, adv, disc_rewards, t_actions, t_logprobs) 

                if t_iter % update_iter_generator_interval == 0 :
                    self._generator.update_iter(self._game, self._env)

                if t_iter % log_interval == 0:
                    #collect and print the metrics
                    rew = true_rewards.sum().item()/args.num_episodes

                    print("------------MFG PPO------------")
                    print("Value_loss", v_loss.item())
                    print('true reward', rew)    
                    print("-------------------------------")


            t_step += batch_step 
