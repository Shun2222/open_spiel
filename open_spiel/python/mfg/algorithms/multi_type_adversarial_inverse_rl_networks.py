import os.path as osp
import random
import time

#import joblib
import numpy as np
import torch
import logger
from dataset import Dset
from open_spiel.python.mfg.algorithms import distribution
from utils import onehot, multionehot
from scipy.stats import pearsonr, spearmanr

import torch.optim as optim
from open_spiel.python.mfg.algorithms.multi_type_mfg_ppo import MultiTypeMFGPPO, convert_distrib
from open_spiel.python.mfg.algorithms.discriminator_networks import Discriminator
from games.predator_prey import goal_distance, divide_obs


class MultiTypeAIRL(object):
    def __init__(self, game, envs, merge_dist, conv_dist, device, experts, ppo_policies, disc_type='s_mu_a'):
        self._game = game
        self._envs = envs
        self._device = device
        self._num_agent = len(envs)
        self._size = game.get_parameters()['size']
        self._disc_type = disc_type

        env = envs[0]
        self._horizon = env.game.get_parameters()['horizon']
        self._experts = experts
        self._nacs = env.action_spec()['num_actions']
        self._nobs = env.observation_spec()['info_state'][0]
        self._nmu  = self._num_agent 

        self._generator = [MultiTypeMFGPPO(game, envs[i], merge_dist, conv_dist, device, player_id=i, expert_policy=ppo_policies[i]) for i in range(self._num_agent)]
        state_size = self._nobs -1 - self._horizon # nobs-1: obs size (exposed own mu), nmu: all agent mu size, horizon: horizon size
        obs_xym_size = self._nobs -1 - self._horizon + self._nmu # nobs-1: obs size (exposed own mu), nmu: all agent mu size, horizon: horizon size
        if disc_type=='s_mu_a':
            inputs = [state_size, self._nmu, self._nacs]
            labels = ['state', 'mu', 'act']
        elif disc_type=='sa_mu':
            inputs = [state_size+self._nacs, self._nmu]
            labels = ['state_a', 'mu']
        elif disc_type=='s_mua':
            inputs = [state_size, self._nmu+self._nacs]
            labels = ['state', 'mu_a']
        elif disc_type=='dxy_mu_a':
            inputs = [2, self._nmu, self._nacs]
            labels = ['dxy', 'mu', 'act']
        elif disc_type=='dxya_mu':
            inputs = [2+self._nacs, self._nmu]
            labels = ['dxy_a', 'mu']
        elif disc_type=='dxy_mua':
            inputs = [2, self._nmu+self._nacs]
            labels = ['dxy', 'mu_a']
        else:
            assert False, f'not matched disc type: {disc_type}'

        self._discriminator = [Discriminator(inputs, obs_xym_size, labels, device) for _ in range(self._num_agent)]
        self._optimizers = [optim.Adam(self._discriminator[i].parameters(), lr=0.01) for i in range(self._num_agent)]


    def run(self, total_step, total_step_gen, num_episodes, batch_step, save_interval=1000):
        logger.record_tabular("total_step", total_step)
        logger.record_tabular("total_step_gen(=total_step)", total_step_gen)
        logger.record_tabular("num_episodes", num_episodes)
        logger.record_tabular("batch_step", batch_step)
        logger.dump_tabular()

        t_step = 0
        num_update_eps = 0
        num_update_iter = 0
        batch_size = batch_step//self._envs[0].max_game_length
        batch_step = batch_size * self._envs[0].max_game_length

        buffer = [None for _ in range(self._num_agent)]
        while(t_step < total_step):
            for neps in range(num_episodes): 
                rollouts = []
                mus = []
                for i in range(self._num_agent):
                    obs_pth, actions_pth, logprobs_pth, true_rewards_pth, dones_pth, values_pth, entropies_pth, t_actions_pth, t_logprobs_pth, mu_pth, ret \
                        = self._generator[i].rollout(self._envs[i], batch_step)
                    rollouts.append([obs_pth, actions_pth, logprobs_pth, true_rewards_pth, dones_pth, values_pth, entropies_pth, t_actions_pth, t_logprobs_pth, mu_pth, ret])
                    mus.append(mu_pth)
                merge_mu = []
                for step in range(len(mus[0])):
                    merge_mu.append([mus[i][step] for i in range(self._num_agent)])

                logger.record_tabular(f"timestep", t_step)
                for idx, rout in enumerate(rollouts):
                    obs_pth, actions_pth, logprobs_pth, true_rewards_pth, dones_pth, values_pth, entropies_pth, t_actions_pth, t_logprobs_pth, mu_pth, ret = rout     
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

                    obs_mu = []
                    for step in range(batch_step):
                        obs_list = list(obs[step][:-1])
                        obs_mu.append(obs_list + list(merge_mu[step]))
                    obs_mu = np.array(obs_mu)

                    nobs = obs_mu.copy()
                    nobs[:-1] = obs_mu[1:]
                    nobs[-1] = obs_mu[0]
                    obs_next_mu = nobs
                    obs_next_mu_pth = torch.from_numpy(obs_next_mu).to(self._device)

                    if self._disc_type=='s_mu_a':
                        acs = multionehot(actions, self._nacs)
                        x, y, t, mu = divide_obs(obs_mu, self._size, use_argmax=False)
                        state = np.concatenate([x, y], axis=1)

                        inputs = [torch.from_numpy(state), 
                                  torch.from_numpy(mu), 
                                  torch.from_numpy(acs)]
                    elif self._disc_type=='sa_mu':
                        acs = multionehot(actions, self._nacs)
                        x, y, t, mu = divide_obs(obs_mu, self._size, use_argmax=False)
                        state_a = np.concatenate([x, y, acs], axis=1)

                        inputs = [torch.from_numpy(state_a), 
                                  torch.from_numpy(mu)]
                    elif self._disc_type=='s_mua':
                        acs = multionehot(actions, self._nacs)
                        x, y, t, mu = divide_obs(obs_mu, self._size, use_argmax=False)
                        state = np.concatenate([x, y], axis=1)
                        mua = np.concatenate([mu, acs], axis=1)

                        inputs = [torch.from_numpy(state), 
                                  torch.from_numpy(mua)]
                    elif self._disc_type=='dxy_mu_a':
                        acs = multionehot(actions, self._nacs)
                        x, y, t, mu = divide_obs(obs_mu, self._size, use_argmax=True)
                        dx, dy = goal_distance(x, y, idx)
                        dxy = np.concatenate([dx, dy], axis=1)

                        inputs = [torch.from_numpy(dxy), 
                                  torch.from_numpy(mu),
                                  torch.from_numpy(acs),]
                    elif self._disc_type=='dxya_mu':
                        acs = multionehot(actions, self._nacs)
                        x, y, t, mu = divide_obs(obs_mu, self._size, use_argmax=True)
                        dx, dy = goal_distance(x, y, idx)
                        dxy_a = np.concatenate([dx, dy, acs], axis=1)

                        inputs = [torch.from_numpy(dxy_a),
                                  torch.from_numpy(mu),]
                    elif self._disc_type=='dxy_mua':
                        acs = multionehot(actions, self._nacs)
                        x, y, t, mu = divide_obs(obs_mu, self._size, use_argmax=True)
                        dx, dy = goal_distance(x, y, idx)
                        dxy = np.concatenate([dx, dy], axis=1)
                        mua = np.concatenate([mu, acs], axis=1)

                        inputs = [torch.from_numpy(dxy),
                                  torch.from_numpy(mua),]


                    x, y, t, mu = divide_obs(obs_mu, self._size, use_argmax=False)
                    obs_xym = np.concatenate([x, y, mu], axis=1)

                    nobs = obs_xym.copy()
                    nobs[:-1] = obs_xym[1:]
                    nobs[-1] = obs_xym[0]
                    obs_next_xym = nobs

                    disc_rewards_pth = self._discriminator[idx].get_reward(
                        inputs, 
                        torch.from_numpy(obs_xym).to(self._device),
                        torch.from_numpy(obs_next_xym).to(self._device),
                        torch.from_numpy(logprobs).to(self._device),
                        discrim_score=False) # For competitive tasks, log(D) - log(1-D) empirically works better (discrim_score=True)

                    disc_rewards = disc_rewards_pth.cpu().detach().numpy().reshape(batch_step)
                    disc_rewards_pth = torch.from_numpy(disc_rewards).to(self._device)

                    adv_pth, returns = self._generator[idx].cal_Adv(disc_rewards_pth, values_pth, dones_pth)
                    v_loss = self._generator[idx].update_eps(obs_pth, logprobs_pth, actions_pth, adv_pth, returns, t_actions_pth, t_logprobs_pth)

                    mh_obs = [np.array(obs)]
                    mh_actions = [np.array(multionehot(actions, self._nacs))]
                    mh_obs_next = [np.array(obs_next)]
                    all_obs = np.array(obs)
                    mh_values = [np.array(values)]
                    mh_obs_mu = [np.array(obs_mu)]
                    mh_obs_next_mu = [np.array(obs_next_mu)]

                    if buffer[idx]:
                        buffer[idx].update(mh_obs_mu, mh_actions, mh_obs_next_mu, all_obs, mh_values)
                    else:
                        buffer[idx] = Dset(mh_obs_mu, mh_actions, mh_obs_next_mu, all_obs, mh_values, 
                                      randomize=True, num_agents=1, nobs_flag=True)

                    e_obs_mu, e_actions, e_nobs, e_all_obs, _ = self._experts[idx].get_next_batch(batch_step)
                    g_obs_mu, g_actions, g_nobs, g_all_obs, _ = buffer[idx].get_next_batch(batch_step)

                    e_a = [np.argmax(e_actions[k], axis=1) for k in range(len(e_actions))]
                    g_a = [np.argmax(g_actions[k], axis=1) for k in range(len(g_actions))]

                    e_log_prob = [] 
                    g_log_prob = [] 
                    for i in range(len(e_obs_mu[0])):
                        e_log_prob.append(self._generator[idx].get_log_action_prob(
                            torch.from_numpy(np.array([e_obs_mu[0][i][:self._nobs]])).to(torch.float32).to(self._device), 
                            torch.from_numpy(np.array([e_a[0][i]])).to(torch.int64).to(self._device)).cpu().detach().numpy())

                        g_log_prob.append(self._generator[idx].get_log_action_prob(
                            torch.from_numpy(np.array([g_obs_mu[0][i][:self._nobs]])).to(torch.float32).to(self._device), 
                            torch.from_numpy(np.array([g_a[0][i]])).to(torch.int64).to(self._device)).cpu().detach().numpy())

                    e_log_prob = np.array([e_log_prob])
                    g_log_prob = np.array([g_log_prob])

                    g_x, g_y, g_t, g_mu = divide_obs(g_obs_mu[0], self._size, use_argmax=False)
                    g_state = np.concatenate([g_x, g_y], axis=1)
                    g_nx, g_ny, g_nt, g_nmu = divide_obs(g_nobs[0], self._size, use_argmax=False)
                    g_mx, g_my, _, _ = divide_obs(g_obs_mu[0], self._size, use_argmax=True)
                    g_dx, g_dy = goal_distance(g_mx, g_my, idx)
                    g_dxy = np.concatenate([g_dx, g_dy], axis=1)

                    e_x, e_y, e_t, e_mu = divide_obs(e_obs_mu[0], self._size, use_argmax=False)
                    e_state = np.concatenate([e_x, e_y], axis=1)
                    e_nx, e_ny, e_nt, e_nmu = divide_obs(e_nobs[0], self._size, use_argmax=False)
                    e_mx, e_my, _, _ = divide_obs(e_obs_mu[0], self._size, use_argmax=True)
                    e_dx, e_dy = goal_distance(e_mx, e_my, idx)
                    e_dxy = np.concatenate([e_dx, e_dy], axis=1)

                    if self._disc_type=='s_mu_a':
                        d_state = np.concatenate([g_state, e_state], axis=0)
                        d_mu = np.concatenate([g_mu, e_mu], axis=0)
                        d_acs = np.concatenate([g_actions[0], e_actions[0]], axis=0)

                        inputs = [torch.from_numpy(d_state), 
                                  torch.from_numpy(d_mu), 
                                  torch.from_numpy(d_acs)]
                    elif self._disc_type=='sa_mu':
                        g_state_a = np.concatenate([g_state, g_actions[0]], axis=1)
                        e_state_a = np.concatenate([e_state, e_actions[0]], axis=1)

                        d_state_a = np.concatenate([g_state_a, e_state_a], axis=0)
                        d_mu = np.concatenate([g_mu, e_mu], axis=0)

                        inputs = [torch.from_numpy(d_state_a), 
                                  torch.from_numpy(d_mu)]
                    elif self._disc_type=='s_mua':
                        g_mua = np.concatenate([g_mu, g_actions[0]], axis=1)
                        e_mua = np.concatenate([e_mu, e_actions[0]], axis=1)

                        d_state = np.concatenate([g_state, e_state], axis=0)
                        d_mua = np.concatenate([g_mua, e_mua], axis=0)

                        inputs = [torch.from_numpy(d_state), 
                                  torch.from_numpy(d_mua)]
                    elif self._disc_type=='dxy_mu_a':
                        d_dxy = np.concatenate([g_dxy, e_dxy], axis=0)
                        d_mua = np.concatenate([g_mu, e_mu], axis=0)
                        d_acs = np.concatenate([g_actions[0], e_actions[0]], axis=0)

                        inputs = [torch.from_numpy(d_dxy), 
                                  torch.from_numpy(d_mu),
                                  torch.from_numpy(d_acs),]
                    elif self._disc_type=='dxya_mu':
                        g_dxya = np.concatenate([g_dxy, g_actions[0]], axis=1)
                        e_dxya = np.concatenate([e_dxy, e_actions[0]], axis=1)

                        d_dxya = np.concatenate([g_dxya, e_dxya], axis=0)
                        d_mu = np.concatenate([g_mu, e_mu], axis=0)

                        inputs = [torch.from_numpy(d_dxya), 
                                  torch.from_numpy(d_mu),]
                    elif self._disc_type=='dxy_mua':
                        g_mua = np.concatenate([g_mu, g_actions[0]], axis=1)
                        e_mua = np.concatenate([e_mu, e_actions[0]], axis=1)

                        d_dxy = np.concatenate([g_dxy, e_dxy], axis=0)
                        d_mua = np.concatenate([g_mua, e_mua], axis=0)

                        inputs = [torch.from_numpy(d_dxy), 
                                  torch.from_numpy(d_mua),]
                                  

                    g_obs_xym = np.concatenate([g_x, g_y, g_mu], axis=1)
                    g_nobs_xym = np.concatenate([g_nx, g_ny, g_nmu], axis=1)
                    e_obs_xym = np.concatenate([e_x, e_y, e_mu], axis=1)
                    e_nobs_xym = np.concatenate([e_nx, e_ny, e_nmu], axis=1)
                    d_obs_xym = np.concatenate([g_obs_xym, e_obs_xym], axis=0)
                    d_nobs_xym = np.concatenate([g_nobs_xym, e_nobs_xym], axis=0)
                    #d_nobs = np.concatenate([np.array(g_nobs[0])[:, :self._nobs], np.array(e_nobs[0])[:, :self._nobs]], axis=0)
                    d_lprobs = np.concatenate([g_log_prob.reshape([-1, 1]), e_log_prob.reshape([-1, 1])], axis=0)
                    d_labels = np.concatenate([np.zeros([g_obs_xym.shape[0], 1]), np.ones([e_obs_xym.shape[0], 1])], axis=0)

                    total_loss = self._discriminator[idx].train(
                        inputs, 
                        self._optimizers[idx],
                        torch.from_numpy(d_obs_xym).to(torch.float32).to(self._device),
                        torch.from_numpy(d_nobs_xym).to(torch.float32).to(self._device),
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

                    logger.record_tabular(f"generator_loss{idx}", v_loss.item())
                    logger.record_tabular(f"discriminator_loss{idx}", total_loss)
                    logger.record_tabular(f"mean_ret{idx}", np.mean(ret))
                    logger.record_tabular(f"pearsonr{idx}", pear)
                    logger.record_tabular(f"spearson{idx}", spear)
                logger.dump_tabular()

                t_step += batch_step 
                num_update_eps += 1
                if(num_update_eps%save_interval==0):
                    for i in range(self._num_agent):
                        fname = f"{num_update_eps}_{num_update_iter}-{i}"
                        self._generator[i].save(self._game, filename=fname)
                        self._discriminator[i].save(filename=fname)


            #if t_step < total_step_gen:
            mfg_dists = []
            for i in range(self._num_agent):
                policy = self._generator[i]._ppo_policy
                mfg_dist = distribution.DistributionPolicy(self._game, policy)
                mfg_dists.append(mfg_dist)
            
            merge_dist = distribution.MergeDistribution(self._game, mfg_dists)
            conv_dist = convert_distrib(self._envs, merge_dist)
            for i in range(self._num_agent):
                nashc_ppo = self._generator[i].update_iter(self._game, self._envs[i], merge_dist, conv_dist, nashc=True, population=i)
                logger.record_tabular(f"nashc_ppo{i}", nashc_ppo)
                nashc_expert = self._generator[i].calc_nashc(self._game, merge_dist, use_expert_policy=True, population=i)
                logger.record_tabular(f"nashc_expert{i}", nashc_expert)
            logger.dump_tabular()
            num_update_iter += 1

        for i in range(self._num_agent):
            fname = f"{num_update_eps}_{num_update_iter}-{i}"
            self._generator[i].save(self._game, filename=fname)
            self._discriminator[i].save(filename=fname)
