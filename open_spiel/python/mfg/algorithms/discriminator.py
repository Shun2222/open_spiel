import numpy as np

import torch
from torch import nn
from torch import optim
from torch.distributions.categorical import Categorical

import joblib
from rl.acktr.utils import Scheduler, find_trainable_variables
from rl.acktr.utils import fc, mse
from rl.acktr import kfac
from irl.mack.tf_util import relu_layer, linear, tanh_layer

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
  torch.nn.init.orthogonal_(layer.weight, std)
  torch.nn.init.constant_(layer.bias, bias_const)
  return layer

disc_types = ['decentralized', 'centralized', 'single', 'decentralized-all']


class Discriminator(object):
    def __init__(self, ob_space, ac_space, state_only, discount,
                 nstack, index, disc_type='decentralized', hidden_size=128,
                 lr_rate=0.01, total_steps=50000, scope="discriminator", kfac_clip=0.001, max_grad_norm=0.5,
                 l2_loss_ratio=0.01):
        self.lr = lr_rate
        self.disc_type = disc_type
        self.l2_loss_ratio = l2_loss_ratio
        if disc_type not in disc_types:
            assert False
        self.state_only = state_only
        self.gamma = discount
        self.scope = scope
        self.ob_shape = ob_space.shape[0] * nstack
        self.all_ob_shape = sum([obs.shape[0] for obs in ob_spaces]) * nstack
        nact = ac_space.n
        self.ac_shape = nact * nstack
        self.all_ob_shape = sum([obs.shape[0] for obs in ob_spaces]) * nstack

        if disc_type == 'decentralized':
            self.obs = tf.placeholder(tf.float32, (None, self.ob_shape))
            self.nobs = tf.placeholder(tf.float32, (None, self.ob_shape))
            self.act = tf.placeholder(tf.float32, (None, self.ac_shape))
            self.labels = tf.placeholder(tf.float32, (None, 1))
            self.lprobs = tf.placeholder(tf.float32, (None, 1))

        self.lr_rate = 

        rew_input = obs_space
        self.reward = nn.Sequential(
            layer_init(nn.Linear(np.array(rew_input).prod(), 1)),
            nn.RelU()
        )

        self.value_n = nn.Sequential(
            layer_init(nn.Linear(np.array(self.nobs).prod(), 1)),
            nn.RelU()
        )

        self.value = nn.Sequential(
            layer_init(nn.Linear(np.array(self.obs).prod(), 1)),
            nn.RelU()
        )

        optimizer = optim.Adam(self.???.parameters(), lr=args.lr,eps=1e-5)

    def update(self, max_grad_norm=5):
        optimizer.zero_grad()

        log_q_tau = self.lprobs
        log_p_tau = self.reward + self.gamma * self.value_n - self.value
        log_pq = tf.reduce_logsumexp([log_p_tau, log_q_tau], axis=0)
        self.discrim_output = tf.exp(log_p_tau - log_pq)

        self.total_loss = -tf.reduce_mean(self.labels * (log_p_tau - log_pq) + (1 - self.labels) * (log_q_tau - log_pq))
        self.var_list = self.get_trainable_variables()
        params = find_trainable_variables(self.scope)
        self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in params]) * self.l2_loss_ratio
        self.total_loss += self.l2_loss

        grads = tf.gradients(self.total_loss, params)
        with tf.variable_scope(self.scope + '/d_optim'):
            d_optim = tf.train.AdamOptimizer(learning_rate=self.lr_rate)
            train_op = d_optim.apply_gradients(list(zip(grads, params)))
        self.d_optim = train_op
        self.saver = tf.train.Saver(self.get_variables())

        self.params_flat = self.get_trainable_variables()

        loss.backward()
        nn.utils.clip_grad_norm_(self.???.parameters(), max_grad_norm)
        optimizer.step()


    def get_reward(self, obs):
        return self.reward(obs)

    def get_value_next(self, nobs):
        return self.value_n(nobs)

    def get_value_next(self, obs):
        return self.value(obs)

    with torch.no_grad():

    def train(self, g_obs, g_acs, g_nobs, g_probs, e_obs, e_acs, e_nobs, e_probs):
        labels = np.concatenate((np.zeros([g_obs.shape[0], 1]), np.ones([e_obs.shape[0], 1])), axis=0)
        feed_dict = {self.obs: np.concatenate([g_obs, e_obs], axis=0),
                     self.act: np.concatenate([g_acs, e_acs], axis=0),
                     self.nobs: np.concatenate([g_nobs, e_nobs], axis=0),
                     self.lprobs: np.concatenate([g_probs, e_probs], axis=0),
                     self.labels: labels,
                     self.lr_rate: self.lr.value()}
        loss, _ = self.sess.run([self.total_loss, self.d_optim], feed_dict)
        return loss

    def restore(self, path):
        print('restoring from:' + path)
        self.saver.restore(self.sess, path)

    def save(self, save_path):
        ps = self.sess.run(self.params_flat)
        joblib.dump(ps, save_path)

    def load(self, load_path):
        loaded_params = joblib.load(load_path)
        restores = []
        for p, loaded_p in zip(self.params_flat, loaded_params):
            restores.append(p.assign(loaded_p))
        self.sess.run(restores)
