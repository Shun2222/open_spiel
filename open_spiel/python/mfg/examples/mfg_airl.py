class Model(object):
    def __init__(self, policy, env
                 nprocs=2, nsteps=200,
                 nstack=1, ent_coef=0.00, vf_coef=0.5, 
                 vf_fisher_coef=1.0, lr=0.25, max_grad_norm=0.5,
                 kfac_clip=0.001, lrschedule='linear', device="cpu",
                 identical=None):

        nbatch = nenvs * nsteps
        self.num_agents = 
        self.num_actions = 
        self.device = device

        A, ADV, R, PG_LR = [], [], [], []
        A = torch.zeros((nbatch,), device=self.device)
        ADV = torch.zeros((nbatch,), device=self.device)
        R = torch.zeros((nbatch,), device=self.device)
        PG_LR = torch.zeros((nbatch,), device=self.device)

        #pg_loss, entropy, vf_loss, train_loss = [], [], [], []
        self.model = step_model = policy
        self.model2 = train_model = policy
        self.log_pac = []

        logpac = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=train_model.pi, labels=A)

        def train(obs, states, rewards, masks, actions, values):
            mfgppo.update()
            return policy_loss, value_loss, policy_entropy

        def clone(obs, actions):
            return lld_loss

        def get_log_action_prob(obs, actions):
            return action_prob

        self.get_log_action_prob = get_log_action_prob

        def get_log_action_prob_step(obs, actions):
            return action_prob

        self.get_log_action_prob_step = get_log_action_prob_step

        def save(save_path):

        def load(load_path):

        self.train = train
        self.clone = clone
        self.save = save
        self.load = load
        self.train_model = train_model
        self.step_model = step_model

        def step(ob, av, *_args, **_kwargs):
            return a, v, s

        self.step = step

        def value(obs, av):
            return v

        self.value = value
        self.initial_state = [step_model[k].initial_state for k in range(num_agents)]

        def rollout():
            return mfgppo.rollout()


class Runner(object):
    def __init__(self, env, model, discriminator, nsteps, nstack, gamma, lam, disc_type, nobs_flag=False):
        self.env = env
        self.model = model
        self.discriminator = discriminator
        self.disc_type = disc_type
        self.nobs_flag = nobs_flag
        self.num_agents = len(env.observation_space)
        self.nenv = nenv = env.num_envs
        self.batch_ob_shape = 
        self.obs = 
        self.actions =

        obs = 
        self.update_obs(obs)
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.states = 
        self.num_actions = 
        self.dones =

    def update_obs(self, obs):
        self.obs = obs

    def run(self):
        mb_obs = []
        mb_obs_next = []
        mb_true_rewards = []
        mb_rewards = []
        mb_report_rewards = []
        mb_actions = []
        mb_values = []
        mb_dones = []
        mb_masks = []
        mb_states = self.states
        for n in range(self.nsteps):
            actions, values, states = self.model.step(self.obs, self.actions)

            self.actions = actions
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            actions_list = []
            obs, true_rewards, dones, _ = self.env.step(actions_list)

            for k in range(self.num_agents):
                for ni, done in enumerate(dones[k]):
                    if done:
                        obs[k][ni] = obs[k][ni] * 0.0
            for k in range(self.num_agents):
                mb_obs_next[k].append(np.copy(obs[k]))

            re_obs = self.obs
            re_actions = self.actions
            re_obs_next = obs
            re_path_prob = self.model.get_log_action_prob_step(re_obs, re_actions)  # [num_agent, nenv, 1]
            re_actions_onehot = [multionehot(re_actions[k], self.n_actions[k]) for k in range(self.num_agents)]

            # get reward from discriminator
            if self.disc_type == 'decentralized':
                rewards = []
                report_rewards = []
                for k in range(self.num_agents):
                    rewards.append(np.squeeze(self.discriminator[k].get_reward(re_obs[k],
                                                                               multionehot(re_actions[k], self.n_actions[k]),
                                                                               re_obs_next[k],
                                                                               re_path_prob[k],
                                                                               discrim_score=False))) # For competitive tasks, log(D) - log(1-D) empirically works better (discrim_score=True)
                    report_rewards.append(np.squeeze(self.discriminator[k].get_reward(re_obs[k],
                                                                               multionehot(re_actions[k], self.n_actions[k]),
                                                                               re_obs_next[k],
                                                                               re_path_prob[k],
                                                                               discrim_score=False)))

            mb_rewards.append(rewards)
            mb_report_rewards.append(report_rewards)

            self.states = states
            self.dones = dones
            self.update_obs(obs)

            mb_true_rewards.append(true_rewards)
        mb_dones.append(self.dones)

        # batch of steps to batch of rollouts
        for k in range(self.num_agents):
            mb_obs[k] = np.asarray(mb_obs[k], dtype=np.float32).swapaxes(1, 0).reshape(self.batch_ob_shape[k])
            mb_obs_next[k] = np.asarray(mb_obs_next[k], dtype=np.float32).swapaxes(1, 0).reshape(self.batch_ob_shape[k])
            mb_true_rewards[k] = np.asarray(mb_true_rewards[k], dtype=np.float32).swapaxes(1, 0)
            mb_rewards[k] = np.asarray(mb_rewards[k], dtype=np.float32).swapaxes(1, 0)
            mb_report_rewards[k] = np.asarray(mb_report_rewards[k], dtype=np.float32).swapaxes(1, 0)
            mb_actions[k] = np.asarray(mb_actions[k], dtype=np.int32).swapaxes(1, 0)
            mb_values[k] = np.asarray(mb_values[k], dtype=np.float32).swapaxes(1, 0)
            mb_dones[k] = np.asarray(mb_dones[k], dtype=np.bool).swapaxes(1, 0)
            mb_masks[k] = mb_dones[k][:, :-1]
            mb_dones[k] = mb_dones[k][:, 1:]

        mb_returns = [np.zeros_like(mb_rewards[k]) for k in range(self.num_agents)]
        mb_report_returns = [np.zeros_like(mb_rewards[k]) for k in range(self.num_agents)]
        mb_true_returns = [np.zeros_like(mb_rewards[k]) for k in range(self.num_agents)]
        last_values = self.model.value(self.obs, self.actions)
        # discount/bootstrap off value fn
        for k in range(self.num_agents):
            for n, (rewards, report_rewards, true_rewards, dones, value) in enumerate(zip(mb_rewards[k], mb_report_rewards[k], mb_true_rewards[k], mb_dones[k], last_values[k].tolist())):
                rewards = rewards.tolist()
                report_rewards = report_rewards.tolist()
                dones = dones.tolist()
                true_rewards = true_rewards.tolist()
                if dones[-1] == 0:
                    rewards = discount_with_dones(rewards + [value], dones + [0], self.gamma)[:-1]
                    report_rewards = discount_with_dones(report_rewards + [value], dones + [0], self.gamma)[:-1]
                    true_rewards = discount_with_dones(true_rewards + [value], dones + [0], self.gamma)[:-1]
                else:
                    rewards = discount_with_dones(rewards, dones, self.gamma)
                    report_rewards = discount_with_dones(report_rewards, dones, self.gamma)
                    true_rewards = discount_with_dones(true_rewards, dones, self.gamma)
                mb_returns[k][n] = rewards
                mb_report_returns[k][n] = report_rewards
                mb_true_returns[k][n] = true_rewards

        for k in range(self.num_agents):
            mb_returns[k] = mb_returns[k].flatten()
            mb_report_returns[k] = mb_report_returns[k].flatten()
            mb_masks[k] = mb_masks[k].flatten()
            mb_values[k] = mb_values[k].flatten()
            mb_actions[k] = mb_actions[k].flatten()

        mh_actions = [multionehot(mb_actions[k], self.n_actions[k]) for k in range(self.num_agents)]
        mb_all_obs = np.concatenate(mb_obs, axis=1)
        mb_all_nobs = np.concatenate(mb_obs_next, axis=1)
        mh_all_actions = np.concatenate(mh_actions, axis=1)
        if self.nobs_flag:
            return mb_obs, mb_obs_next, mb_states, mb_returns, mb_report_returns, mb_masks, mb_actions, \
                   mb_values, mb_all_obs, mb_all_nobs, mh_actions, mh_all_actions, mb_rewards, mb_true_rewards, mb_true_returns
        else:
            return mb_obs, mb_states, mb_returns, mb_report_returns, mb_masks, mb_actions,\
                   mb_values, mb_all_obs, mh_actions, mh_all_actions, mb_rewards, mb_true_rewards, mb_true_returns


def learn(policy, expert, env, env_id, seed, total_timesteps=int(40e6), gamma=0.99, lam=0.95, log_interval=1, nprocs=32,
          nsteps=20, nstack=1, ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.25, max_grad_norm=0.5,
          kfac_clip=0.001, save_interval=100, lrschedule='linear', dis_lr=0.001, disc_type='decentralized',
          bc_iters=500, identical=None, l2=0.1, d_iters=1, rew_scale=0.1):
    tf.reset_default_graph()
    set_global_seeds(seed)
    buffer = None

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    num_agents = (len(ob_space))
    make_model = lambda: Model(policy, ob_space, ac_space, nenvs, total_timesteps, nprocs=nprocs, nsteps=nsteps,
                               nstack=nstack, ent_coef=ent_coef, vf_coef=vf_coef, vf_fisher_coef=vf_fisher_coef,
                               lr=lr, max_grad_norm=max_grad_norm, kfac_clip=kfac_clip,
                               lrschedule=lrschedule, identical=identical)
    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))
    model = make_model()
    if disc_type == 'decentralized' or disc_type == 'decentralized-all':
        discriminator = [
            Discriminator(model.sess, ob_space, ac_space,
                          state_only=True, discount=gamma, nstack=nstack, index=k, disc_type=disc_type,
                          scope="Discriminator_%d" % k, # gp_coef=gp_coef,
