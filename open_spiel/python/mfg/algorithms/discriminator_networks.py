import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logger
import os
import os.path as osp
import copy
import matplotlib.pyplot as plt
import re

def onehot(value, depth):
    a = np.zeros([depth])
    a[value] = 1
    return a


def multionehot(values, depth):
    a = np.zeros([values.shape[0], depth])
    for i in range(values.shape[0]):
        a[i, int(values[i])] = 1
    return a

def get_num_hidden(pathname):
    res = re.search(r'hidden(\d+)', pathname)
    if res:
        return int(res.group(1))
    else:
        return 1

def net_labels(net_input):
    if net_input=='s_mu_a':
        labels = ['state', 'mu', 'act']
    elif net_input=='sa_mu':
        labels = ['state_a', 'mu']
    elif net_input=='s_mua':
        labels = ['state', 'mu_a']
    elif net_input=='dxy_mu_a':
        labels = ['dxy', 'mu', 'act']
    elif net_input=='dxya_mu':
        labels = ['dxy_a', 'mu']
    elif net_input=='dxy_mua':
        labels = ['dxy', 'mu_a']
    elif net_input=='s_mu':
        labels = ['state', 'mu']
    elif net_input=='dxy_mu':
        labels = ['dxy', 'mu']
    else:
        assert False, f'not matched disc type: {net_input}'
    return labels

def get_net_inputs():
    net_inputs = ['s_mu_a',
                  'sa_mu',
                  's_mua',
                  'dxy_mu_a',
                  'dxya_mu',
                  'dxy_mua',
                  's_mu',
                  'dxy_mu',
                  ]
    return net_inputs

def get_input_shape(net_input, env, num_agent):
    nacs = env.action_spec()['num_actions']
    nobs = env.observation_spec()['info_state'][0]
    horizon = env.game.get_parameters()['horizon']

    nmu = num_agent
    size = env.game.get_parameters()['size']
    state_size = nobs -1 - horizon # nobs-1: obs size (exposed own mu), nmu: all agent mu size, horizon: horizon size
    obs_xym_size = nobs -1 - horizon + nmu # nobs-1: obs size (exposed own mu), nmu: all agent mu size, horizon: horizon size
    discriminators = []
    if net_input=='s_mu_a':
        inputs = [state_size, nmu, nacs]
    elif net_input=='sa_mu':
        inputs = [state_size+nacs, nmu]
    elif net_input=='s_mua':
        inputs = [state_size, nmu+nacs]
    elif net_input=='dxy_mu_a':
        inputs = [2, nmu, nacs]
    elif net_input=='dxya_mu':
        inputs = [2+nacs, nmu]
    elif net_input=='dxy_mua':
        inputs = [2, nmu+nacs]
    elif net_input=='s_mu':
        inputs = [state_size, nmu]
    elif net_input=='dxy_mu':
        #inputs = [state_size*2-1, nmu]
        inputs = [2, nmu]
    else:
        assert False, f'not matched disc type: {net_input}'
    return inputs

def create_disc_input(size, net_input, obs_mu, onehot_acs, player_id):
    # assert 

    from games.predator_prey import divide_obs, goal_distance
    acs = onehot_acs
    idx = player_id

    if net_input=='s_mu_a':
        x, y, t, mu = divide_obs(obs_mu, size, use_argmax=False)
        state = np.concatenate([x, y], axis=1)

        inputs = [torch.from_numpy(state), 
                    torch.from_numpy(mu), 
                    torch.from_numpy(acs)]
    elif net_input=='sa_mu':
        x, y, t, mu = divide_obs(obs_mu, size, use_argmax=False)
        state_a = np.concatenate([x, y, acs], axis=1)

        inputs = [torch.from_numpy(state_a), 
                    torch.from_numpy(mu)]
    elif net_input=='s_mua':
        x, y, t, mu = divide_obs(obs_mu, size, use_argmax=False)
        state = np.concatenate([x, y], axis=1)
        mua = np.concatenate([mu, acs], axis=1)

        inputs = [torch.from_numpy(state), 
                    torch.from_numpy(mua)]
    elif net_input=='dxy_mu_a':
        x, y, t, mu = divide_obs(obs_mu, size, use_argmax=True)
        dx, dy = goal_distance(x, y, idx)
        dxy = np.concatenate([dx, dy], axis=1)

        inputs = [torch.from_numpy(dxy), 
                    torch.from_numpy(mu),
                    torch.from_numpy(acs),]
    elif net_input=='dxya_mu':
        x, y, t, mu = divide_obs(obs_mu, size, use_argmax=True)
        dx, dy = goal_distance(x, y, idx)
        dxy_a = np.concatenate([dx, dy, acs], axis=1)

        inputs = [torch.from_numpy(dxy_a),
                    torch.from_numpy(mu),]
    elif net_input=='dxy_mua':
        x, y, t, mu = divide_obs(obs_mu, size, use_argmax=True)
        dx, dy = goal_distance(x, y, idx)
        dxy = np.concatenate([dx, dy], axis=1)
        mua = np.concatenate([mu, acs], axis=1)

        inputs = [torch.from_numpy(dxy),
                    torch.from_numpy(mua),]
    elif net_input=='s_mu':
        x, y, t, mu = divide_obs(obs_mu, size, use_argmax=False)
        state = np.concatenate([x, y], axis=1)

        inputs = [torch.from_numpy(state), 
                    torch.from_numpy(mu)]
    elif net_input=='dxy_mu':
        x, y, t, mu = divide_obs(obs_mu, size, use_argmax=True)
        dx, dy = goal_distance(x, y, idx)
        dxy = np.concatenate([dx, dy], axis=1)
        dxy_abs = np.abs(dxy)

        dx[dx<0] = np.abs(dx[dx<0])+size
        dy[dy<0] = np.abs(dy[dy<0])+size
        dx_onehot = multionehot(dx, size*2)
        dy_onehot = multionehot(dy, size*2)
        dxy_onehot = np.concatenate([dx_onehot, dy_onehot], axis=1)

        inputs = [torch.from_numpy(dxy),
                    torch.from_numpy(mu),]


    x, y, t, mu = divide_obs(obs_mu, size, use_argmax=False)
    obs_xym = np.concatenate([x, y, mu], axis=1)

    nobs = obs_xym.copy()
    nobs[:-1] = obs_xym[1:]
    nobs[-1] = obs_xym[0]
    obs_next_xym = nobs
    return inputs, obs_xym, obs_next_xym

def is_networks(filename):
    res = get_net_input(filename)
    if res!=None:
        return True
    else:
        return False

def get_net_labels(net_input):
    return net_labels(net_input)

def get_net_input(filename):
    net_inputs = get_net_inputs()
    detected_input = []
    for net_input in net_inputs:
        if net_input in filename:
            detected_input.append(net_input)
    if len(detected_input)>0:
        num = [len(d) for d in detected_input]
        idx = np.argmax(num)
        print(f'Detected model as {detected_input[idx]}')
        return detected_input[idx]
    else:
        return None

class Discriminator(nn.Module):
    def __init__(self, input_shapes, obs_shape, labels, device, discount=0.99, hidden_size=128, l2_loss_ratio=0.01, num_hidden=1, ppo_value_net=None):
        super(Discriminator, self).__init__()
        assert len(input_shapes)<=len(labels), f'not enough labels'

        self.gamma = discount
        self.hidden_size = hidden_size
        self.l2_loss_ratio = l2_loss_ratio
        self._device = device
        self.labels = labels

        self.n_networks = len(input_shapes)
        self.networks = []
        for i in range(self.n_networks):
            # Define layers for reward network
            if num_hidden==1:
                net = nn.Sequential(
                    nn.Linear(input_shapes[i], hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, 1)
                ).to(self._device)
            elif num_hidden==2:
                net = nn.Sequential(
                    nn.Linear(input_shapes[i], hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, 1)
                ).to(self._device)
            elif num_hidden==3:
                net = nn.Sequential(
                    nn.Linear(input_shapes[i], hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, 1)
                ).to(self._device)
            self.networks.append(net)


        self.reward_net = nn.Sequential(
            nn.Linear(self.n_networks, 1),
        ).to(self._device)

        # Define layers for value function network
        if ppo_value_net:
            self.value_net = ppo_value_net
            self._ppo_value = True
        else:
            self.value_net = nn.Sequential(
                nn.Linear(obs_shape, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            ).to(self._device)
            self._ppo_value = False

        self.value_next_net = self.value_net


        self.l2_loss = nn.MSELoss()

    def forward(self, inputs, obs, obs_next, path_probs):
        #rew_input = obs if self.state_only else torch.cat([obs, acs], dim=1)
        outputs = [self.networks[i](inputs[i].to(torch.float32)) for i in range(self.n_networks)] 
        outputs = torch.cat(outputs, dim=1)
        reward = self.reward_net(outputs.to(torch.float32))

        if self._ppo_value:
            with torch.no_grad():
                value_fn = self.value_net(obs)
                value_fn_next = self.value_next_net(obs_next)
        else:
            value_fn = self.value_net(obs)
            value_fn_next = self.value_next_net(obs_next)


        log_q_tau = path_probs
        log_p_tau = reward + self.gamma * value_fn_next - value_fn
        log_pq = torch.logsumexp(torch.stack([log_p_tau, log_q_tau]), dim=0).to(self._device)
        discrim_output = torch.exp(log_p_tau - log_pq)

        return log_q_tau, log_p_tau, log_pq, discrim_output

    def calculate_loss(self, inputs, obs, obs_next, path_probs, labels):
        log_q_tau, log_p_tau, log_pq, discrim_output = self.forward(inputs, obs, obs_next, path_probs)
        loss = -torch.mean(labels * (log_p_tau - log_pq) + (1 - labels) *  (log_q_tau - log_pq)).to(self._device)

        # Calculate L2 loss on model parameters
        l2_loss = 0.01 * sum(self.l2_loss(p, torch.zeros_like(p)) for p in self.parameters())

        return loss + self.l2_loss_ratio * l2_loss

    def train(self, inputs, optimizer, obs, obs_next, path_probs, labels):
        optimizer.zero_grad()
        loss = self.calculate_loss(inputs, obs, obs_next, path_probs, labels)
        loss.backward()
        optimizer.step()
        return loss.item()

    def get_reward(self, inputs, obs, obs_next, path_probs, discrim_score=False, only_rew=True, weighted_rew=False):
        with torch.no_grad():
            if discrim_score:
                log_q_tau, log_p_tau, log_pq, discrim_output = self(inputs, obs, obs_next, path_probs)
                score = torch.log(discrim_output + 1e-20) - torch.log(1 - discrim_output + 1e-20)
            else:
                outputs = [self.networks[i](inputs[i].to(torch.float32)) for i in range(self.n_networks)] 
                for i in range(self.n_networks):
                    if len(outputs[i].shape)==1:
                        outputs[i] = outputs[i].reshape(1, 1)
                rew_inputs = torch.cat(outputs, dim=1)
                score = self.reward_net(rew_inputs.to(torch.float32))
        if weighted_rew:
            weights = copy.deepcopy(self.reward_net.state_dict()['0.weight'][0].numpy())
            outputs = [weights[i]*outputs[i] for i in range(len(outputs))]
            return score, outputs 
        elif only_rew:
            return score
        else:
            return score, outputs 

        return reward2, p_tau, p_tau2 

    def get_value(self, obs):
        with torch.no_grad():
            value = self.value_net(obs.to(torch.float32))
            return value 

        return reward2, p_tau, p_tau2 

    def get_reward_weighted(self, inputs, obs, obs_next, path_probs, rate=[0.1, 0.1], expert_prob=True):
        with torch.no_grad():
            outputs = [self.networks[i](inputs[i].to(torch.float32)) for i in range(self.n_networks)] 
            rew_inputs = torch.cat(outputs, dim=1)
            reward = self.reward_net(rew_inputs.to(torch.float32)).numpy()

            outputs = rew_inputs.numpy()
            weights = copy.deepcopy(self.reward_net.state_dict()['0.weight'][0].numpy())
            weights += weights*np.array(rate)
            bias = self.reward_net.state_dict()['0.bias'][0].numpy()
            reward2 = outputs @ weights.T + bias

            p_tau = None
            p_tau2 = None
            if expert_prob:
                #rew_input = obs if self.state_only else torch.cat([obs, acs], dim=1)
                value_fn = self.value_net(obs.to(torch.float32)).numpy()
                value_fn_next = self.value_next_net(obs_next.to(torch.float32)).numpy()


                log_p_tau = reward + self.gamma * value_fn_next - value_fn
                tf = np.abs(log_p_tau)<5
                p_tau = np.zeros(log_p_tau.shape)
                p_tau[tf] = np.exp(-np.abs(log_p_tau[tf]))
                p_tau = p_tau.flatten()

                log_p_tau2 = reward2 + self.gamma * value_fn_next - value_fn
                tf = np.abs(log_p_tau2)<5
                p_tau2 = np.zeros(log_p_tau2.shape)
                p_tau2[tf] = np.exp(-np.abs(log_p_tau2[tf]))
                p_tau2 = p_tau2.flatten()


        return reward, reward2, p_tau, p_tau2 

    def get_num_nets(self):
        return self.n_networks

    def get_net_labels(self):
        return self.labels

    def save(self, filename=""):
        for i in range(self.n_networks):
            fname = osp.join(logger.get_dir(), "disc_"+f"{self.labels[i]}"+filename+".pth")
            torch.save(self.networks[i].state_dict(), fname)

        fname = osp.join(logger.get_dir(), "disc_reward"+filename+".pth")
        torch.save(self.reward_net.state_dict(), fname)

        fname = osp.join(logger.get_dir(), "disc_value"+filename+".pth")
        torch.save(self.value_net.state_dict(), fname)
        print(f'Saved discriminator param (reward, value -{filename})')

    def load(self, path, filename, use_eval=False):
        for i in range(self.n_networks):
            fname = osp.join(path, "disc_"+f"{self.labels[i]}"+filename+".pth")
            self.networks[i].load_state_dict(torch.load(fname))

        fname = osp.join(path, "disc_reward"+filename+".pth")
        self.reward_net.load_state_dict(torch.load(fname))
        fname = osp.join(path, "disc_value"+filename+".pth")
        self.value_net.load_state_dict(torch.load(fname))
        if use_eval:
            # if you want to erase noise of output, you should do use_eval=True
            for i in range(self.n_networks):
                self.networks[i].eval()
            self.reward_net.eval()
            self.value_net.eval()

    def load_with_path(self, pathes, reward_path, value_path, use_eval=False):
        for i in range(self.n_networks):
            self.networks[i].load_state_dict(torch.load(pathes[i]))
        self.reward_net.load_state_dict(torch.load(reward_path))
        self.value_net.load_state_dict(torch.load(value_path))
        if use_eval:
            # if you want to erase noise of output, you should do use_eval=True
            for i in range(self.n_networks):
                self.networks[i].eval()
            self.reward_net.eval()
            self.value_net.eval()

    def savefig_weights(self, path):
        net = self.reward_net
        weights = copy.deepcopy(self.reward_net.state_dict()['0.weight'][0].numpy()).reshape(1, self.n_networks)[0]
        bias = copy.deepcopy(self.reward_net.state_dict()['0.bias'][0].numpy()).reshape(1, 1)[0]
        data = list(weights)+list(bias)
        label = self.labels + ['bias']
        plt.figure()
        plt.title('Weights')
        plt.bar(label, data)
        plt.savefig(path)
        plt.close()
        print(f'Saved as {path}')

    def print_weights(self, only_rew=True):
        if only_rew:
            networks = [self.reward_net]
        else:
            networks = self.networks + [self.reward_net]

        for net in networks:
            weights = net.state_dict()
            for name, param in weights.items():
                print(name, param.size())
                print(param)

    def create_inputs(self, ob_shape, nacs, horizon, mu_dists):
        from games.predator_prey import goal_distance
        num_agents = len(mu_dists)
        inputs = [{} for _ in range(num_agents)]
        size = ob_shape[0]

        for idx in range(num_agents):
            for x in range(ob_shape[1]):
                for y in range(ob_shape[0]):
                    for t in range(horizon):
                        mu = [mu_dists[i][t, y, x] for i in range(len(mu_dists))]
                        for a in range(nacs):
                            x_onehot = onehot(x, ob_shape[0]).tolist()
                            y_onehot = onehot(y, ob_shape[1]).tolist()
                            a_onehot = onehot(a, nacs).tolist()
                            state = x_onehot + y_onehot
                            dx, dy = goal_distance(x, y, idx)
                            dxy = [dx, dy]
                            dxy_abs = np.abs(dxy).tolist()

                            dx = dx if dx<0 else np.abs(dx)+size
                            dy = dy if dy<0 else np.abs(dy)+size
                            dx_onehot = onehot(dx, size*2).tolist()
                            dy_onehot = onehot(dy, size*2).tolist()
                            dxy_onehot = dx_onehot+dy_onehot 
                            input = []
                            for n in range(self.n_networks):
                                input_str = self.labels[n]
                                if input_str == 'state':
                                    input.append(torch.Tensor(state))
                                elif input_str == 'state_a':
                                    input.append(torch.Tensor(state+a_onehot))
                                elif input_str == 'mu':
                                    input.append(torch.Tensor(mu))
                                elif input_str == 'mu_a':
                                    input.append(torch.Tensor(mu+a_onehot))
                                elif input_str == 'dxy':
                                    input.append(torch.Tensor(dxy))
                                elif input_str == 'dxy_a':
                                    input.append(torch.Tensor(dxy+a_onehot))
                                elif input_str == 'act':
                                    input.append(torch.Tensor(a_onehot))
                            inputs[idx][f'{x}-{y}-{t}-{a}-m'] = input
                            t_onehot = onehot(t, horizon)
                            #obs = torch.Tensor(state+[mu[idx]])
                            obs = torch.Tensor(state+mu)
                            inputs[idx][f'obs-{x}-{y}-{t}-m'] = obs 
        return inputs



if __name__ == "__main__":
    rning_rate = 0.01
    hidden_size = 128
    l2_loss_ratio = 0.01
    discount = 0.9
    batch_size = 32
    total_steps = 100
