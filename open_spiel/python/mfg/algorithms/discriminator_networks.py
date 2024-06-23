import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logger
import os
import os.path as osp

def onehot(value, depth):
    a = np.zeros([depth])
    a[value] = 1
    return a


def multionehot(values, depth):
    a = np.zeros([values.shape[0], depth])
    for i in range(values.shape[0]):
        a[i, int(values[i])] = 1
    return a

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
    else:
        assert False, f'not matched disc type: {net_input}'
    return labels

def get_net_inputs():
    net_inputs = ['s_mu_a',
                  'sa_mu',
                  's_mua',
                  'dxy_mu_a',
                  'dxya_mu',
                  'dxy_mua',]
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
    else:
        assert False, f'not matched disc type: {net_input}'

def is_networks(filename):
    labels = get_net_inputs()
    for label in labels:
        if label in filename:
            print(f'filename is detected as {label} model.')
            return True
    return False 

def get_net_label(filename):
    net_inputs = get_net_inputs()
    for net_input in net_inputs:
        if net_input in filename:
            return net_labels(net_input)
    return None

def get_net_input(filename):
    net_inputs = get_net_inputs()
    for net_input in net_inputs:
        if net_input in filename:
            return net_inputs
    return None

class Discriminator(nn.Module):
    def __init__(self, input_shapes, obs_shape, labels, device, discount=0.99, hidden_size=128, l2_loss_ratio=0.01):
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
            net = nn.Sequential(
                nn.Linear(input_shapes[i], hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            ).to(self._device)
            self.networks.append(net)


        self.reward_net = nn.Sequential(
            nn.Linear(self.n_networks, 1),
        ).to(self._device)

        # Define layers for value function network
        self.value_net = nn.Sequential(
            nn.Linear(obs_shape, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        ).to(self._device)

        self.value_next_net = self.value_net


        self.l2_loss = nn.MSELoss()

    def forward(self, inputs, obs, obs_next, path_probs):
        #rew_input = obs if self.state_only else torch.cat([obs, acs], dim=1)
        outputs = [self.networks[i](inputs[i].to(torch.float32)) for i in range(self.n_networks)] 
        outputs = torch.cat(outputs, dim=1)
        reward = self.reward_net(outputs.to(torch.float32))

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
                rew_inputs = torch.cat(outputs, dim=1)
                score = self.reward_net(rew_inputs.to(torch.float32))
        if weighted_rew:
            weights = self.reward_net.state_dict()['0.weight'][0].numpy()
            outputs = [weights[i]*outputs[i] for i in range(len(outputs))]
            return score, outputs 
        elif only_rew:
            return score
        else:
            return score, outputs 

    def get_num_nets(self):
        return self.n_networks

    def get_nets_labels(self):
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
                            input = []
                            for n in range(self.n_networks):
                                input_str = self.labels[n]
                                if input_str == 'state':
                                    input.append(torch.Tensor(state))
                                elif input_str == 'state_a':
                                    input.append(torch.Tensor(state+a_onehot))
                                elif input_str == 'mu':
                                    input.append(mu)
                                elif input_str == 'mu_a':
                                    input.append(torch.Tensor(mu+a_onehot))
                                elif input_str == 'dxy':
                                    input.append(dxy)
                                elif input_str == 'dxy_a':
                                    input.append(torch.Tensor(dxy+a_onehot))
                                elif input_str == 'act':
                                    input.append(torch.Tensor(a_onehot))
                            inputs[idx][f'{x}-{y}-{t}-{a}-m'] = input
        return inputs



if __name__ == "__main__":
    rning_rate = 0.01
    hidden_size = 128
    l2_loss_ratio = 0.01
    discount = 0.9
    batch_size = 32
    total_steps = 100
