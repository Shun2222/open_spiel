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
    elif net_input=='dist_mu':
        labels = ['dist', 'mu']
    elif net_input=='mu':
        labels = ['mu']
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
                  'dist_mu',
                  'mu',
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
    elif net_input=='dist_mu':
        #inputs = [state_size*2-1, nmu]
        inputs = [1, nmu]
    elif net_input=='mu':
        inputs = [nmu]
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
        #dxy_abs = np.abs(dxy)

        #dx[dx<0] = np.abs(dx[dx<0])+size
        #dy[dy<0] = np.abs(dy[dy<0])+size
        #dx_onehot = multionehot(dx, size*2)
        #dy_onehot = multionehot(dy, size*2)
        #dxy_onehot = np.concatenate([dx_onehot, dy_onehot], axis=1)

        inputs = [torch.from_numpy(dxy),
                    torch.from_numpy(mu),]
    elif net_input=='dist_mu':
        x, y, t, mu = divide_obs(obs_mu, size, use_argmax=True)
        dx, dy = goal_distance(x, y, idx)
        dist = np.array(np.sqrt(dx**2+dy**2))

        inputs = [torch.from_numpy(dist),
                    torch.from_numpy(mu),]
    elif net_input=='mu':
        x, y, t, mu = divide_obs(obs_mu, size, use_argmax=True)
        inputs = [torch.from_numpy(mu),]


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

def is_divided_value(filename):
    if "divided_value" in filename:
        return True
    return False

def get_net_input(filename):
    net_inputs = get_net_inputs()
    detected_input = []
    ignore_words = ["multi"]

    for word in ignore_words:
        filename = filename.replace(word, '')


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
    def __init__(self, input_shapes, obs_shape, labels, device, discount=0.99, hidden_size=128, l2_loss_ratio=0.01, num_hidden=1):
        super(Discriminator, self).__init__()
        assert len(input_shapes)<=len(labels), f'not enough labels'

        self.gamma = discount
        self.hidden_size = hidden_size
        self.l2_loss_ratio = l2_loss_ratio
        self._device = device
        self.labels = labels

        self.n_networks = len(input_shapes)
        self.networks = []

        def create_net(input_shape, num_hidden):
            if num_hidden==1:
                net = nn.Sequential(
                    nn.Linear(input_shape, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, 1)
                )
            elif num_hidden==2:
                net = nn.Sequential(
                    nn.Linear(input_shape, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, 1)
                )
            elif num_hidden==3:
                net = nn.Sequential(
                    nn.Linear(input_shape, hidden_size),
                    nn.Linear(input_shape, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, 1)
                )
            return net

        self.net1 = create_net(input_shapes[0], num_hidden).to(self._device)

        self.reward_net = nn.Sequential(
            nn.Linear(self.n_networks, 1, bias=False),
        ).to(self._device)

        # Define layers for value function network
        self.value_net1 = create_net(input_shapes[0], num_hidden).to(self._device)
        self.value_next_net1 = self.value_net1

        self.l2_loss = nn.MSELoss()

    def forward(self, input1, input1_next, path_probs):
        #rew_input = obs if self.state_only else torch.cat([obs, acs], dim=1)
        output = self.net1(input1.to(torch.float32)) 
        reward = self.reward_net(output.to(torch.float32)) 

        value_fn = self.value_net1(input1.to(torch.float32))
        value_fn_next = self.value_next_net1(input1_next.to(torch.float32))

        ws = self.get_weights()
        value_fn = ws[0] * value_fn
        value_fn_next = ws[0] * value_fn_next

        log_q_tau = path_probs
        log_p_tau = reward + self.gamma * value_fn_next - value_fn
        log_pq = torch.logsumexp(torch.stack([log_p_tau, log_q_tau]), dim=0).to(self._device)
        discrim_output = torch.exp(log_p_tau - log_pq)

        return log_q_tau, log_p_tau, log_pq, discrim_output

    def calculate_loss(self, input1, input1_next, path_probs, labels):
        log_q_tau, log_p_tau, log_pq, discrim_output = self.forward(input1, input1_next, path_probs)
        loss = -torch.mean(labels * (log_p_tau - log_pq) + (1 - labels) *  (log_q_tau - log_pq)).to(self._device)

        # Calculate L2 loss on model parameters
        l2_loss = 0.01 * sum(self.l2_loss(p, torch.zeros_like(p)) for p in self.parameters())

        return loss + self.l2_loss_ratio * l2_loss

    def train(self, input1, input1_next, optimizer, path_probs, labels):
        optimizer.zero_grad()
        loss = self.calculate_loss(input1, input1_next, path_probs, labels)
        loss.backward()
        optimizer.step()
        return loss.item()

    def get_reward(self, inputs, discrim_score=False, only_rew=True, weighted_rew=False):
        with torch.no_grad():
            if discrim_score:
                #log_q_tau, log_p_tau, log_pq, discrim_output = self(inputs, obs, obs_next, path_probs)
                #score = torch.log(discrim_output + 1e-20) - torch.log(1 - discrim_output + 1e-20)
                return None
            else:
                input1 = inputs[0]
                output1 = self.net1(input1.to(torch.float32)) 
                if len(output1.shape)==1:
                    output1 = output1.reshape(1, 1)
                score = self.reward_net(output1.to(torch.float32))
                outputs = [output1]
        if weighted_rew:
            weights = copy.deepcopy(self.reward_net.state_dict()['0.weight'][0].numpy())
            outputs = [output1]
            outputs = [weights[i]*outputs[i] for i in range(len(outputs))]
            return score, outputs 
        elif only_rew:
            return score
        else:
            return score, outputs 


    def get_value(self, inputs, only_value=True, weighted_value=False):
        with torch.no_grad():
            input1 = inputs[0]
            value_fn1 = self.value_net1(input1.to(torch.float32))

            ws = self.get_weights()
            value = ws[0] * value_fn1 
            if only_value:
                return value 
            elif weighted_value:
                return value, [ws[0] * value_fn1]
            else:
                return value, [value_fn1]

    def get_reward_weighted(self, inputs, rate=[0.1, 0.1]):
        with torch.no_grad():
            input1 = inputs[0]
            output1 = self.net1(input1.to(torch.float32)) 
            if len(output1.shape)==1:
                output1 = output1.reshape(1, 1)
            reward = self.reward_net(output1.to(torch.float32)).numpy()

            outputs = output1.numpy()
            weights = copy.deepcopy(self.reward_net.state_dict()['0.weight'][0].numpy())
            for i in range(len(weights)):
                weights[i] = weights[i]*np.array(rate[i])

            #bias = self.reward_net.state_dict()['0.bias'][0].numpy()
            #reward2 = outputs @ weights.T + bias
            reward2 = torch.from_numpy(outputs @ weights.T )
            outputs = torch.from_numpy(outputs)
            return reward, reward2, outputs


    def get_reward_weighted_with_probs(self, inputs, inputs_next, rate=[0.1, 0.1], expert_prob=True):
        with torch.no_grad():
            input1 = inputs[0]
            input1_next = inputs_next[0]
            output1 = self.net1(input1.to(torch.float32)) 
            if len(output1.shape)==1:
                output1 = output1.reshape(1, 1)
            reward = self.reward_net(output1.to(torch.float32)).numpy()

            outputs = output1.numpy()
            weights = copy.deepcopy(self.reward_net.state_dict()['0.weight'][0].numpy())
            weights = weights*np.array(rate)

            #bias = self.reward_net.state_dict()['0.bias'][0].numpy()
            #reward2 = outputs @ weights.T + bias
            reward2 = outputs @ weights.T 

            p_tau = None
            p_tau2 = None
            if expert_prob:
                value_fn1 = self.value_net1(input1.to(torch.float32)).numpy()
                value_fn_next1 = self.value_next_net1(input1_next.to(torch.float32)).numpy()

                ws = self.get_weights()
                value_fn = ws[0] * value_fn1
                value_fn_next = ws[0] * value_fn_next1

                log_p_tau = reward + self.gamma * value_fn_next - value_fn
                tf = np.abs(log_p_tau)<5
                p_tau = np.zeros(log_p_tau.shape)
                p_tau[tf] = np.exp(-np.abs(log_p_tau[tf]))
                p_tau = p_tau.flatten()

                ws = self.get_weights()
                value_fn = weights[0] * value_fn1 + weights[1]
                value_fn_next = weights[0] * value_fn_next1

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
        fname = osp.join(logger.get_dir(), "disc_"+f"{self.labels[0]}"+filename+".pth")
        torch.save(self.net1.state_dict(), fname)

        fname = osp.join(logger.get_dir(), "disc_reward"+filename+".pth")
        torch.save(self.reward_net.state_dict(), fname)

        fname = osp.join(logger.get_dir(), "disc_value_"+f"{self.labels[0]}"+filename+".pth")
        torch.save(self.value_net1.state_dict(), fname)
        print(f'Saved discriminator param (reward, value -{filename}) in {logger.get_dir()}')

    def load(self, path, filename, use_eval=False):
        fname = osp.join(path, "disc_"+f"{self.labels[0]}"+filename+".pth")
        self.net1.load_state_dict(torch.load(fname))

        fname = osp.join(path, "disc_reward"+filename+".pth")
        self.reward_net.load_state_dict(torch.load(fname))

        fname = osp.join(path, "disc_value_"+f"{self.labels[0]}"+filename+".pth")
        self.value_net1.load_state_dict(torch.load(fname))
        if use_eval:
            # if you want to erase noise of output, you should do use_eval=True
            self.net1.eval()
            self.reward_net.eval()
            self.value_net1.eval()

    def load_with_path(self, pathes, reward_path, value_pathes, use_eval=False):
        self.net1.load_state_dict(torch.load(pathes[0]))
        self.reward_net.load_state_dict(torch.load(reward_path))
        self.value_net1.load_state_dict(torch.load(value_path[0]))
        if use_eval:
            # if you want to erase noise of output, you should do use_eval=True
            self.reward_net.eval()
            self.value_net1.eval()

    def savefig_weights(self, path):
        net = self.reward_net
        weights = copy.deepcopy(self.reward_net.state_dict()['0.weight'][0].numpy()).reshape(1, self.n_networks)[0]
        data = list(weights)
        label = self.labels

        plt.figure()
        plt.title('Weights')
        plt.bar(label, data)
        plt.savefig(path)
        plt.close()

        print(f'Saved as {path}')
        print(f'Not exist weight.')

    def get_weights(self, only_rew=True):
        return copy.deepcopy(self.reward_net.state_dict()['0.weight'][0].numpy())

    def print_weights(self, only_rew=True):
        if only_rew:
            networks = [self.reward_net]
        else:
            networks = [self.net1] + [self.reward_net]

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
                        mu = [mu_dists[idx][t, y, x]]
                        for k in range(num_agents):
                            if k!=idx:
                                mu.append(mu_dists[k][t, y, x])
                        for a in range(nacs):
                            x_onehot = onehot(x, ob_shape[0]).tolist()
                            y_onehot = onehot(y, ob_shape[1]).tolist()
                            a_onehot = onehot(a, nacs).tolist()
                            state = x_onehot + y_onehot
                            dx, dy = goal_distance(x, y, idx)
                            dist = list(np.array([np.sqrt(dx**2+dy**2)]))
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
                                elif input_str == 'dist':
                                    input.append(torch.Tensor(dist))
                                else:
                                    assert False, f'unexpected label is detected: {input_str}'
                            inputs[idx][f'{x}-{y}-{t}-{a}-m'] = input
                            t_onehot = onehot(t, horizon)
                            #obs = torch.Tensor(state+[mu[idx]])
                            obs = torch.Tensor(state+mu)
                            inputs[idx][f'obs-{x}-{y}-{t}-m'] = obs 
        return inputs

class Discriminator_2nets(nn.Module):
    def __init__(self, input_shapes, obs_shape, labels, device, discount=0.99, hidden_size=128, l2_loss_ratio=0.01, num_hidden=1):
        super(Discriminator_2nets, self).__init__()
        assert len(input_shapes)<=len(labels), f'not enough labels'

        self.gamma = discount
        self.hidden_size = hidden_size
        self.l2_loss_ratio = l2_loss_ratio
        self._device = device
        self.labels = labels

        self.n_networks = len(input_shapes)
        self.networks = []
        def create_net(input_shape, num_hidden):
            if num_hidden==1:
                net = nn.Sequential(
                    nn.Linear(input_shape, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, 1)
                )
            elif num_hidden==2:
                net = nn.Sequential(
                    nn.Linear(input_shape, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, 1)
                )
            elif num_hidden==3:
                net = nn.Sequential(
                    nn.Linear(input_shape, hidden_size),
                    nn.Linear(input_shape, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, 1)
                )
            return net
        self.net1 = create_net(input_shapes[0], num_hidden).to(self._device)
        self.net2 = create_net(input_shapes[1], num_hidden).to(self._device)


        self.reward_net = nn.Sequential(
            nn.Linear(self.n_networks, 1, bias=False),
        ).to(self._device)

        # Define layers for value function network
        self.value_net1 = create_net(input_shapes[0], num_hidden).to(self._device)
        self.value_net2 = create_net(input_shapes[1], num_hidden).to(self._device)
        self.value_next_net1 = self.value_net1
        self.value_next_net2 = self.value_net2

        self.l2_loss = nn.MSELoss()

    def forward(self, input1, input2, input1_next, input2_next, path_probs):
        #rew_input = obs if self.state_only else torch.cat([obs, acs], dim=1)
        output1 = self.net1(input1.to(torch.float32)) 
        output2 = self.net2(input2.to(torch.float32)) 
        outputs = torch.cat((output1, output2), dim=1)
        reward = self.reward_net(outputs.to(torch.float32))

        value_fn1 = self.value_net1(input1.to(torch.float32))
        value_fn2 = self.value_net2(input2.to(torch.float32))
        value_fn_next1 = self.value_next_net1(input1_next.to(torch.float32))
        value_fn_next2 = self.value_next_net2(input2_next.to(torch.float32))

        ws = self.get_weights()
        value_fn = ws[0] * value_fn1 + ws[1] * value_fn2
        value_fn_next = ws[0] * value_fn_next1 + ws[1] * value_fn_next2

        log_q_tau = path_probs
        log_p_tau = reward + self.gamma * value_fn_next - value_fn
        log_pq = torch.logsumexp(torch.stack([log_p_tau, log_q_tau]), dim=0).to(self._device)
        discrim_output = torch.exp(log_p_tau - log_pq)

        return log_q_tau, log_p_tau, log_pq, discrim_output

    def calculate_loss(self, input1, input2, input1_next, input2_next, path_probs, labels):
        log_q_tau, log_p_tau, log_pq, discrim_output = self.forward(input1, input2, input1_next, input2_next, path_probs)
        loss = -torch.mean(labels * (log_p_tau - log_pq) + (1 - labels) *  (log_q_tau - log_pq)).to(self._device)

        # Calculate L2 loss on model parameters
        l2_loss = 0.01 * sum(self.l2_loss(p, torch.zeros_like(p)) for p in self.parameters())

        return loss + self.l2_loss_ratio * l2_loss

    def train(self, input1, input2, input1_next, input2_next, optimizer, path_probs, labels):
        optimizer.zero_grad()
        loss = self.calculate_loss(input1, input2, input1_next, input2_next, path_probs, labels)
        loss.backward()
        optimizer.step()
        return loss.item()

    def get_reward(self, inputs, discrim_score=False, only_rew=True, weighted_rew=False):
        with torch.no_grad():
            if discrim_score:
                #log_q_tau, log_p_tau, log_pq, discrim_output = self(inputs, obs, obs_next, path_probs)
                #score = torch.log(discrim_output + 1e-20) - torch.log(1 - discrim_output + 1e-20)
                return None
            else:
                input1 = inputs[0]
                input2 = inputs[1]
                output1 = self.net1(input1.to(torch.float32)) 
                if len(output1.shape)==1:
                    output1 = output1.reshape(1, 1)
                output2 = self.net2(input2.to(torch.float32)) 
                if len(output2.shape)==1:
                    output2 = output2.reshape(1, 1)
                rew_inputs = torch.cat((output1, output2), dim=1)
                score = self.reward_net(rew_inputs.to(torch.float32))
                outputs = [output1, output2]
        if weighted_rew:
            weights = copy.deepcopy(self.reward_net.state_dict()['0.weight'][0].numpy())
            outputs = [output1, output2]
            outputs = [weights[i]*outputs[i] for i in range(len(outputs))]
            return score, outputs 
        elif only_rew:
            return score
        else:
            return score, outputs 


    def get_value(self, inputs, only_value=True, weighted_value=False):
        with torch.no_grad():
            input1 = inputs[0]
            input2 = inputs[1]
            value_fn1 = self.value_net1(input1.to(torch.float32))
            value_fn2 = self.value_net2(input2.to(torch.float32))

            ws = self.get_weights()
            value = ws[0] * value_fn1 + ws[1] * value_fn2
            if only_value:
                return value 
            elif weighted_value:
                return value, [ws[0] * value_fn1, ws[1] * value_fn2]
            else:
                return value, [value_fn1, value_fn2]


    def get_reward_weighted(self, inputs, rate=[0.1, 0.1]):
        with torch.no_grad():
            input1 = inputs[0]
            input2 = inputs[1]
            output1 = self.net1(input1.to(torch.float32)) 
            if len(output1.shape)==1:
                output1 = output1.reshape(1, 1)
            output2 = self.net2(input2.to(torch.float32)) 
            if len(output2.shape)==1:
                output2 = output2.reshape(1, 1)
            rew_inputs = torch.cat((output1, output2), dim=1)
            reward = self.reward_net(rew_inputs.to(torch.float32)).numpy()

            outputs = rew_inputs.numpy()
            weights = copy.deepcopy(self.reward_net.state_dict()['0.weight'][0].numpy())
            for i in range(len(weights)):
                weights[i] = weights[i]*np.array(rate[i])

            #bias = self.reward_net.state_dict()['0.bias'][0].numpy()
            #reward2 = outputs @ weights.T + bias
            reward2 = torch.from_numpy(outputs @ weights.T )
            outputs = torch.from_numpy(outputs)
            return reward, reward2, outputs


    def get_reward_weighted_with_probs(self, inputs, inputs_next, rate=[0.1, 0.1], expert_prob=True):
        with torch.no_grad():
            input1 = inputs[0]
            input2 = inputs[1]
            input1_next = inputs_next[0]
            input2_next = inputs_next[1]
            output1 = self.net1(input1.to(torch.float32)) 
            if len(output1.shape)==1:
                output1 = output1.reshape(1, 1)
            output2 = self.net2(input2.to(torch.float32)) 
            if len(output2.shape)==1:
                output2 = output2.reshape(1, 1)
            rew_inputs = torch.cat((output1, output2), dim=1)
            reward = self.reward_net(rew_inputs.to(torch.float32)).numpy()

            outputs = rew_inputs.numpy()
            weights = copy.deepcopy(self.reward_net.state_dict()['0.weight'][0].numpy())
            weights = weights*np.array(rate)

            #bias = self.reward_net.state_dict()['0.bias'][0].numpy()
            #reward2 = outputs @ weights.T + bias
            reward2 = outputs @ weights.T 

            p_tau = None
            p_tau2 = None
            if expert_prob:
                value_fn1 = self.value_net1(input1.to(torch.float32)).numpy()
                value_fn2 = self.value_net2(input2.to(torch.float32)).numpy()
                value_fn_next1 = self.value_next_net1(input1_next.to(torch.float32)).numpy()
                value_fn_next2 = self.value_next_net2(input2_next.to(torch.float32)).numpy()

                ws = self.get_weights()
                value_fn = ws[0] * value_fn1 + ws[1] * value_fn2
                value_fn_next = ws[0] * value_fn_next1 + ws[1] * value_fn_next2

                log_p_tau = reward + self.gamma * value_fn_next - value_fn
                tf = np.abs(log_p_tau)<5
                p_tau = np.zeros(log_p_tau.shape)
                p_tau[tf] = np.exp(-np.abs(log_p_tau[tf]))
                p_tau = p_tau.flatten()

                ws = self.get_weights()
                value_fn = weights[0] * value_fn1 + weights[1] * value_fn2
                value_fn_next = weights[0] * value_fn_next1 + weights[1] * value_fn_next2

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
        fname = osp.join(logger.get_dir(), "disc_"+f"{self.labels[0]}"+filename+".pth")
        torch.save(self.net1.state_dict(), fname)
        fname = osp.join(logger.get_dir(), "disc_"+f"{self.labels[1]}"+filename+".pth")
        torch.save(self.net2.state_dict(), fname)

        fname = osp.join(logger.get_dir(), "disc_reward"+filename+".pth")
        torch.save(self.reward_net.state_dict(), fname)

        fname = osp.join(logger.get_dir(), "disc_value_"+f"{self.labels[0]}"+filename+".pth")
        torch.save(self.value_net1.state_dict(), fname)
        fname = osp.join(logger.get_dir(), "disc_value_"+f"{self.labels[1]}"+filename+".pth")
        torch.save(self.value_net2.state_dict(), fname)
        print(f'Saved discriminator param (reward, value -{filename}) in {logger.get_dir()}')

    def load(self, path, filename, use_eval=False):
        fname = osp.join(path, "disc_"+f"{self.labels[0]}"+filename+".pth")
        self.net1.load_state_dict(torch.load(fname))
        fname = osp.join(path, "disc_"+f"{self.labels[1]}"+filename+".pth")
        self.net2.load_state_dict(torch.load(fname))

        fname = osp.join(path, "disc_reward"+filename+".pth")
        self.reward_net.load_state_dict(torch.load(fname))
        fname = osp.join(path, "disc_value_"+f"{self.labels[0]}"+filename+".pth")
        self.value_net1.load_state_dict(torch.load(fname))
        fname = osp.join(path, "disc_value_"+f"{self.labels[1]}"+filename+".pth")
        self.value_net2.load_state_dict(torch.load(fname))
        if use_eval:
            # if you want to erase noise of output, you should do use_eval=True
            self.net1.eval()
            self.net2.eval()
            self.reward_net.eval()
            self.value_net1.eval()
            self.value_net2.eval()

    def load_with_path(self, pathes, reward_path, value_pathes, use_eval=False):
        self.net1.load_state_dict(torch.load(pathes[0]))
        self.net2.load_state_dict(torch.load(pathes[1]))
        self.reward_net.load_state_dict(torch.load(reward_path))
        self.value_net1.load_state_dict(torch.load(value_path[0]))
        self.value_net2.load_state_dict(torch.load(value_pathes[1]))
        if use_eval:
            # if you want to erase noise of output, you should do use_eval=True
            self.net1.eval()
            self.net2.eval()
            self.reward_net.eval()
            self.value_net.eval()

    def savefig_weights(self, path):
        net = self.reward_net
        weights = copy.deepcopy(self.reward_net.state_dict()['0.weight'][0].numpy()).reshape(1, self.n_networks)[0]
        #bias = copy.deepcopy(self.reward_net.state_dict()['0.bias'][0].numpy()).reshape(1, 1)[0]
        #data = list(weights)+list(bias)
        data = list(weights)
        #label = self.labels + ['bias']
        label = self.labels
        plt.figure()
        plt.title('Weights')
        plt.bar(label, data)
        plt.savefig(path)
        plt.close()
        print(f'Saved as {path}')

    def get_weights(self, only_rew=True):
        return copy.deepcopy(self.reward_net.state_dict()['0.weight'][0].numpy())

    def print_weights(self, only_rew=True):
        if only_rew:
            networks = [self.reward_net]
        else:
            networks = [self.net1, self.net2] + [self.reward_net]

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
                        mu = [mu_dists[idx][t, y, x]]
                        for k in range(num_agents):
                            if k!=idx:
                                mu.append(mu_dists[k][t, y, x])
                        for a in range(nacs):
                            x_onehot = onehot(x, ob_shape[0]).tolist()
                            y_onehot = onehot(y, ob_shape[1]).tolist()
                            a_onehot = onehot(a, nacs).tolist()
                            state = x_onehot + y_onehot
                            dx, dy = goal_distance(x, y, idx)
                            dist = list(np.array([np.sqrt(dx**2+dy**2)]))
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
                                elif input_str == 'dist':
                                    input.append(torch.Tensor(dist))
                                else:
                                    assert False, f'unexpected label is detected: {input_str}'
                            inputs[idx][f'{x}-{y}-{t}-{a}-m'] = input
                            t_onehot = onehot(t, horizon)
                            #obs = torch.Tensor(state+[mu[idx]])
                            obs = torch.Tensor(state+mu)
                            inputs[idx][f'obs-{x}-{y}-{t}-m'] = obs 
        return inputs

class Discriminator_3nets(nn.Module):
    def __init__(self, input_shapes, obs_shape, labels, device, discount=0.99, hidden_size=128, l2_loss_ratio=0.01, num_hidden=1):
        super(Discriminator_3nets, self).__init__()
        assert len(input_shapes)<=len(labels), f'not enough labels'

        self.gamma = discount
        self.hidden_size = hidden_size
        self.l2_loss_ratio = l2_loss_ratio
        self._device = device
        self.labels = labels

        self.n_networks = len(input_shapes)
        self.networks = []
        def create_net(input_shape, num_hidden):
            if num_hidden==1:
                net = nn.Sequential(
                    nn.Linear(input_shape, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, 1)
                )
            elif num_hidden==2:
                net = nn.Sequential(
                    nn.Linear(input_shape, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, 1)
                )
            elif num_hidden==3:
                net = nn.Sequential(
                    nn.Linear(input_shape, hidden_size),
                    nn.Linear(input_shape, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, 1)
                )
            return net
        self.net1 = create_net(input_shapes[0], num_hidden).to(self._device)
        self.net2 = create_net(input_shapes[1], num_hidden).to(self._device)
        self.net3 = create_net(input_shapes[2], num_hidden).to(self._device)


        self.reward_net = nn.Sequential(
            nn.Linear(self.n_networks, 1, bias=False),
        ).to(self._device)

        # Define layers for value function network
        self.value_net1 = create_net(input_shape[0], num_hidden).to(self._device)
        self.value_net2 = create_net(input_shape[1], num_hidden).to(self._device)
        self.value_net3 = create_net(input_shape[2], num_hidden).to(self._device)
        self.value_next_net1 = self.value_net1
        self.value_next_net2 = self.value_net2
        self.value_next_net3 = self.value_net3

        self.l2_loss = nn.MSELoss()

    def forward(self, input1, input2, input3, input1_next, input2_next, input3_next, path_probs):
        #rew_input = obs if self.state_only else torch.cat([obs, acs], dim=1)
        output1 = self.net1(input1.to(torch.float32)) 
        output2 = self.net2(input2.to(torch.float32)) 
        output3 = self.net3(input3.to(torch.float32)) 
        outputs = torch.cat((output1, output2, output3), dim=1)
        reward = self.reward_net(outputs.to(torch.float32))

        value_fn1 = self.value_net1(input1.to(torch.float32))
        value_fn2 = self.value_net2(input2.to(torch.float32))
        value_fn3 = self.value_net3(input3.to(torch.float32))
        value_fn_next1 = self.value_next_net1(input1_next)
        value_fn_next2 = self.value_next_net2(input2_next)
        value_fn_next3 = self.value_next_net3(input3_next)

        ws = self.get_weights()
        value_fn = ws[0] * value_fn1 + ws[1] * value_fn2 + ws[2] * value_fn3
        value_fn_next = ws[0] * value_fn_next1 + ws[1] * value_fn_next2 + ws[2] * value_fn_next3

        log_q_tau = path_probs
        log_p_tau = reward + self.gamma * value_fn_next - value_fn
        log_pq = torch.logsumexp(torch.stack([log_p_tau, log_q_tau]), dim=0).to(self._device)
        discrim_output = torch.exp(log_p_tau - log_pq)

        return log_q_tau, log_p_tau, log_pq, discrim_output

    def calculate_loss(self, input1, input2, input3, input1_next, input2_next, input3_next, path_probs, labels):
        log_q_tau, log_p_tau, log_pq, discrim_output = self.forward(input1, input2, input3, input1_next, input2_next, input3_next, path_probs)
        loss = -torch.mean(labels * (log_p_tau - log_pq) + (1 - labels) *  (log_q_tau - log_pq)).to(self._device)

        # Calculate L2 loss on model parameters
        l2_loss = 0.01 * sum(self.l2_loss(p, torch.zeros_like(p)) for p in self.parameters())

        return loss + self.l2_loss_ratio * l2_loss

    def train(self, input1, input2, input3, input1_next, input2_next, input3_next, optimizer, path_probs, labels):
        optimizer.zero_grad()
        loss = self.calculate_loss(input1, input2, input3, input1_next, input2_next, input3_next, path_probs, labels)
        loss.backward()
        optimizer.step()
        return loss.item()

    def get_reward(self, inputs, discrim_score=False, only_rew=True, weighted_rew=False):
        with torch.no_grad():
            if discrim_score:
                #log_q_tau, log_p_tau, log_pq, discrim_output = self(inputs, inputs_next, path_probs)
                #score = torch.log(discrim_output + 1e-20) - torch.log(1 - discrim_output + 1e-20)
                return None
            else:
                input1 = inputs[0]
                input2 = inputs[1]
                input3 = inputs[2]
                output1 = self.net1(input1.to(torch.float32)) 
                if len(output1.shape)==1:
                    output1 = output1.reshape(1, 1)
                output2 = self.net2(input2.to(torch.float32)) 
                if len(output2.shape)==1:
                    output2 = output2.reshape(1, 1)
                output3 = self.net3(input3.to(torch.float32)) 
                if len(output3.shape)==1:
                    output3 = output3.reshape(1, 1)
                rew_inputs = torch.cat((output1, output2, output3), dim=1)
                score = self.reward_net(rew_inputs.to(torch.float32))
        if weighted_rew:
            weights = copy.deepcopy(self.reward_net.state_dict()['0.weight'][0].numpy())
            outputs = [output1, output2, output3]
            outputs = [weights[i]*outputs[i] for i in range(len(outputs))]
            return score, outputs 
        elif only_rew:
            return score
        else:
            return score, outputs 

        return reward2, p_tau, p_tau2 

    def get_value(self, inputs):
        with torch.no_grad():
            input1 = inputs[0]
            input2 = inputs[1]
            input3 = inputs[2]
            value_fn1 = self.value_net1(input1.to(torch.float32))
            value_fn2 = self.value_net2(input2.to(torch.float32))
            value_fn3 = self.value_net3(input3.to(torch.float32))

            ws = self.get_weights()
            value = ws[0] * value_fn1 + ws[1] * value_fn2 + ws[2] * value_fn3
            return value 

        return reward2, p_tau, p_tau2 

    def get_reward_weighted(self, inputs, inputs_next, rate=[0.1, 0.1], expert_prob=True):
        with torch.no_grad():
            input1 = inputs[0]
            input2 = inputs[1]
            input3 = inputs[2]
            input1_next = inputs_next[0]
            input2_next = inputs_next[1]
            input3_next = inputs_next[2]
            output1 = self.net1(input1.to(torch.float32)) 
            if len(output1.shape)==1:
                output1 = output1.reshape(1, 1)
            output2 = self.net2(input2.to(torch.float32)) 
            if len(output2.shape)==1:
                output2 = output2.reshape(1, 1)
            output3 = self.net2(input3.to(torch.float32)) 
            if len(output3.shape)==1:
                output3 = output3.reshape(1, 1)
            rew_inputs = torch.cat((output1, output2, output3), dim=1)
            reward = self.reward_net(rew_inputs.to(torch.float32)).numpy()

            outputs = rew_inputs.numpy()
            weights = copy.deepcopy(self.reward_net.state_dict()['0.weight'][0].numpy())
            weights = weights*np.array(rate)

            #bias = self.reward_net.state_dict()['0.bias'][0].numpy()
            #reward2 = outputs @ weights.T + bias
            reward2 = outputs @ weights.T 

            p_tau = None
            p_tau2 = None
            if expert_prob:
                value_fn1 = self.value_net1(input1.to(torch.float32)).numpy()
                value_fn2 = self.value_net2(input2.to(torch.float32)).numpy()
                value_fn3 = self.value_net3(input3.to(torch.float32)).numpy()
                value_fn_next1 = self.value_next_net1(input1_next).numpy()
                value_fn_next2 = self.value_next_net2(input2_next).numpy()
                value_fn_next3 = self.value_next_net3(input3_next).numpy()

                ws = self.get_weights()
                value_fn = ws[0] * value_fn1 + ws[1] * value_fn2 + ws[2] * value_fn3
                value_fn_next = ws[0] * value_fn_next1 + ws[1] * value_fn_next2 + ws[2] * value_fn_next3

                log_p_tau = reward + self.gamma * value_fn_next - value_fn
                tf = np.abs(log_p_tau)<5
                p_tau = np.zeros(log_p_tau.shape)
                p_tau[tf] = np.exp(-np.abs(log_p_tau[tf]))
                p_tau = p_tau.flatten()

                ws = self.get_weights()
                value_fn = weights[0] * value_fn1 + weights[1] * value_fn2 + weights[2] * value_fn3
                value_fn_next = weights[0] * value_fn_next1 + weights[1] * value_fn_next2 + weights[2] * value_fn_next3

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
        fname = osp.join(logger.get_dir(), "disc_"+f"{self.labels[0]}"+filename+".pth")
        torch.save(self.net1.state_dict(), fname)
        fname = osp.join(logger.get_dir(), "disc_"+f"{self.labels[1]}"+filename+".pth")
        torch.save(self.net2.state_dict(), fname)
        fname = osp.join(logger.get_dir(), "disc_"+f"{self.labels[2]}"+filename+".pth")
        torch.save(self.net3.state_dict(), fname)

        fname = osp.join(logger.get_dir(), "disc_reward"+filename+".pth")
        torch.save(self.reward_net.state_dict(), fname)

        fname = osp.join(logger.get_dir(), "disc_value_"+f"{self.labels[0]}"+filename+".pth")
        torch.save(self.value_net1.state_dict(), fname)
        fname = osp.join(logger.get_dir(), "disc_value_"+f"{self.labels[1]}"+filename+".pth")
        torch.save(self.value_net2.state_dict(), fname)
        fname = osp.join(logger.get_dir(), "disc_value_"+f"{self.labels[2]}"+filename+".pth")
        torch.save(self.value_net3.state_dict(), fname)
        print(f'Saved discriminator param (reward, value -{filename}) in {logger.get_dir()}')

    def load(self, path, filename, use_eval=False):
        fname = osp.join(path, "disc_"+f"{self.labels[0]}"+filename+".pth")
        self.net1.load_state_dict(torch.load(fname))
        fname = osp.join(path, "disc_"+f"{self.labels[1]}"+filename+".pth")
        self.net2.load_state_dict(torch.load(fname))
        fname = osp.join(path, "disc_"+f"{self.labels[2]}"+filename+".pth")
        self.net3.load_state_dict(torch.load(fname))

        fname = osp.join(path, "disc_reward"+filename+".pth")
        self.reward_net.load_state_dict(torch.load(fname))
        fname = osp.join(path, "disc_value_"+f"{self.labels[0]}"+filename+".pth")
        self.value_net1.load_state_dict(torch.load(fname))
        fname = osp.join(path, "disc_value_"+f"{self.labels[1]}"+filename+".pth")
        self.value_net2.load_state_dict(torch.load(fname))
        fname = osp.join(path, "disc_value_"+f"{self.labels[2]}"+filename+".pth")
        self.value_net3.load_state_dict(torch.load(fname))
        if use_eval:
            self.net1.eval()
            self.net2.eval()
            self.net3.eval()
            self.reward_net.eval()
            self.value_net1.eval()
            self.value_net2.eval()
            self.value_net3.eval()

    def load_with_path(self, pathes, reward_path, value_pathes, use_eval=False):
        self.net1.load_state_dict(torch.load(pathes[0]))
        self.net2.load_state_dict(torch.load(pathes[1]))
        self.net3.load_state_dict(torch.load(pathes[2]))
        self.reward_net.load_state_dict(torch.load(reward_path))
        self.value_net1.load_state_dict(torch.load(value_path[0]))
        self.value_net2.load_state_dict(torch.load(value_pathes[1]))
        self.value_net3.load_state_dict(torch.load(value_pathes[2]))
        if use_eval:
            # if you want to erase noise of output, you should do use_eval=True
            self.net1.eval()
            self.net2.eval()
            self.net3.eval()
            self.reward_net.eval()
            self.value_net1.eval()
            self.value_net2.eval()
            self.value_net3.eval()

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

    def get_weights(self, only_rew=True):
        return copy.deepcopy(self.reward_net.state_dict()['0.weight'][0].numpy())

    def print_weights(self, only_rew=True):
        if only_rew:
            networks = [self.reward_net]
        else:
            networks = [self.net1, self.net2, self.net3] + [self.reward_net]

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
