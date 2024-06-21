import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logger
import os
import os.path as osp


class Discriminator(nn.Module):
    def __init__(self, input_shapes, obs_shape, labels, device, discount=0.99, hidden_size=128, l2_loss_ratio=0.01):
        super(Discriminator, self).__init__()
        assert len(input_shapes)<=len(labels), f'not enough labels'

        self.gamma = discount
        self.hidden_size = hidden_size
        self.l2_loss_ratio = l2_loss_ratio
        self._device = device

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

    def get_reward(self, inputs, obs, obs_next, path_probs, discrim_score=False, only_rew=True):
        with torch.no_grad():
            if discrim_score:
                log_q_tau, log_p_tau, log_pq, discrim_output = self(inputs, obs, obs_next, path_probs)
                score = torch.log(discrim_output + 1e-20) - torch.log(1 - discrim_output + 1e-20)
            else:
                outputs = [self.networks[i](inputs[i].to(torch.float32)) for i in range(self.n_networks)] 
                outputs = torch.cat(outputs, dim=1)
                score = self.reward_net(outputs.to(torch.float32))
        if only_rew:
            return score
        else:
            return score, outputs 


    def save(self, filename=""):
        for i in range(self.n_networks):
            fname = osp.join(logger.get_dir(), "disc_"+f"{labels[i]}"+filename+".pth")
            torch.save(self.networks[i].state_dict(), fname)

        fname = osp.join(logger.get_dir(), "disc_reward"+filename+".pth")
        torch.save(self.reward_net.state_dict(), fname)

        fname = osp.join(logger.get_dir(), "disc_value"+filename+".pth")
        torch.save(self.value_net.state_dict(), fname)
        print(f'Saved discriminator param (reward, value -{filename})')

    def load(self, pathes, ac_path, reward_path, value_path, use_eval=False):
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



if __name__ == "__main__":
    rning_rate = 0.01
    hidden_size = 128
    l2_loss_ratio = 0.01
    discount = 0.9
    batch_size = 32
    total_steps = 100

    # モデルのインスタンス化
    ob_shape =  2 # obsの形状を設定
    ac_shape =  2 # acsの形状を設定
    state_only =  True# state_onlyを設定
    discriminator = Discriminator(ob_shape, ac_shape, state_only, discount, hidden_size, l2_loss_ratio)

    # オプティマイザの設定
    optimizer = Adam(discriminator.parameters(), lr=learning_rate)

    # サンプルデータを生成
    num_samples = 1000
    obs = np.random.rand(100, 2)
    obs[50:, :] = 1
    acs = np.random.rand(100, 2)
    obs[50:, :] = 1
    obs_next = np.random.rand(100, 2)
    obs_next[50:, :] = 1
    path_probs = np.ones((100, 1))
    labels = np.zeros((100, 1))
    labels[50:, :] = 1

    # NumPyデータをPyTorchテンソルに変換
    obs_tensor = torch.FloatTensor(obs)
    acs_tensor = torch.FloatTensor(acs)
    obs_next_tensor = torch.FloatTensor(obs_next)
    path_probs_tensor = torch.FloatTensor(path_probs)
    labels_tensor = torch.FloatTensor(labels)

    # データローダーを作成
    dataset = TensorDataset(obs_tensor, acs_tensor, obs_next_tensor, path_probs_tensor, labels_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 学習ループ
    for step in range(total_steps):
        for batch_obs, batch_acs, batch_obs_next, batch_path_probs, batch_labels in data_loader:
            loss = discriminator.train(optimizer, batch_obs, batch_acs, batch_obs_next, batch_path_probs, batch_labels)

    if step % 10 == 0:
        print(f"Step {step}: Loss {loss:.4f}")

    # モデルの保存
    # torch.save(discriminator.state_dict(), 'discriminator.pth')
