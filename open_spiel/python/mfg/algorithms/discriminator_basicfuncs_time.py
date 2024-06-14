import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logger
import os
import os.path as osp

def divide_obs(obs, size, one_vec=False):
    if one_vec:
        obs_x = np.argmax(obs[:size])
        obs_y = np.argmax(obs[size:2*size])
        obs_t = np.argmax(obs[2*size:-4])
        obs_hatena = obs[-4]
        obs_mu = obs[-3:]

        obs_x = obs_x.reshape(1, 1)
        obs_y = obs_y.reshape(1, 1)
        obs_t = obs_y.reshape(1, 1)
        obs_mu = obs_mu.reshape(1, 3)

    else:
        obs = obs.T
        obs_x = np.argmax(obs[:size].T, axis=1)
        obs_y = np.argmax(obs[size:2*size].T, axis=1)
        obs_t = np.argmax(obs[2*size:-3].T, axis=1)
        obs_mu = obs[-3:].T

        obs_x = obs_x.reshape(len(obs_x), 1)
        obs_y = obs_y.reshape(len(obs_y), 1)
        obs_t = obs_y.reshape(len(obs_t), 1)
        obs_mu = obs_mu.reshape(len(obs_mu), 3)
    return obs_x, obs_y, obs_t, obs_mu

class Discriminator(nn.Module):
    def __init__(self, n_agent, time_size, distance_size, ob_shape, ac_shape, state_only, device, discount=0.99, hidden_size=128, l2_loss_ratio=0.01):
        super(Discriminator, self).__init__()
        self.state_only = state_only
        self.gamma = discount
        self.hidden_size = hidden_size
        self.l2_loss_ratio = l2_loss_ratio
        self._device = device

        # Define layers for reward network
        self.distance_net = nn.Sequential(
            nn.Linear(distance_size+time_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        ).to(self._device)

        self.mu_net = nn.Sequential(
            nn.Linear(n_agent+time_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        ).to(self._device)

        self.reward_net = nn.Sequential(
            nn.Linear(2, 1),
        ).to(self._device)

        # Define layers for value function network
        self.value_net = nn.Sequential(
            nn.Linear(ob_shape, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        ).to(self._device)

        # Define layers for value next function network
#         self.value_next_net = nn.Sequential(
#             nn.Linear(ob_shape, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, 1)
#         )
        self.value_next_net = self.value_net


        self.l2_loss = nn.MSELoss()

    def forward(self, distance, mus, obs, acs, obs_next, path_probs):
        #rew_input = obs if self.state_only else torch.cat([obs, acs], dim=1)
        distance_output = self.distance_net(distance.to(torch.float32))
        mu_output = self.mu_net(mus.to(torch.float32))
        reward_input = torch.cat((distance_output, mu_output), dim=1)
        reward = self.reward_net(reward_input)
        value_fn = self.value_net(obs)
        value_fn_next = self.value_next_net(obs_next)

        log_q_tau = path_probs
        log_p_tau = reward + self.gamma * value_fn_next - value_fn
        log_pq = torch.logsumexp(torch.stack([log_p_tau, log_q_tau]), dim=0).to(self._device)
        discrim_output = torch.exp(log_p_tau - log_pq)

        return log_q_tau, log_p_tau, log_pq, discrim_output

    def calculate_loss(self, distance, mus, obs, acs, obs_next, path_probs, labels):
        log_q_tau, log_p_tau, log_pq, discrim_output = self.forward(distance, mus, obs, acs, obs_next, path_probs)
        loss = -torch.mean(labels * (log_p_tau - log_pq) + (1 - labels) *  (log_q_tau - log_pq)).to(self._device)

        # Calculate L2 loss on model parameters
        l2_loss = 0.01 * sum(self.l2_loss(p, torch.zeros_like(p)) for p in self.parameters())

        return loss + self.l2_loss_ratio * l2_loss

    def train(self, distance, mus, optimizer, obs, acs, obs_next, path_probs, labels):
        optimizer.zero_grad()
        loss = self.calculate_loss(distance, mus, obs, acs, obs_next, path_probs, labels)
        loss.backward()
        optimizer.step()
        return loss.item()

    def get_reward(self, distance, mus, obs, acs, obs_next, path_probs, discrim_score=False, only_rew=True):
        with torch.no_grad():
            if discrim_score:
                log_q_tau, log_p_tau, log_pq, discrim_output = self(distance, mus, obs, acs, obs_next, path_probs)
                score = torch.log(discrim_output + 1e-20) - torch.log(1 - discrim_output + 1e-20)
            else:
                distance_output = self.distance_net(distance.to(torch.float32))
                mu_output = self.mu_net(mus.to(torch.float32))
                rew_input = torch.cat((distance_output, mu_output), dim=1)
                score = self.reward_net(rew_input.to(torch.float32))
        if only_rew:
            return score
        else:
            return score, distance_output, mu_output


    def save(self, filename=""):
        fname = osp.join(logger.get_dir(), "disc_distance"+filename+".pth")
        torch.save(self.distance_net.state_dict(), fname)
        fname = osp.join(logger.get_dir(), "disc_mu"+filename+".pth")
        torch.save(self.mu_net.state_dict(), fname)
        fname = osp.join(logger.get_dir(), "disc_reward"+filename+".pth")
        torch.save(self.reward_net.state_dict(), fname)
        fname = osp.join(logger.get_dir(), "disc_value"+filename+".pth")
        torch.save(self.value_net.state_dict(), fname)
        print(f'Saved discriminator param (reward, value -{filename})')

    def load(self, distance_path, mu_path, reward_path, value_path, use_eval=False):
        self.distance_net.load_state_dict(torch.load(distance_path))
        self.mu_net.load_state_dict(torch.load(mu_path))
        self.reward_net.load_state_dict(torch.load(reward_path))
        self.value_net.load_state_dict(torch.load(value_path))
        if use_eval:
            # if you want to erase noise of output, you should do use_eval=True
            self.distance_net.eval()
            self.mu_net.eval()
            self.reward_net.eval()
            self.value_net.eval()

    def print_weights(self, only_rew=True):
        if only_rew:
            networks = [self.reward_net]
        else:
            networks = [self.distance_net, self.mu_net, self.reward_net]

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
