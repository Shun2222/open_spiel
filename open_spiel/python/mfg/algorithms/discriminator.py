import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Discriminator(nn.Module):
    def __init__(self, ob_shape, ac_shape, state_only, device, discount=0.99, hidden_size=128, l2_loss_ratio=0.01):
        super(Discriminator, self).__init__()
        self.state_only = state_only
        self.gamma = discount
        self.hidden_size = hidden_size
        self.l2_loss_ratio = l2_loss_ratio

        # Define layers for reward network
        self.reward_net = nn.Sequential(
            nn.Linear(ob_shape if state_only else ob_shape + ac_shape, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        # Define layers for value function network
        self.value_net = nn.Sequential(
            nn.Linear(ob_shape, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        # Define layers for value next function network
        self.value_next_net = nn.Sequential(
            nn.Linear(ob_shape, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )


        self.l2_loss = nn.MSELoss()

    def forward(self, obs, acs, obs_next, path_probs):
        rew_input = obs if self.state_only else torch.cat([obs, acs], dim=1)
        reward = self.reward_net(rew_input)
        value_fn = self.value_net(obs)
        value_fn_next = self.value_next_net(obs_next)

        log_q_tau = path_probs
        log_p_tau = reward + self.gamma * value_fn_next - value_fn
        log_pq = torch.logsumexp(torch.stack([log_p_tau, log_q_tau]), dim=0)
        discrim_output = torch.exp(log_p_tau - log_pq)

        return log_q_tau, log_p_tau, log_pq, discrim_output

    def calculate_loss(self, obs, acs, obs_next, path_probs, labels):
        log_q_tau, log_p_tau, log_pq, discrim_output = self.forward(obs, acs, obs_next, path_probs)
        loss = -torch.mean(labels * (log_p_tau - log_pq) + (1 - labels) *  (log_q_tau - log_pq))

        # Calculate L2 loss on model parameters
        l2_loss = 0.01 * sum(self.l2_loss(p, torch.zeros_like(p)) for p in self.parameters())

        return loss + self.l2_loss_ratio * l2_loss

    def train(self, optimizer, obs, acs, obs_next, path_probs, labels):
        optimizer.zero_grad()
        loss = self.calculate_loss(obs, acs, obs_next, path_probs, labels)
        loss.backward()
        optimizer.step()
        return loss.item()

    def get_reward(self, obs, acs, obs_next, path_probs, discrim_score=False):
        with torch.no_grad():
            if discrim_score:
                log_q_tau, log_p_tau, log_pq, discrim_output = self(obs, acs, obs_next, path_probs)
                score = torch.log(discrim_output + 1e-20) - torch.log(1 - discrim_output + 1e-20)
            else:
                score = self.reward_net(obs)
        return score.cpu().numpy()

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
