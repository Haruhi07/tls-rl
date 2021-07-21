import torch
import torch.nn as nn
import torch.nn.functional as F


class Q_network(nn.Module):
    def __init__(self, state_dim):
        super(Q_network, self).__init__()
        self.fc1 = nn.Linear(state_dim, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            nn.init.normal_(m.weight.data, 0, 0.1)
            nn.init.constant_(m.bias.data, 0.01)

class Critic():
    def __init__(self, state_dim, device, args):
        self.state_dim =state_dim

        self.device = device
        self.network = Q_network(state_dim=self.state_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=args.lr)
        self.loss_func = nn.MSELoss()

        self.time_step = 0
        self.epsilon = args.epsilon
        self.gamma = args.gamma

    def train_Q_network(self, state, reward, next_state):
        s = torch.FloatTensor(state).to(self.device)
        s_ = torch.FloatTensor(next_state).to(self.device)

        # Forward Propagation
        v = self.network.forward(s)
        v_ = self.network.forward(s_)

        # Backward Propagation
        loss_q = self.loss_func(reward + self.gamma * v_, v)
        self.optimizer.zero_grad()
        loss_q.backward()
        self.optimizer.step()

        with torch.no_grad():
            td_error = reward + GAMMA * v_ - v
        return td_error