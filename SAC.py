from collections import namedtuple

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from tensorboardX import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

Transition = namedtuple('Transition', ['s', 'a', 'r', 's_', 'd'])
min_Val = torch.tensor(1e-7).float().to(device)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, min_log_std=-10, max_log_std=2):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu_head = nn.Linear(256, int(action_dim))
        self.log_std_head = nn.Linear(256, int(action_dim))
        self.max_action = max_action

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_std_head = F.relu(self.log_std_head(x))
        log_std_head = torch.clamp(log_std_head, self.min_log_std, self.max_log_std)
        return mu, log_std_head


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Q(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Q, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, s, a):
        s = s.reshape(-1, self.state_dim)
        a = a.reshape(-1, self.action_dim)
        x = torch.cat((s, a), -1)  # combination s and a
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SAC():
    def __init__(self, lr, state_dim, action_dim, max_action, capacity=1000000, min_buffer_size=10000,
                 gradient_steps=1, batch_size=128, gamma=0.99, tau=0.005):
        super(SAC, self).__init__()

        self.capacity = capacity
        self.min_buffer_size = min_buffer_size
        self.gradient_steps = gradient_steps
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        self.policy_net = Actor(state_dim, max_action, action_dim).to(device)
        self.value_net = Critic(state_dim).to(device)
        self.Target_value_net = Critic(state_dim).to(device)
        self.Q_net1 = Q(state_dim, action_dim).to(device)
        self.Q_net2 = Q(state_dim, action_dim).to(device)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        self.Q1_optimizer = optim.Adam(self.Q_net1.parameters(), lr=lr)
        self.Q2_optimizer = optim.Adam(self.Q_net2.parameters(), lr=lr)

        self.replay_buffer = []
        self.num_transition = 0  # pointer of replay buffer
        self.num_training = 1
        self.writer = SummaryWriter('./exp-SAC_dual_Q_network')

        self.value_criterion = nn.MSELoss()
        self.Q1_criterion = nn.MSELoss()
        self.Q2_criterion = nn.MSELoss()

        for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        os.makedirs('./SAC_model/', exist_ok=True)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        mu, log_sigma = self.policy_net(state)
        sigma = torch.exp(log_sigma)
        dist = Normal(mu, sigma)
        z = dist.sample()
        action = torch.tanh(z).detach().cpu().numpy()
        return action  # return a scalar, float32

    def store(self, s, a, r, s_, d):
        index = self.num_transition % self.capacity
        transition = Transition(s, a, r, s_, d)
        if len(self.replay_buffer) < (self.capacity - 1):
            self.replay_buffer.append(transition)
        else:
            del self.replay_buffer[0:int(self.capacity / 5)]
        self.num_transition += 1

    def evaluate(self, state):
        batch_mu, batch_log_sigma = self.policy_net(state)
        batch_sigma = torch.exp(batch_log_sigma)
        dist = Normal(batch_mu, batch_sigma)
        noise = Normal(0, 1)

        z = noise.sample()
        action = torch.tanh(batch_mu + batch_sigma*z.to(device))
        log_prob = dist.log_prob(batch_mu + batch_sigma * z.to(device)) - torch.log(1 - action.pow(2) + min_Val)
        return action, log_prob, z, batch_mu, batch_log_sigma

    def update(self):
        if self.num_training % 500 == 0:
            print("Training ... {} times ".format(self.num_training))
        s = torch.tensor([t.s for t in self.replay_buffer]).float().to(device)
        a = torch.tensor([t.a for t in self.replay_buffer]).float().to(device)
        r = torch.tensor([t.r for t in self.replay_buffer]).float().to(device)
        s_ = torch.tensor([t.s_ for t in self.replay_buffer]).float().to(device)
        d = torch.tensor([t.d for t in self.replay_buffer]).float().to(device)

        for _ in range(self.gradient_steps):
            #for index in BatchSampler(SubsetRandomSampler(range(args.capacity)), args.batch_size, False):
            index = np.random.choice(range(len(self.replay_buffer)), self.batch_size, replace=False)
            bn_s = s[index]
            bn_a = a[index].reshape(-1, 1)
            bn_r = r[index].reshape(-1, 1)
            bn_s_ = s_[index]
            bn_d = d[index].reshape(-1, 1)

            target_value = self.Target_value_net(bn_s_)
            next_q_value = bn_r + (1 - bn_d) * self.gamma * target_value

            excepted_value = self.value_net(bn_s)
            excepted_Q1 = self.Q_net1(bn_s, bn_a)
            excepted_Q2 = self.Q_net2(bn_s, bn_a)
            sample_action, log_prob, z, batch_mu, batch_log_sigma = self.evaluate(bn_s)
            excepted_new_Q = torch.min(self.Q_net1(bn_s, sample_action), self.Q_net2(bn_s, sample_action))
            next_value = excepted_new_Q - log_prob

            # !!!Note that the actions are sampled according to the current policy,
            # instead of replay buffer. (From original paper)
            V_loss = self.value_criterion(excepted_value, next_value.detach()).mean()  # J_V

            # Dual Q net
            Q1_loss = self.Q1_criterion(excepted_Q1, next_q_value.detach()).mean() # J_Q
            Q2_loss = self.Q2_criterion(excepted_Q2, next_q_value.detach()).mean()

            pi_loss = (log_prob - excepted_new_Q).mean() # according to original paper

            self.writer.add_scalar('Loss/V_loss', V_loss, global_step=self.num_training)
            self.writer.add_scalar('Loss/Q1_loss', Q1_loss, global_step=self.num_training)
            self.writer.add_scalar('Loss/Q2_loss', Q2_loss, global_step=self.num_training)
            self.writer.add_scalar('Loss/policy_loss', pi_loss, global_step=self.num_training)

            # mini batch gradient descent
            self.value_optimizer.zero_grad()
            V_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.value_optimizer.step()

            self.Q1_optimizer.zero_grad()
            Q1_loss.backward(retain_graph = True)
            nn.utils.clip_grad_norm_(self.Q_net1.parameters(), 0.5)
            self.Q1_optimizer.step()

            self.Q2_optimizer.zero_grad()
            Q2_loss.backward(retain_graph = True)
            nn.utils.clip_grad_norm_(self.Q_net2.parameters(), 0.5)
            self.Q2_optimizer.step()

            self.policy_optimizer.zero_grad()
            pi_loss.backward(retain_graph = True)
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.policy_optimizer.step()

            # update target v net update
            for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
                target_param.data.copy_(target_param * (1 - self.tau) + param * self.tau)

            self.num_training += 1

    def save(self):
        torch.save(self.policy_net.state_dict(), './SAC_model/policy_net.pth')
        torch.save(self.value_net.state_dict(), './SAC_model/value_net.pth')
        torch.save(self.Q_net1.state_dict(), './SAC_model/Q_net1.pth')
        torch.save(self.Q_net2.state_dict(), './SAC_model/Q_net2.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self):
        self.policy_net.load_state_dict(torch.load('./SAC_model/policy_net.pth'))
        self.value_net.load_state_dict(torch.load( './SAC_model/value_net.pth'))
        self.Q_net1.load_state_dict(torch.load('./SAC_model/Q_net1.pth'))
        self.Q_net2.load_state_dict(torch.load('./SAC_model/Q_net2.pth'))
        print("model has been load")
