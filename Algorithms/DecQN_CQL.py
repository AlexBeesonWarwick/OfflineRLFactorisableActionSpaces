# Import modules
import numpy as np
import random
import copy
import math

import torch
import torch.nn.functional as F
import torch.nn as nn

# VectorizedLinear taken from - https://github.com/tinkoff-ai/CORL/blob/main/algorithms/sac_n.py
class VectorizedLinear(nn.Module):
    def __init__(self, in_features, out_features, ensemble_size):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        # default pytorch init for nn.Linear module
        for layer in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[layer], a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input: [ensemble_size, batch_size, input_size]
        # weight: [ensemble_size, input_size, out_size]
        # out: [ensemble_size, batch_size, out_size]
        return x @ self.weight + self.bias

class VectorizedDQN(nn.Module):
    def __init__(self, state_dim, action_dim, sub_action_dim, num_critics=2, hidden_dim=512):
        super(VectorizedDQN, self).__init__()

        self.l1 = VectorizedLinear(state_dim, hidden_dim, num_critics)
        self.l2 = VectorizedLinear(hidden_dim, hidden_dim, num_critics)
        self.qs = VectorizedLinear(hidden_dim, action_dim * sub_action_dim, num_critics)

        self.num_critics = num_critics
        self.action_dim = action_dim
        self.sub_action_dim = sub_action_dim

    def forward(self, state):
        state = state.unsqueeze(0).repeat_interleave(self.num_critics, dim=0)

        q_values = F.relu(self.l1(state))
        q_values = F.relu(self.l2(q_values))
        q_values = self.qs(q_values)

        return q_values.reshape(self.num_critics, -1, self.action_dim, self.sub_action_dim)


class Agent():
    def __init__(self, state_dim, action_dim, sub_action_dim=3, num_critics=2,
                 batch_size=256, lr=3e-4, gamma=0.99, tau=0.005, device="cpu"):

        # Initialisation
        self.dqn = VectorizedDQN(state_dim, action_dim, sub_action_dim, num_critics).to(device)
        self.dqn_target = copy.deepcopy(self.dqn)
        self.dqn_optimizer = torch.optim.Adam(self.dqn.parameters(), lr=lr)

        # Record losses
        self.dqn_loss_history = []

        # Set remaining parameters
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.num_critics = num_critics
        self.batch_size = batch_size

    def choose_action(self, state):
        # Greedy sub-actions based on utility values
        with torch.no_grad():
            state = torch.Tensor(state.reshape(1, -1)).to(self.device)
            u_vals = self.dqn(state).mean(0)
            action = torch.argmax(u_vals, -1)

        return action.cpu().numpy().flatten()

    def train(self, replay_buffer, iterations=1, alpha_prime=1):

        for it in range(iterations):
            # Sample batch from replay buffer
            state, action, reward, next_state, done = replay_buffer.sample(self.batch_size)

            with torch.no_grad():
                next_u_vals = self.dqn_target(next_state)
                next_u_vals_mean = next_u_vals.mean(0)
                next_u_vals_max = next_u_vals_mean.max(-1)[0]
                next_q_vals_max = next_u_vals_max.mean(-1)
                q_hat = reward + self.gamma * (1 - done) * next_q_vals_max

            u_vals = self.dqn(state)
            actions = action.long().reshape(self.batch_size, self.action_dim, -1)
            u_vals_action = torch.zeros(self.num_critics, self.batch_size, self.action_dim).to(self.device)
            for k in range(self.num_critics):
                u_vals_action[k] = u_vals[k].gather(2, actions).squeeze(-1)
            q_vals_action = u_vals_action.mean(-1)
            dqn_loss = F.huber_loss(q_vals_action, q_hat)  # Ignore warning - we want broadcasting

            # CQL
            u_vals_mean = u_vals.mean(0)
            logsumexp = torch.logsumexp(u_vals_mean, -1)
            u_vals_action_mean = u_vals_action.mean(0)
            p = (logsumexp - u_vals_action_mean).mean()  # This takes into account mean across sub-action dims
            dqn_loss += (alpha_prime * p)

            self.dqn_loss_history.append(dqn_loss.item())
            self.dqn_optimizer.zero_grad()
            dqn_loss.backward()
            self.dqn_optimizer.step()

            ### Polyak target network updates ###
            for target_param, param in zip(self.dqn_target.parameters(), self.dqn.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def train_sum(self, replay_buffer, iterations=1, alpha_prime=1):

        for it in range(iterations):
            # Sample batch from replay buffer
            state, action, reward, next_state, done = replay_buffer.sample(self.batch_size)

            with torch.no_grad():
                next_u_vals = self.dqn_target(next_state)
                next_u_vals_mean = next_u_vals.mean(0)
                next_u_vals_max = next_u_vals_mean.max(-1)[0]
                next_q_vals_max = next_u_vals_max.sum(-1)
                q_hat = reward + self.gamma * (1 - done) * next_q_vals_max

            u_vals = self.dqn(state)
            actions = action.long().reshape(self.batch_size, self.action_dim, -1)
            u_vals_action = torch.zeros(self.num_critics, self.batch_size, self.action_dim).to(self.device)
            for k in range(self.num_critics):
                u_vals_action[k] = u_vals[k].gather(2, actions).squeeze(-1)
            q_vals_action = u_vals_action.sum(-1)
            dqn_loss = F.huber_loss(q_vals_action, q_hat)  # Ignore warning - we want broadcasting

            # CQL
            u_vals_mean = u_vals.mean(0)
            logsumexp = torch.logsumexp(u_vals_mean, -1)
            u_vals_action_mean = u_vals_action.mean(0)
            p = (logsumexp.sum(-1) - u_vals_action_mean.sum(-1)).mean()
            dqn_loss += (alpha_prime * p)

            self.dqn_loss_history.append(dqn_loss.item())
            self.dqn_optimizer.zero_grad()
            dqn_loss.backward()
            self.dqn_optimizer.step()

            ### Polyak target network updates ###
            for target_param, param in zip(self.dqn_target.parameters(), self.dqn.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
