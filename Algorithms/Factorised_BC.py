# Import modules
import torch
import torch.nn.functional as F
import torch.nn as nn

# Define policy for each sub-action space
class Policy_BC(nn.Module):
    def __init__(self, state_dim, action_dim, sub_action_dim, hidden_dim=512):
        super(Policy_BC, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.action = nn.Linear(hidden_dim, action_dim * sub_action_dim)

        self.action_dim = action_dim
        self.sub_action_dim = sub_action_dim

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = self.action(a)
        a = a.reshape(-1, self.action_dim, self.sub_action_dim)
        a = F.log_softmax(a, dim=-1)

        return a

class Agent():
    def __init__(self, state_dim, action_dim, sub_action_dim=3, hidden_dim=512, batch_size=256, lr=3e-4, device="cpu"):

        self.policy_bc = Policy_BC(state_dim, action_dim, sub_action_dim, hidden_dim).to(device)
        self.policy_bc_optimizer = torch.optim.Adam(self.policy_bc.parameters(), lr=lr)

        self.policy_bc_loss_history = []
        self.policy_bc_loss_val_history = []

        self.batch_size = batch_size
        self.device = device
        self.action_dim = action_dim

    def choose_action(self, state):
        # Greedy sub-actions
        with torch.no_grad():
            state = torch.Tensor(state.reshape(1, -1)).to(self.device)
            action_logits = self.policy_bc(state)
            action = torch.argmax(action_logits, -1)

        return action.cpu().numpy().flatten()

    def train(self, replay_buffer, iterations=1):
        for it in range(iterations):
            # Sample batch from replay buffer
            state, action, _, _, _ = replay_buffer.sample(self.batch_size)
            actions = action.long().reshape(self.batch_size, self.action_dim, -1)

            ## Behavioural cloning ##
            action_logits = self.policy_bc(state)
            policy_bc_loss = -action_logits.gather(2, actions).mean()

            self.policy_bc_loss_history.append(policy_bc_loss.item())
            self.policy_bc_optimizer.zero_grad()
            policy_bc_loss.backward()
            self.policy_bc_optimizer.step()

    def validate(self, replay_buffer, iterations=1):
        for it in range(iterations):
            # Sample batch from replay buffer
            state, action, _, _, _ = replay_buffer.sample(self.batch_size)
            actions = action.long().reshape(self.batch_size, self.action_dim, -1)

            ## Behavioural cloning ##
            action_logits = self.policy_bc(state)
            policy_bc_loss = -action_logits.gather(2, actions).mean()

            self.policy_bc_loss_val_history.append(policy_bc_loss.item())
