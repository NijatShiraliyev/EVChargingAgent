from gymnasium import Env
import torch
import torch.nn as nn
import torch.optim as optim
import random

from utils import ReplayBuffer

# 1. THE NETWORK
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        # : Define 3 Linear Layers. 
        # Input size = 11 (State), Output size = 6 (Actions)
        # Use ReLU activations between layers.
        self.f1 = nn.Linear(input_dim, 128)
        self.f2 = nn.Linear(128, 128)
        self.f3 = nn.Linear(128, output_dim)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.f1(x))
        x = self.relu(self.f2(x))
        return self.f3(x)
        # : Implement forward pass
        

# 2. THE AGENT CLASS
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.policy_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        
        # : Copy weights from policy_net to target_net
        # self.target_net.load_state_dict(...)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.buffer = ReplayBuffer()
        
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def select_action(self, state):
        # : Implement Epsilon-Greedy Logic
        # Generate random number 0-1.
        # If < self.epsilon: Return random action (Exploration)
        # Else: Return argmax(policy_net(state)) (Exploitation)
        sample = random.random()
        if sample < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
                return self.policy_net(state_tensor).argmax().item()

    def train_step(self, batch_size=64):
        if self.buffer.size() < batch_size:
            return
        
        # 1. Sample batch from buffer
        batch = self.buffer.sample(batch_size)
        # : Convert batch to PyTorch Tensors (states, actions, rewards...)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        # 2. Calculate Q_Targets (The "Ground Truth" guess)
        # Logic: Reward + Gamma * max(Target_Net(next_state))
        # Note: If done=True, the target is just Reward.
        q_targets = rewards + (0.99 * self.target_net(next_states).max(1)[0].unsqueeze(1) * (1 - dones))
        # 3. Calculate Q_Expected (The "Prediction")
        # Logic: Policy_Net(state).gather(action)
        q_expected = self.policy_net(states).gather(1, actions)
        # 4. Calculate Loss
        loss = nn.MSELoss()(q_expected, q_targets)
        
        # 5. Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 6. Decay Epsilon
        # self.epsilon = max(min, epsilon * decay)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)