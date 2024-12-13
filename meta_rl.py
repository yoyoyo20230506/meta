import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import deque
import random

class MetaRLModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MetaRLModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class MetaRL:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.policy = MetaRLModel(state_dim, action_dim).to('cuda')
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.memory = deque(maxlen=10000)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.policy.fc3.out_features - 1)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to('cuda')
        with torch.no_grad():
            q_values = self.policy(state)
        return torch.argmax(q_values).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self, batch_size):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(states, dtype=torch.float32).to('cuda')
        actions = torch.tensor(actions, dtype=torch.int64).to('cuda')
        rewards = torch.tensor(rewards, dtype=torch.float32).to('cuda')
        next_states = torch.tensor(next_states, dtype=torch.float32).to('cuda')
        dones = torch.tensor(dones, dtype=torch.float32).to('cuda')

        q_values = self.policy(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q_values = self.policy(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()