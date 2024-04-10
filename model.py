import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import random
from collections import deque

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def push(self, event):
        self.memory.append(event)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

class QTrainer:
    def __init__(self, model, lr, gamma, target_update, memory_size, batch_size):
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.model = model
        self.target_model = Linear_QNet(input_size=model.linear1.in_features, hidden_size=model.linear1.out_features, output_size=model.linear2.out_features)
        self.update_target_model()
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.memory = ReplayMemory(memory_size)
        self.target_update = target_update
        self.update_count = 0

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def train_step(self):
        if len(self.memory.memory) < self.batch_size:
            return
        batch = self.memory.sample(self.batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*batch)

        batch_state = torch.tensor(batch_state, dtype=torch.float)
        batch_next_state = torch.tensor(batch_next_state, dtype=torch.float)
        batch_action = torch.tensor(batch_action, dtype=torch.long)
        batch_reward = torch.tensor(batch_reward, dtype=torch.float)
        batch_done = torch.tensor(batch_done, dtype=torch.bool)

        current_q_values = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(batch_next_state).max(1)[0].detach()
        expected_q_values = batch_reward + self.gamma * next_q_values * (~batch_done)

        loss = self.criterion(current_q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.update_target_model()



