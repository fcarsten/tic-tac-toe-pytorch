from collections import OrderedDict

import torch
import torch.optim as optim
from torch import nn as nn

from tic_tac_toe.DoubleDQNPlayer import DoubleDQNPlayer

class DuelingQNetwork(nn.Module):
    def __init__(self, learning_rate: float, device: torch.device):
        super().__init__()
        self.device = device

        # Shared feature extractor with labels
        self.feature_layer = nn.Sequential(OrderedDict([
            ('Feature_Linear', nn.Linear(27, 128)),
            ('Feature_ReLU', nn.ReLU())
        ]))

        # Value stream with labels
        self.value_stream = nn.Sequential(OrderedDict([
            ('Value_Hidden', nn.Linear(128, 64)),
            ('Value_ReLU', nn.ReLU()),
            ('Value_Output', nn.Linear(64, 1))
        ]))

        # Advantage stream with labels
        self.advantage_stream = nn.Sequential(OrderedDict([
            ('Advantage_Hidden', nn.Linear(128, 64)),
            ('Advantage_ReLU', nn.ReLU()),
            ('Advantage_Output', nn.Linear(64, 9))
        ]))

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.to(device)

    def forward(self, x):
        # 1. Ensure input is 2D (Batch Size, Features)
        # If x is [27], it becomes [1, 27]
        if x.dim() == 1:
            x = x.unsqueeze(0)

        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Combining logic
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

    def train_batch(self, inputs, targets, writer=None, name = None, game_number=None):
        self.optimizer.zero_grad()
        q_pred = self.forward(inputs)
        loss = self.loss_fn(q_pred, targets)
        loss.backward()
        # Log Loss to TensorBoard
        if self.writer:
            self.writer.add_scalar(f'{name}/Training_Loss', loss, game_number)

            if game_number % 100 == 0:
                self.log_weights(writer, name, game_number)

        self.optimizer.step()
        return loss.item()  # Return loss for logging

class DuelingDoubleDQNPlayer(DoubleDQNPlayer):
    def __init__(self, name: str = "DuelingDoubleDQNPlayer", **kwargs):
        super().__init__(name, **kwargs)
        # Override networks with Dueling architecture

    def _create_network(self, learning_rate):
        self.nn= DuelingQNetwork(learning_rate, self.device)

