from collections import OrderedDict

import torch
import torch.optim as optim
from torch import nn as nn

from tic_tac_toe.Board import BOARD_SIZE
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
        self.loss_fn = nn.MSELoss()
        self.to(device)

    def log_weights(self, writer=None, name=None, game_number=None):
        """Logs histograms of weights and biases for all layers."""
        if not writer:
            return

        for n, param in self.feature_layer.named_parameters():
            writer.add_histogram(f'{name}/Weights/{n}', param, game_number)
            if param.grad is not None:
                writer.add_histogram(f'{name}/Gradients/{n}', param.grad, game_number)

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

        self.log_weights(writer, name, game_number)

        self.optimizer.step()
        return loss.item()  # Return loss for logging

class DuelingDoubleDQNPlayer(DoubleDQNPlayer):
    def __init__(self, name: str = "DuelingDoubleDQNPlayer", **kwargs):
        super().__init__(name, **kwargs)
        # Override networks with Dueling architecture

    def _create_network(self, learning_rate) -> nn.Module:
        return DuelingQNetwork(learning_rate, self.device)


    def log_graph(self):
        if not self.writer: return

        # A wrapper module specifically for TensorBoard to label the nodes
        class TrainingFlow(nn.Module):
            def __init__(self, policy_net, loss_fn):
                super().__init__()
                self.policy_net = policy_net
                self.loss_fn = loss_fn

            def forward(self, board_state, target_q):
                # These names will appear as labels in the graph
                predicted_q = self.policy_net(board_state)
                loss = self.loss_fn(predicted_q, target_q)
                return loss

        # Instantiate wrapper
        wrapper = TrainingFlow(self.nn, self.nn.loss_fn)

        # Dummy inputs for tracing
        dummy_state = torch.zeros((1, 27), device=self.device)
        dummy_target = torch.zeros((1, 9), device=self.device)

        # Log to TensorBoard
        self.writer.add_graph(wrapper, (dummy_state, dummy_target))