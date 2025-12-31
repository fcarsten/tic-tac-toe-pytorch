import random
from collections import OrderedDict

import numpy as np
from tic_tac_toe.DoubleDQNPlayer import DoubleDQNPlayer

import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

from util import board_state_to_cnn_input


class ConvDuelingQNetwork(nn.Module):
    def __init__(self, learning_rate: float, device: torch.device, input_shape=(3, 3, 3)):
        super().__init__()
        self.device = device

        # 1. Convolutional Feature Extractor
        # Assuming input_shape is (C, H, W)
        self.features = nn.Sequential(OrderedDict([
            ('Conv1', nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1)),
            ('ReLU1', nn.ReLU()),
            ('Conv2', nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)),
            ('ReLU2', nn.ReLU()),
            ('Flatten', nn.Flatten())
        ]))

        # Calculate the size of the flattened features for the linear layers
        # This helper ensures we don't have to hard-code the linear input size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            n_flatten = self.features(dummy_input).shape[1]

        # 2. Value stream (State Value V(s))
        self.value_stream = nn.Sequential(OrderedDict([
            ('Value_Linear', nn.Linear(n_flatten, 128)),
            ('Value_ReLU', nn.ReLU()),
            ('Value_Output', nn.Linear(128, 1))
        ]))

        # 3. Advantage stream (Action Advantage A(s, a))
        self.advantage_stream = nn.Sequential(OrderedDict([
            ('Advantage_Linear', nn.Linear(n_flatten, 128)),
            ('Advantage_ReLU', nn.ReLU()),
            ('Advantage_Output', nn.Linear(128, 9))  # 9 actions
        ]))

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.to(device)

    def forward(self, x):
        # Ensure input is in (Batch, C, H, W) format
        if x.dim() == 3:
            x = x.unsqueeze(0)

        feat = self.features(x)
        value = self.value_stream(feat)
        advantage = self.advantage_stream(feat)

        # Dueling Logic: Q(s,a) = V(s) + (A(s,a) - Mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

    def log_weights(self, writer=None, name=None, game_number=None):
        """Logs histograms of weights and biases for all layers."""
        if not writer:
            return

        for n, param in self.named_parameters():
            writer.add_histogram(f'{name}/Weights/{n}', param, game_number)
            if param.grad is not None:
                writer.add_histogram(f'{name}/Gradients/{n}', param.grad, game_number)

    def train_batch(self, inputs, expected_q, actions, writer=None, name=None, game_number=None):
        self.optimizer.zero_grad()

        # Get all Q-values, then pick only the ones for the actions taken
        q_pred_all = self.forward(inputs)
        q_pred = q_pred_all.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        loss = self.loss_fn(q_pred, expected_q)
        loss.backward()

        # Log Loss to TensorBoard
        if writer:
            writer.add_scalar(f'{name}/Training_Loss', loss, game_number)

            if game_number % 100 == 0:
                self.log_weights(writer, name, game_number)

        self.optimizer.step()
        return loss.item()



class ConvDuelingDoubleDQNPlayer(DoubleDQNPlayer):
    def __init__(self, name: str = "ConvDuelingDoubleDQNPlayer", **kwargs):
        super().__init__(name, **kwargs)
        # Override networks with Dueling architecture

    def _create_network(self, learning_rate) -> nn.Module:
        return ConvDuelingQNetwork(learning_rate, self.device)

    def board_state_to_nn_input(self, state: np.ndarray) -> torch.Tensor:
        return board_state_to_cnn_input(state, self.device, self.side)

    def log_graph(self):
        if self.writer:
            # Create a dummy input matching the shape (Batch, 27)
            dummy_input = torch.zeros(1,3,3,3, device=self.device)
            self.writer.add_graph(self.nn.features, dummy_input)


    def _train_from_replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # torch.stack is perfect here: it turns [(3,3,3), (3,3,3)...] into (Batch, 3, 3, 3)
        states_v = torch.stack(states).to(self.device)
        actions_v = torch.tensor(actions, device=self.device, dtype=torch.long)
        rewards_v = torch.tensor(rewards, device=self.device, dtype=torch.float)
        dones_v = torch.tensor(dones, device=self.device, dtype=torch.bool)

        # 1. Get current Q values
        current_qs = self.nn(states_v)
        # Extract the Q-values for the specific actions taken
        current_q_values = current_qs.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

        # 2. Get Next-State values from TARGET Network
        with torch.no_grad():
            next_q_max = torch.zeros(self.batch_size, device=self.device)

            # Filter out 'None' or terminal states
            non_final_mask = torch.tensor([ns is not None for ns in next_states], device=self.device)

            if non_final_mask.any():
                # Filter and stack next_states
                non_final_next_states = torch.stack([ns for ns in next_states if ns is not None]).to(self.device)
                target_q_estimates = self.target_nn(non_final_next_states)
                next_q_max[non_final_mask] = target_q_estimates.max(dim=1)[0]

        # 3. Bellman Equation: r + gamma * max_Q(s') * (1 - done)
        # Using 'not done' logic ensures terminal states have a future value of 0
        expected_q = rewards_v + (self.reward_discount * next_q_max * (~dones_v).float())

        # 4. Perform optimization
        # Note: We pass the gathered current_q_values and expected_q to the optimizer
        # This is more standard than cloning the whole matrix
        loss = self.nn.train_batch(states_v, expected_q, actions_v,
                                   writer=self.writer, name=self.name,
                                   game_number=self.game_number)
        # Log Loss to TensorBoard
        if self.writer:
            self.writer.add_scalar(f'{self.name}/Training_Loss', loss, self.game_number)

        # Logging and Target Sync...
        if self.move_step % self.target_update_freq == 0:
            self._update_target_network()