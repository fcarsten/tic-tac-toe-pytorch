from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tic_tac_toe.DoubleDQNPlayer import DoubleDQNPlayer
from util import board_state_to_cnn_input


class ConvDuelingQNetwork(nn.Module):
    def __init__(self, learning_rate: float, device: torch.device, input_shape=(3, 3, 3)):
        super().__init__()
        self.device = device

        # 1. Convolutional Feature Extractor
        # Assuming input_shape is (C, H, W)
        self.features = nn.Sequential(OrderedDict([
            ('Conv1', nn.Conv2d(3, 128, kernel_size=3, padding=1)),
            ('ReLU1', nn.ReLU()),
            ('Conv2', nn.Conv2d(128, 128, kernel_size=3, padding=1)),
            ('ReLU2', nn.ReLU()),
            ('Conv3', nn.Conv2d(128, 64, kernel_size=3, padding=1)),
            ('ReLU3', nn.ReLU()),
            ('Flatten', nn.Flatten())
        ]))

        # The "Shared" dense layer before the Dueling split
        self.shared_dense = nn.Sequential(
            nn.Linear(64 * 3 * 3, 243),  # Larger capacity
            nn.ReLU()
        )
        # # Calculate the size of the flattened features for the linear layers
        # # This helper ensures we don't have to hard-code the linear input size
        # with torch.no_grad():
        #     dummy_input = torch.zeros(1, *input_shape)
        #     n_flatten = self.shared_dense(dummy_input).shape[1]

        # 2. Value stream (State Value V(s))
        self.value_stream = nn.Sequential(OrderedDict([
            ('Value_Linear', nn.Linear(243, 128)),
            ('Value_ReLU', nn.ReLU()),
            ('Value_Output', nn.Linear(128, 1))
        ]))

        # 3. Advantage stream (Action Advantage A(s, a))
        self.advantage_stream = nn.Sequential(OrderedDict([
            ('Advantage_Linear', nn.Linear(243, 128)),
            ('Advantage_ReLU', nn.ReLU()),
            ('Advantage_Output', nn.Linear(128, 9))  # 9 actions
        ]))

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        # For explicit initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

        self.loss_fn = nn.SmoothL1Loss()
        self.to(device)

    def forward(self, x):
        # Ensure input is in (Batch, C, H, W) format
        if x.dim() == 3:
            x = x.unsqueeze(0)

        feat = self.features(x)
        shared = self.shared_dense(feat)

        value = self.value_stream(shared)
        advantage = self.advantage_stream(shared)

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

    def get_regularization_loss(self, l1_beta=1e-5, l2_beta=1e-5):
        l1_loss = 0
        l2_loss = 0
        for name, param in self.named_parameters():
            if 'weight' in name:
                l1_loss += torch.sum(torch.abs(param))
                l2_loss += torch.sum(param ** 2)
        return (l1_beta * l1_loss) + (l2_beta * l2_loss)

    def train_batch(self, inputs, expected_q, actions, writer=None, name=None, game_number=None):
        self.optimizer.zero_grad()

        # Get all Q-values, then pick only the ones for the actions taken
        q_pred_all = self.forward(inputs)
        q_pred = q_pred_all.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        td_loss = self.loss_fn(q_pred, expected_q)

        # 2. L1L2 Regularization Loss
        reg_loss = self.get_regularization_loss(l1_beta=1e-5, l2_beta=1e-5)

        # 3. Total Loss
        total_loss = td_loss + reg_loss
        total_loss.backward()

        # Log Loss to TensorBoard
        if writer:
            writer.add_scalar(f'{name}/TD_Loss', td_loss, game_number)
            writer.add_scalar(f'{name}/Reg_Loss', reg_loss, game_number)

            if game_number % 100 == 0:
                self.log_weights(writer, name, game_number)


        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        self.optimizer.step()
        return total_loss.item()



class ConvDuelingDoubleDQNPlayer(DoubleDQNPlayer):

    # def __init__(self, name: str = "ConvDuelingDoubleDQNPlayer", random_move_decrease: float = 0.9995,
    #              learning_rate =5e-5, **kwargs):

    def __init__(self, name: str = "ConvDuelingDoubleDQNPlayer", reward_discount: float = 0.9,
                 random_move_decrease: float = 0.9999, target_update_freq: int = 3000, learning_rate =1e-5, **kwargs):
        super().__init__(name, reward_discount= reward_discount, target_update_freq=target_update_freq,
                         random_move_decrease=random_move_decrease, learning_rate =learning_rate, **kwargs)

    def _create_network(self, learning_rate) -> nn.Module:
        return ConvDuelingQNetwork(learning_rate, self.device)

    def board_state_to_nn_input(self, state: np.ndarray) -> torch.Tensor:
        return board_state_to_cnn_input(state, self.device, self.side)

    def log_graph(self):
        if self.writer:
            # Create a dummy input matching the shape (Batch, 27)
            dummy_input = torch.zeros(1,3,3,3, device=self.device)
            self.writer.add_graph(self.nn, dummy_input)


    def _train_from_replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
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
#        expected_q = rewards_v + (self.reward_discount * next_q_max)
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