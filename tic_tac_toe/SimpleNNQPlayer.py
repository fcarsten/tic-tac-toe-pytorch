from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter  # Added

from tic_tac_toe.Board import Board, BOARD_SIZE, EMPTY, CROSS, NAUGHT
from tic_tac_toe.Player import Player, GameResult


class QNetwork(nn.Module):
    def __init__(self, learning_rate: float, device: torch.device):
        super().__init__()
        self.device = device

        self.model = nn.Sequential(OrderedDict([
            ('Input_Layer', nn.Linear(BOARD_SIZE * 3, BOARD_SIZE * 3 * 9)),
            ('Activation_1', nn.ReLU()),
            ('Output_Layer', nn.Linear(BOARD_SIZE * 3 * 9, BOARD_SIZE))
        ])).to(device)

        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def train_batch(self, inputs, targets):
        self.optimizer.zero_grad()
        q_pred = self.forward(inputs)
        loss = self.loss_fn(q_pred, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()  # Return loss for logging


class NNQPlayer(Player):
    """
    Neural network Q-learning Tic Tac Toe player rewritten for PyTorch.
    """

    def board_state_to_nn_input(self, state: np.ndarray) -> torch.Tensor:
        # Convert numpy state directly to tensor once
        t_state = torch.as_tensor(state, device=self.device)

        other_side = Board.other_side(self.side)

        # Create masks directly in PyTorch
        is_me = (t_state == self.side).float()
        is_other = (t_state == other_side).float()
        is_empty = (t_state == EMPTY).float()

        # Concatenate into the expected (27,) shape
        return torch.cat([is_me, is_other, is_empty])

    def __init__(self, name: str, reward_discount: float = 0.95,
                 win_value: float = 1.0, draw_value: float = 0.0,
                 loss_value: float = -1.0, learning_rate: float = 0.01,
                 training: bool = True, device: torch.device = torch.device("cpu"),
                 writer: SummaryWriter = None):  # Added writer
        super().__init__()
        self.device = device
        self.name = name
        self.writer = writer
        self.reward_discount = reward_discount
        self.win_value = win_value
        self.draw_value = draw_value
        self.loss_value = loss_value
        self.training = training

        self.training_steps = 0  # To track global steps in TensorBoard
        self.side = None
        self.state_log = []
        self.action_log = []
        self.next_value_log = []
        self.q_log = []

        self.nn = QNetwork(learning_rate, device)


        if self.writer:
            # Create a dummy input matching the shape (Batch, 27)
            dummy_input = torch.zeros((1, BOARD_SIZE * 3), device=self.device)
            self.writer.add_graph(self.nn.model, dummy_input)

    def new_game(self, side: int):
        self.side = side
        self.state_log.clear()
        self.action_log.clear()
        self.next_value_log.clear()
        self.q_log.clear()

        self.log_start_state_q()
    import matplotlib.pyplot as plt
    import io
    import torch

    # ... inside NNQPlayer class ...

    def log_start_state_q(self):
        """
        Logs the max Q-value of a generic empty board from a 'First Mover' perspective.
        This provides a consistent baseline even if the player is currently Naught.
        """
        if self.writer and self.training:
            # Create a 'First Mover' perspective empty board:
            # [9 zeros for 'me', 9 zeros for 'other', 9 ones for 'empty']
            me = torch.zeros(BOARD_SIZE, device=self.device)
            other = torch.zeros(BOARD_SIZE, device=self.device)
            empty = torch.ones(BOARD_SIZE, device=self.device)
            fixed_input = torch.cat([me, other, empty]).unsqueeze(0)

            with torch.no_grad():
                q_values = self.nn(fixed_input)[0]
                max_q = torch.max(q_values).item()
                # This should trend toward 0.0 as the model realizes the game is a draw
                self.writer.add_scalar(f'{self.name}/Baseline_Opening_Q', max_q, self.training_steps)

    def log_q_heatmap(self, q_values, step):
        """Logs a 3x3 heatmap of Q-values to TensorBoard."""
        if not self.writer or self.training_steps % 500 != 0:
            return

        # 1. Prepare the data: Reshape the 9 Q-values into a 3x3 grid
        # Use detach().cpu() to ensure the tensor is off the GPU/graph
        heatmap_data = q_values.detach().cpu().numpy().reshape(3, 3)

        # 2. Create the Matplotlib figure
        fig, ax = plt.subplots(figsize=(4, 4))
        # vmin/vmax ensures the color scale stays consistent as the agent learns
        im = ax.imshow(heatmap_data, cmap='viridis', vmin=-1.0, vmax=1.0)
        plt.colorbar(im)
        ax.set_title(f"Q-Values Heatmap (Step {step})")

        # 3. Use the built-in add_figure method
        # This automatically handles buffer conversion and resizing
        self.writer.add_figure(f'{self.name}/Q_Heatmap', fig, global_step=step)

        # Optional: Close the figure to free up memory
        plt.close(fig)

    def log_weights(self):
        """Logs histograms of weights and biases for all layers."""
        if not self.writer:
            return

        for name, param in self.nn.model.named_parameters():
            self.writer.add_histogram(f'{self.name}/Weights/{name}', param, self.training_steps)
            if param.grad is not None:
                self.writer.add_histogram(f'{self.name}/Gradients/{name}', param.grad, self.training_steps)

    def move(self, board: Board):
        state_tensor = self.board_state_to_nn_input(board.state)
        self.state_log.append(state_tensor)

        q_training = self.nn(state_tensor.unsqueeze(0))[0]
        self.q_log.append(q_training)

        # Log Average Max Q-value for this game's states
        if self.writer and self.training and self.training_steps % 100 == 0:
            self.log_q_heatmap(q_training, self.training_steps)
            self.writer.add_histogram(f'{self.name}/Action_Q_Distribution', q_training, self.training_steps)
            max_q = torch.max(q_training).item()
            self.writer.add_scalar(f'{self.name}/Max_Q_Value', max_q, self.training_steps)
            avg_q = torch.mean(q_training).item()
            self.writer.add_scalar(f'{self.name}/Average_Q_In_Game', avg_q, self.training_steps)
            self.writer.add_scalar(f'{self.name}/Move_Confidence', max_q - avg_q, self.training_steps)

        with torch.no_grad():
            qvalues = q_training.clone()
            probs = torch.softmax(qvalues, dim=0)
            occupied_mask = torch.tensor(board.state != EMPTY, device=self.device, dtype=torch.bool)
            probs[occupied_mask] = -1.0
            move = int(torch.argmax(probs).item())

        if self.action_log:
            self.next_value_log.append(qvalues[move].item())

        self.action_log.append(move)
        _, res, finished = board.move(move, self.side)
        return res, finished

    def final_result(self, result: GameResult):
        if not self.training:
            return

        # Determine reward logic ...
        if (result == GameResult.NAUGHT_WIN and self.side == NAUGHT) or \
                (result == GameResult.CROSS_WIN and self.side == CROSS):
            reward = self.win_value
        elif result == GameResult.DRAW:
            reward = self.draw_value
        else:
            reward = self.loss_value

        self.next_value_log.append(reward)

        states = torch.stack(self.state_log)
        q_pred = torch.stack(self.q_log)
        targets = q_pred.clone().detach()

        actions = torch.tensor(self.action_log, device=self.device)
        next_vals = torch.tensor(self.next_value_log, device=self.device)

        targets[torch.arange(len(actions)), actions] = self.reward_discount * next_vals

        loss = self.nn.train_batch(states, targets)

        # Log Loss to TensorBoard
        if self.writer:
            self.writer.add_scalar(f'{self.name}/Loss', loss, self.training_steps)

        self.training_steps += 1