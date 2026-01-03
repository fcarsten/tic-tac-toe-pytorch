from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tic_tac_toe.Board import Board, BOARD_SIZE, EMPTY, CROSS, NAUGHT
from tic_tac_toe.Player import Player, GameResult
from util import board_state_to_one_hot_nn_input


class QNetwork(nn.Module):
    def __init__(self, learning_rate: float, device: torch.device):
        super().__init__()
        self.device = device

        self.feature_layer = nn.Sequential(OrderedDict([
            ('Input_Layer', nn.Linear(BOARD_SIZE * 3, BOARD_SIZE * 3 * 9)),
            ('Activation_1', nn.ReLU()),
            ('Output_Layer', nn.Linear(BOARD_SIZE * 3 * 9, BOARD_SIZE))
        ])).to(device)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()

    def forward(self, x):
        return self.feature_layer(x)

    def log_weights(self, writer=None, name=None, game_number=None):
        """Logs histograms of weights and biases for all layers."""
        if not writer:
            return

        for n, param in self.named_parameters():
            writer.add_histogram(f'{name}/Weights/{n}', param, game_number)
            if param.grad is not None:
                writer.add_histogram(f'{name}/Gradients/{n}', param.grad, game_number)

    def train_batch(self, inputs, targets, writer=None, name = None, game_number=None):
        self.optimizer.zero_grad()
        q_pred = self.forward(inputs)
        loss = self.loss_fn(q_pred, targets)
        loss.backward()
        # Log Loss to TensorBoard
        if writer:
            writer.add_scalar(f'{name}/Training_Loss', loss, game_number)

        if self.game_number % 100 == 0:
            self.log_weights(writer, name, game_number)

        self.optimizer.step()
        return loss.item()  # Return loss for logging


class NNQPlayer(Player):
    """
    Neural network Q-learning Tic Tac Toe player rewritten for PyTorch.
    """

    def board_state_to_nn_input(self, state: np.ndarray) -> torch.Tensor:
        return board_state_to_one_hot_nn_input(state, self.device, self.side)

    def __init__(self, name: str = "NNQPlayer", reward_discount: float = 0.95,
                 win_value: float = 1.0, draw_value: float = 0.0,
                 loss_value: float = -1.0, learning_rate: float = 0.001,
                 training: bool = True, device: torch.device = torch.device("cpu")):  # Added writer
        super().__init__()
        self.device = device
        self.name = name
        self.writer = None
        self.reward_discount = reward_discount
        self.win_value = win_value
        self.draw_value = draw_value
        self.loss_value = loss_value
        self.training = training

        self.game_number = 0
        self.move_step = 0

        self.side = None
        self.state_log = []
        self.action_log = []
        self.next_value_log = []
        self.q_log = []

        self._create_network(learning_rate)

    def _create_network(self, learning_rate):
        self.nn= QNetwork(learning_rate, self.device)

    def log_graph(self):
        if self.writer:
            # Create a dummy input matching the shape (Batch, 27)
            dummy_input = torch.zeros((1, BOARD_SIZE * 3), device=self.device)
            self.writer.add_graph(self.nn, dummy_input)

    def new_game(self, side: int):
        self.game_number = self.game_number + 1

        self.side = side
        self.state_log.clear()
        self.action_log.clear()
        self.next_value_log.clear()
        self.q_log.clear()

        self.log_start_state_q()

    def log_start_state_q(self):
        """
        Logs the max Q-value of a generic empty board from a 'First Mover' perspective.
        This provides a consistent baseline even if the player is currently Naught.
        """
        if self.writer and self.training:
            b = Board()
            b.reset()

            with torch.no_grad():
                q_values = self.nn(self.board_state_to_nn_input(b.state))[0]
                max_q = torch.max(q_values).item()
                # This should trend toward 0.0 as the model realizes the game is a draw
                self.writer.add_scalar(f'{self.name}/Baseline_Opening_Q', max_q, self.game_number)

    def log_q_heatmap(self, q_values, step):
        """Logs a 3x3 heatmap of Q-values to TensorBoard."""
        if not self.writer or step % 500 != 0:
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

    def move(self, board: Board):
        self.move_step += 1

        state_tensor = self.board_state_to_nn_input(board.state)
        if self.training:
            self.state_log.append(state_tensor)

        # Inference only, no graph
        with torch.no_grad():
            # Detached copy stored on the training device
            q_values = self.nn(state_tensor.unsqueeze(0))[0].detach().clone()
            if self.training:
                self.q_log.append(q_values)

                # Move-level logging (use q_values for logging)
                if self.writer and self.move_step % 100 == 0:
                    # self.log_q_heatmap(q_values, self.move_step)
                    self.writer.add_histogram(f'{self.name}/Action_Q_Distribution', q_values, self.move_step)
                    max_q = float(torch.max(q_values).item())
                    avg_q = float(torch.mean(q_values).item())
                    self.writer.add_scalar(f'{self.name}/Max_Q_Value', max_q, self.move_step)
                    self.writer.add_scalar(f'{self.name}/Average_Q_In_Game', avg_q, self.move_step)
                    self.writer.add_scalar(f'{self.name}/Move_Confidence', max_q - avg_q, self.move_step)

            occupied_mask = torch.as_tensor(board.state != EMPTY, device=self.device, dtype=torch.bool)
            logits = q_values.clone()
            logits[occupied_mask] = -float('inf')
            move = int(torch.argmax(logits).item())

            if self.training:
                if self.action_log: # Skip on first move, ie when action_log is empty
                    self.next_value_log.append(torch.max(q_values).item())
                self.action_log.append(move)

            _, res, finished = board.move(move, self.side)
            return res, finished

    def get_reward_value(self, result):
        if (result == GameResult.NAUGHT_WIN and self.side == NAUGHT) or \
                (result == GameResult.CROSS_WIN and self.side == CROSS):
            return self.win_value
        elif result == GameResult.DRAW:
            return self.draw_value

        return self.loss_value


    def final_result(self, result: GameResult):
        if not self.training:
            return

        reward = self.get_reward_value(result)

        self.next_value_log.append(reward)

        states = torch.stack(self.state_log)
        q_pred = torch.stack(self.q_log)
        targets = q_pred.clone().detach()

        actions = torch.tensor(self.action_log, device=self.device)
        next_vals = torch.tensor(self.next_value_log, device=self.device)

        # 3. Vectorized Bellman Update
        # Identify indices for intermediate moves and the terminal move
        num_moves = len(actions)
        row_indices = torch.arange(num_moves, device=self.device)

        # Calculate discounted values for all (will be wrong for the last one)
        bellman_targets = self.reward_discount * next_vals

        # Override the very last target with the pure reward (no discount)
        bellman_targets[-1] = reward

        # Update only the Q-values for the actions that were actually taken
        targets[row_indices, actions] = bellman_targets

        loss = self.nn.train_batch(states, targets, writer=self.writer,
                                      name=self.name, game_number=self.game_number)

        # Log Loss to TensorBoard
        if self.writer:
            self.writer.add_scalar(f'{self.name}/Training_Loss', loss, self.game_number)

