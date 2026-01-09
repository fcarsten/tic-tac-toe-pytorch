from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tic_tac_toe.Board import Board, BOARD_SIZE
from tic_tac_toe.DQNPlayer import DQNPlayer
from tic_tac_toe.Player import GameResult
from util import board_state_to_cnn_input


class DuelingFusion(nn.Module):
    """
    Groups the Dueling math operations into one named node.
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
    """

    def forward(self, value, advantage):
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


class DeepExpDoubleDuelQPlayerNetwork(nn.Module):
    def __init__(self, learning_rate: float, device: torch.device, beta: float = 0.00001):
        super().__init__()
        self.device = device
        self.learning_rate = learning_rate

        # 1. Group Convolutional Layers
        # This appears as a single "conv_block" node in TensorBoard
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # 2. Group Feature Extraction (Linear)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 3, 3)
            flattened_size = self.conv_block(dummy).view(1, -1).size(1)

        self.feature_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, BOARD_SIZE * 3 * 9),
            nn.ReLU()
        )

        # 3. Dueling Heads
        self.value_head = nn.Linear(BOARD_SIZE * 3 * 9, 1)
        self.advantage_head = nn.Linear(BOARD_SIZE * 3 * 9, BOARD_SIZE)

        # 4. Operations
        self.dueling_fusion = DuelingFusion()
        self.softmax = nn.Softmax(dim=1)

        self.to(self.device)
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, weight_decay=beta)

    def forward(self, x: torch.Tensor):
        # The forward pass becomes much cleaner
        x = self.conv_block(x)
        features = self.feature_block(x)

        value = self.value_head(features)
        advantage = self.advantage_head(features)

        q_values = self.dueling_fusion(value, advantage)
        probabilities = self.softmax(q_values)

        return q_values, probabilities

    def log_weights(self, writer=None, name=None, game_number=None):
        """Logs histograms of weights and biases for all layers."""
        if not writer:
            return

        for n, param in self.named_parameters():
            writer.add_histogram(f'{name}/Weights/{n}', param, game_number)
            if param.grad is not None:
                writer.add_histogram(f'{name}/Gradients/{n}', param.grad, game_number)


class DeepExpDoubleDuelQPlayer(DQNPlayer):
    """
    Tic-tac-toe player based on a Dueling Double Deep Q-Network.
    """

    def __init__(self, name: str = "DeepExpDoubleDuelQPlayer", reward_discount: float = 0.99, win_value: float = 10.0, draw_value: float = 0.0,
                 loss_value: float = -10.0, learning_rate: float = 0.01, training: bool = True,
                 random_move_prob: float = 0.9999, random_move_decrease: float = 0.9997, random_min_prob: float=0.0,
                 pre_training_games: int = 500, tau: float = 0.001, device: torch.device = torch.device("cpu"),
                 batch_size: int = 60, buffer_size: int = 10000,
                 ):
        super().__init__(name, reward_discount=reward_discount, win_value=win_value, draw_value=draw_value,
                         loss_value=loss_value, learning_rate=learning_rate, training=training, device=device,
                         random_move_prob=random_move_prob, random_move_decrease=random_move_decrease,
                         random_min_prob=random_min_prob, batch_size=batch_size, buffer_size=buffer_size)
        self.tau = tau
        self.pre_training_games = pre_training_games

    def log_start_state_q(self):
        """
        Logs the max Q-value of a generic empty board from a 'First Mover' perspective.
        This provides a consistent baseline even if the player is currently Naught.
        """
        if self.writer and self.training:
            b = Board()
            b.reset()

            with torch.no_grad():
                nn_input_tensor= self.board_state_to_nn_input(b.state)

                q_values = self.nn(nn_input_tensor.unsqueeze(0))[0]
                max_q = torch.max(q_values).item()
                # This should trend toward 0.0 as the model realizes the game is a draw
                self.writer.add_scalar(f'{self.name}/Baseline_Opening_Q', max_q, self.game_number)

    def _create_network(self, learning_rate):
        self.nn = DeepExpDoubleDuelQPlayerNetwork(learning_rate, self.device)

    def log_graph(self):
        if self.writer:
            # Create a dummy input matching the shape (Batch, 27)
            dummy_input = torch.zeros(1,3,3,3, device=self.device)
            self.writer.add_graph(self.nn, dummy_input)

    def board_state_to_nn_input(self, state: np.ndarray) -> torch.Tensor:
        return board_state_to_cnn_input(state, self.device, self.side)

    def soft_update(self):
        """ Updates target network using tau factor. """
        with torch.no_grad():
            for target_param, main_param in zip(self.target_nn.parameters(), self.nn.parameters()):
                target_param.mul_(1.0 - self.tau).add_(self.tau * main_param)

    def move(self, board: Board) -> Tuple[GameResult, bool]:
        """ Chooses a move based on epsilon-greedy exploration. """
        self.move_step += 1

        nn_input_tensor = self.board_state_to_nn_input(board.state)
        if self.training:
            self.state_log.append(nn_input_tensor.detach())

        if self.training and (self.game_number < self.pre_training_games or np.random.rand() < self.random_move_prob):
            move = board.random_empty_spot()
        else:
            with torch.no_grad():

                nn_input_tensor_unsqueezed = nn_input_tensor.unsqueeze(0)
                q_values, _ = self.nn(nn_input_tensor_unsqueezed)

                if self.training and self.writer and self.move_step % 100 == 0:
                    # self.log_q_heatmap(q_values, self.move_step)
                    self.writer.add_histogram(f'{self.name}/Action_Q_Distribution', q_values, self.move_step)
                    max_q = float(torch.max(q_values).item())
                    avg_q = float(torch.mean(q_values).item())
                    self.writer.add_scalar(f'{self.name}/Max_Q_Value', max_q, self.move_step)
                    self.writer.add_scalar(f'{self.name}/Average_Q_In_Game', avg_q, self.move_step)
                    self.writer.add_scalar(f'{self.name}/Move_Confidence', max_q - avg_q, self.move_step)

                q_values = q_values.squeeze(0)

                illegal_mask = torch.tensor(
                    [not board.is_legal(i) for i in range(BOARD_SIZE)],
                    dtype=torch.bool,
                    device=q_values.device
                )

                q_values = q_values.masked_fill(illegal_mask, -torch.inf)
                move = int(torch.argmax(q_values).item())

        if self.training:
            self.action_log.append(move)

        _, res, finished = board.move(move, self.side)
        return res, finished

    def final_result(self, result: GameResult):
        if not self.training: return

        """ Handles training at the end of a game. """
        reward = self.get_reward_value(result)

        self.add_game_to_replay_buffer(reward, result)

        if self.training and (self.game_number > self.pre_training_games):
            self._train_from_replay()
            self.random_move_prob *= self.random_move_decrease
            self.soft_update()

    def add_game_to_replay_buffer(self, reward: float, result: GameResult):
        """ Adds experience tuples to the replay buffer using tensor states. """
        game_length = len(self.action_log)

        for i in range(game_length - 1):
            self.memory.push(
                [self.state_log[i], self.action_log[i], self.state_log[i + 1], 0.0],
                result.value - 1
            )

        self.memory.push(
            [self.state_log[-1], self.action_log[-1], None, reward],
            result.value - 1
        )

    def _train_from_replay(self):
        self.nn.train()

        samples = self.memory.sample(self.batch_size)

        # Convert once
        states = torch.stack([s[0] for s in samples])

        actions = torch.as_tensor(
            [s[1] for s in samples],
            dtype=torch.int64,
            device=self.device
        )

        rewards = torch.as_tensor(
            [s[3] for s in samples],
            dtype=torch.float32,
            device=self.device
        )

        non_final_mask = torch.tensor(
            [s[2] is not None for s in samples],
            dtype=torch.bool,
            device=self.device
        )

        # Current Q(s,a)
        current_q_values, _ = self.nn(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values
        target_q_values = rewards.clone()

        with torch.no_grad():
            if non_final_mask.any():
                next_states = torch.stack(
                    [s[2] for s in samples if s[2] is not None]
                )

                # Double DQN
                next_q_main, _ = self.nn(next_states)
                best_actions = next_q_main.argmax(dim=1, keepdim=True)

                next_q_target, _ = self.target_nn(next_states)
                max_next_q = next_q_target.gather(1, best_actions).squeeze(1)

                target_q_values[non_final_mask] += self.reward_discount * max_next_q

        loss = F.mse_loss(current_q_values, target_q_values)

        self.nn.optimizer.zero_grad()
        loss.backward()
        self.nn.optimizer.step()

        if self.writer:
            if self.game_number % 500 == 0:
                self.nn.log_weights(self.writer, self.name, self.game_number)

            self.writer.add_scalar(f'{self.name}/Training_Loss', loss.item(), self.game_number)
            self.writer.add_scalar(
                f'{self.name}/Random_Move_Probability',
                self.random_move_prob,
                self.game_number
            )
