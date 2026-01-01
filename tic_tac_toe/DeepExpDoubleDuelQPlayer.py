import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import random
from typing import List, Tuple

from tic_tac_toe.Board import Board, BOARD_SIZE, EMPTY, CROSS, NAUGHT
from tic_tac_toe.Player import Player, GameResult


class QNetwork(nn.Module):
    """
    PyTorch implementation of the Dueling Q-Network for Tic Tac Toe.
    """

    def __init__(self, learning_rate: float, device: torch.device, beta: float = 0.00001):
        super(QNetwork, self).__init__()
        self.device = device

        # Architecture matching the original Conv layers
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        flattened_size = 64 * 3 * 3
        self.fc1 = nn.Linear(flattened_size, BOARD_SIZE * 3 * 9)

        # Dueling Heads
        self.value_head = nn.Linear(BOARD_SIZE * 3 * 9, 1)
        self.advantage_head = nn.Linear(BOARD_SIZE * 3 * 9, BOARD_SIZE)

        # Move model to device before creating the optimizer
        self.to(self.device)

        # Optimizer with L2 regularization (weight_decay) via beta
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, weight_decay=beta)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))

        value = self.value_head(x)
        advantage = self.advantage_head(x)

        # Dueling DQN formula
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        probabilities = F.softmax(q_values, dim=1)

        return q_values, probabilities

    def log_weights(self, writer=None, name=None, game_number=None):
        """Logs histograms of weights and biases for all layers."""
        if not writer:
            return

        for n, param in self.named_parameters():
            writer.add_histogram(f'{name}/Weights/{n}', param, game_number)
            if param.grad is not None:
                writer.add_histogram(f'{name}/Gradients/{n}', param.grad, game_number)


class ReplayBuffer:
    """ Manages the Experience Replay buffer. """

    def __init__(self, buffer_size=3000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience: list):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, size) -> list:
        size = min(len(self.buffer), size)
        return random.sample(self.buffer, size)


class DeepExpDoubleDuelQPlayer(Player):
    """
    Tic Tac Toe player based on a Dueling Double Deep Q-Network.
    """

    def __init__(self, name: str, reward_discount: float = 0.99, win_value: float = 10.0, draw_value: float = 0.0,
                 loss_value: float = -10.0, learning_rate: float = 0.01, training: bool = True,
                 random_move_prob: float = 0.9999, random_move_decrease: float = 0.9997, batch_size=60,
                 pre_training_games: int = 500, tau: float = 0.001, device: torch.device = torch.device("cpu"),
                 writer: SummaryWriter=None):
        super().__init__()
        self.name = name
        self.device = device

        self.tau = tau
        self.batch_size = batch_size
        self.reward_discount = reward_discount
        self.win_value = win_value
        self.draw_value = draw_value
        self.loss_value = loss_value

        self.training = training
        self.random_move_prob = random_move_prob
        self.random_move_decrease = random_move_decrease
        self.game_counter = 0
        self.pre_training_games = pre_training_games

        # Networks initialized on the specific device
        self.q_net = QNetwork(learning_rate, self.device)
        self.target_net = QNetwork(learning_rate, self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        # Replay Buffers
        self.replay_buffer_win = ReplayBuffer()
        self.replay_buffer_loss = ReplayBuffer()
        self.replay_buffer_draw = ReplayBuffer()

        self.board_position_log = []
        self.action_log = []
        self.side = None
        self.writer = writer

    def log_graph(self):
        if self.writer:
            # Create a dummy input matching the shape (Batch, 27)
            dummy_input = torch.zeros(1,3,3,3, device=self.device)
            self.writer.add_graph(self.q_net, dummy_input)

    def board_state_to_nn_input(self, state: np.ndarray) -> np.ndarray:
        """ Converts state to (Channels, Height, Width) for PyTorch. """
        res = np.array([(state == self.side).astype(float),
                        (state == Board.other_side(self.side)).astype(float),
                        (state == EMPTY).astype(float)])
        return res.reshape(3, 3, 3)

    def soft_update(self):
        """ Updates target network using tau factor. """
        for target_param, main_param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * main_param.data + (1.0 - self.tau) * target_param.data)

    def new_game(self, side: int):
        """ Resets game logs for a new match. """
        self.side = side
        self.board_position_log = []
        self.action_log = []

    def move(self, board: Board) -> Tuple[GameResult, bool]:
        """ Chooses a move based on epsilon-greedy exploration. """
        self.board_position_log.append(board.state.copy())

        if self.training and (self.game_counter < self.pre_training_games or np.random.rand() < self.random_move_prob):
            move = board.random_empty_spot()
        else:
            self.q_net.eval()
            with torch.no_grad():
                # Ensure input tensor is on the correct device
                nn_input = torch.as_tensor(self.board_state_to_nn_input(board.state),
                                           dtype=torch.float32, device=self.device).unsqueeze(0)
                q_values, _ = self.q_net(nn_input)
                q_values = q_values.cpu().numpy()[0]

                # Filter illegal moves
                for i in range(BOARD_SIZE):
                    if not board.is_legal(i):
                        q_values[i] = -np.inf
                move = np.argmax(q_values)

        self.action_log.append(move)
        _, res, finished = board.move(move, self.side)
        return res, finished

    def final_result(self, result: GameResult):
        """ Handles training at the end of a game. """
        self.game_counter += 1

        if (result == GameResult.NAUGHT_WIN and self.side == NAUGHT) or \
                (result == GameResult.CROSS_WIN and self.side == CROSS):
            reward = self.win_value
        elif (result == GameResult.NAUGHT_WIN and self.side == CROSS) or \
                (result == GameResult.CROSS_WIN and self.side == NAUGHT):
            reward = self.loss_value
        else:
            reward = self.draw_value

        self.add_game_to_replay_buffer(reward)

        if self.training and (self.game_counter > self.pre_training_games):
            self.train_step()
            self.random_move_prob *= self.random_move_decrease
            self.soft_update()

    def add_game_to_replay_buffer(self, reward: float):
        """ Adds experience tuples to the appropriate buffer. """
        game_length = len(self.action_log)
        if reward == self.win_value:
            buffer = self.replay_buffer_win
        elif reward == self.loss_value:
            buffer = self.replay_buffer_loss
        else:
            buffer = self.replay_buffer_draw

        for i in range(game_length - 1):
            buffer.add([self.board_position_log[i], self.action_log[i], self.board_position_log[i + 1], 0.0])

        buffer.add([self.board_position_log[-1], self.action_log[-1], None, reward])

    def train_step(self):
        """ Performs one Gradient Descent step on a batch. """
        self.q_net.train()

        batch_third = self.batch_size // 3
        samples = self.replay_buffer_win.sample(batch_third) + \
                  self.replay_buffer_loss.sample(batch_third) + \
                  self.replay_buffer_draw.sample(batch_third)

        # Prepare batch tensors on the correct device
        states = torch.as_tensor([self.board_state_to_nn_input(s[0]) for s in samples],
                                 dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([s[1] for s in samples],
                                  dtype=torch.int64, device=self.device)
        next_states_mask = torch.tensor([s[2] is not None for s in samples],
                                        dtype=torch.bool, device=self.device)
        rewards = torch.as_tensor([s[3] for s in samples],
                                  dtype=torch.float32, device=self.device)

        # Current Q Values
        current_q_values, _ = self.q_net(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q Values
        with torch.no_grad():
            target_q_values = rewards.clone()
            if next_states_mask.any():
                next_states = torch.as_tensor([self.board_state_to_nn_input(s[2]) for s in samples if s[2] is not None],
                                              dtype=torch.float32, device=self.device)

                # Double DQN logic: Main Net selects action, Target Net evaluates it
                next_q_main, _ = self.q_net(next_states)
                best_actions = next_q_main.argmax(1, keepdim=True)

                next_q_target, _ = self.target_net(next_states)
                max_next_q = next_q_target.gather(1, best_actions).squeeze(1)

                target_q_values[next_states_mask] += self.reward_discount * max_next_q

        # Optimization step
        loss = F.mse_loss(current_q_values, target_q_values)
        self.q_net.optimizer.zero_grad()
        loss.backward()
        self.q_net.optimizer.step()

        if self.writer:
            if self.game_counter % 100 == 0:
                self.q_net.log_weights(self.writer, self.name, self.game_counter)

            self.writer.add_scalar(f'{self.name}/Training_Loss', loss.item(), self.game_counter)
            self.writer.add_scalar(f'{self.name}/Random_Move_Probability', self.random_move_prob, self.game_counter)