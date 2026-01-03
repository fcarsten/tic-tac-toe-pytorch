from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from tic_tac_toe.ReplayMemory import ReplayMemory
from tic_tac_toe.Board import Board, BOARD_SIZE, EMPTY, NAUGHT, CROSS, GameResult
from tic_tac_toe.Player import Player


class PolicyGradientNetwork(nn.Module):
    """
    The Policy Gradient PyTorch Network
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(PolicyGradientNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        # Initialization logic similar to original VarianceScaling
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # Returns logits


class DirectPolicyAgent(Player):
    def state_to_tensor(self, state: np.ndarray) -> torch.Tensor:
        """
        Converts board state to a bit-array Tensor (size 27) on the selected device.
        Minimizes future conversions by doing this once per move.
        """
        # Create the 3-channel bit representation
        res = np.array([(state == self.side).astype(float),
                        (state == Board.other_side(self.side)).astype(float),
                        (state == EMPTY).astype(float)])
        return torch.from_numpy(res.reshape(-1)).float().to(self.device)

    def log_graph(self):
        if self.writer:
            # Create a dummy input matching the shape (Batch, 27)
            dummy_input = torch.zeros((1, BOARD_SIZE * 3), device=self.device)
            self.writer.add_graph(self.nn, dummy_input)

    def __init__(self, name: str, gamma: float = 0.1, learning_rate: float = 0.001,
                 win_value: float = 1.0, loss_value: float = 0.0, draw_value: float = 0.5,
                 training: bool = True, random_move_probability: float = 0.9,
                 beta: float = 0.000001, random_move_decrease: float = 0.9997,
                 pre_training_games: int = 500, batch_size: int = 60,
                 buffer_size: int = 3000, writer: SummaryWriter = None, device: torch.device = None):
        super().__init__()  #
        self.name=name
        self.device = device if device is not None else torch.device('cpu')
        self.writer = writer
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.win_value = win_value
        self.draw_value = draw_value
        self.loss_value = loss_value
        self.training = training
        self.batch_size = batch_size
        self.beta = beta
        self.random_move_probability = random_move_probability
        self.random_move_decrease = random_move_decrease
        self.pre_training_games = pre_training_games

        self.game_counter = 0
        self.side = None

        # Logs now store Tensors directly to avoid repeated conversions
        self.board_position_log: List[torch.Tensor] = []
        self.action_log: List[int] = []

        # Network setup
        input_dim = BOARD_SIZE * 3
        hidden_dim = BOARD_SIZE * 3 * 9
        output_dim = 9

        self.nn = PolicyGradientNetwork(input_dim, hidden_dim, output_dim).to(self.device)
        self.optimizer = optim.Adam(self.nn.parameters(), lr=self.learning_rate)

        # Integration with ReplayMemory
        self.memory = ReplayMemory(buffer_size=buffer_size)

    def new_game(self, side: int):
        self.side = side  #
        self.board_position_log = []
        self.action_log = []

    def get_valid_probs(self, state_tensor: torch.Tensor, board: Board) -> torch.Tensor:
        """
        Computes valid move probabilities using the device-resident tensor.
        """
        self.nn.eval()
        with torch.no_grad():
            # Ensure input has batch dimension
            logits = self.nn(state_tensor.unsqueeze(0))
            probs = F.softmax(logits, dim=-1).squeeze(0)

        # Mask illegal moves
        mask = torch.zeros(9, device=self.device)
        for i in range(9):
            if board.is_legal(i):
                mask[i] = 1.0

        masked_probs = probs * mask
        sum_probs = masked_probs.sum()

        if sum_probs > 0:
            return masked_probs / sum_probs
        return mask / mask.sum()  # Fallback to uniform over legal moves

    def move(self, board: Board) -> Tuple[GameResult, bool]:
        # Convert state to tensor immediately and store it
        state_tensor = self.state_to_tensor(board.state)
        self.board_position_log.append(state_tensor)

        if self.training and (self.game_counter < self.pre_training_games):
            move = board.random_empty_spot()  #
        else:
            probs = self.get_valid_probs(state_tensor, board)
            # Sample move on CPU for numpy compatibility in Board class
            move = np.random.choice(9, p=probs.cpu().numpy())

        self.action_log.append(move)
        _, res, finished = board.move(move, self.side)  #
        return res, finished

    def calculate_rewards(self, final_reward: float, length: int) -> torch.Tensor:
        """Returns rewards as a Tensor on the correct device."""
        discounted_r = np.zeros(length)
        running_add = final_reward
        for t in reversed(range(0, length)):
            discounted_r[t] = running_add
            running_add = running_add * self.gamma
        return torch.tensor(discounted_r, dtype=torch.float32, device=self.device)

    def final_result(self, result: GameResult):
        # Determine reward and buffer index (0=win, 1=loss, 2=draw)
        if (result == GameResult.NAUGHT_WIN and self.side == NAUGHT) or \
                (result == GameResult.CROSS_WIN and self.side == CROSS):
            final_reward, buffer_idx = self.win_value, 0
        elif (result == GameResult.NAUGHT_WIN and self.side == CROSS) or \
                (result == GameResult.CROSS_WIN and self.side == NAUGHT):
            final_reward, buffer_idx = self.loss_value, 1
        else:
            final_reward, buffer_idx = self.draw_value, 2

        self.game_counter += 1
        rewards = self.calculate_rewards(final_reward, len(self.action_log))

        # Push pre-computed tensors to memory
        for i in range(len(self.action_log)):
            experience = (self.board_position_log[i], self.action_log[i], rewards[i])
            self.memory.push(experience, buffer_idx=buffer_idx)

        if self.training and (self.game_counter > self.pre_training_games):
            self.train_network()
            self.random_move_probability *= self.random_move_decrease

    def train_network(self):
        self.nn.train()
        train_batch = self.memory.sample(self.batch_size)  #
        if not train_batch: return

        # Since elements are already tensors on the device, we stack them efficiently
        states = torch.stack([x[0] for x in train_batch])
        actions = torch.tensor([x[1] for x in train_batch], device=self.device).unsqueeze(1)
        rewards = torch.stack([x[2] for x in train_batch])

        self.optimizer.zero_grad()

        logits = self.nn(states)
        probs = F.softmax(logits, dim=-1)

        # Select probabilities for actions taken
        responsible_outputs = probs.gather(1, actions).squeeze()

        # Policy loss + L2 regularization
        policy_loss = -torch.mean(torch.log(responsible_outputs + 1e-9) * rewards)

        l2_reg = sum(p.pow(2.0).sum() for p in self.nn.parameters())
        total_loss = policy_loss + (self.beta * l2_reg)

        total_loss.backward()
        self.optimizer.step()