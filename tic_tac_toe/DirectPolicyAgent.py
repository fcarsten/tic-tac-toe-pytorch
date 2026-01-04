from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tic_tac_toe.Board import Board, BOARD_SIZE, GameResult
from tic_tac_toe.SimpleNNQPlayer import NNQPlayer


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

    def log_weights(self, writer=None, name=None, game_number=None):
        """Logs histograms of weights and biases for all layers."""
        if not writer:
            return

        for n, param in self.named_parameters():
            writer.add_histogram(f'{name}/Weights/{n}', param, game_number)
            if param.grad is not None:
                writer.add_histogram(f'{name}/Gradients/{n}', param.grad, game_number)

class DirectPolicyAgent(NNQPlayer):

    def __init__(self, name: str = "DirectPolicyAgent", reward_discount: float = 1.0, learning_rate: float = 0.0001,
                 win_value: float = 1.0, loss_value: float = 0.0, draw_value: float = 0.5,
                 training: bool = True, random_move_prob: float = 0.9,
                 beta: float = 0.000001, random_move_decrease: float = 0.9997,
                 device: torch.device = torch.device("cpu")):
        super().__init__(name, reward_discount, win_value, draw_value, loss_value, learning_rate,
                         training, device)
        self.writer = None

        self.training = training
        self.beta = beta
        self.random_move_prob = random_move_prob
        self.random_move_decrease = random_move_decrease
        self.running_baseline = 0.5
        # Network setup

        self.optimizer = optim.Adam(self.nn.parameters(), lr=learning_rate)

    def _create_network(self, learning_rate):
        input_dim = BOARD_SIZE * 3
        hidden_dim = BOARD_SIZE * 3 * 9
        output_dim = 9
        self.nn= PolicyGradientNetwork(input_dim, hidden_dim, output_dim).to(self.device)

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
        self.move_step += 1

        # Convert state to tensor immediately and store it
        state_tensor = self.board_state_to_nn_input(board.state)
        self.state_log.append(state_tensor)

        probs = self.get_valid_probs(state_tensor, board)
        if self.training and self.writer and self.move_step % 100 == 0:
            move_entropy = -torch.sum(probs * torch.log(probs + 1e-9))

            self.writer.add_scalar(f'{self.name}/Move_Entropy', move_entropy, self.move_step)
            self.writer.add_scalar(f'{self.name}/Move_Confidence', torch.max(probs).item(), self.move_step)
            if self.move_step % 500 == 0:
                self.writer.add_histogram(f'{self.name}/Move_prob_Distribution', probs, self.move_step)

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
            running_add = running_add * self.reward_discount
        return torch.tensor(discounted_r, dtype=torch.float32, device=self.device)

    def final_result(self, result: GameResult):
        if self.training:
            self.train_network(result)
            self.random_move_prob *= self.random_move_decrease

    def train_network(self, game_result: GameResult):
        if not self.training: return

        # Since elements are already tensors on the device, we stack them efficiently
        states = torch.stack(self.state_log)
        actions = torch.tensor(self.action_log, device=self.device).unsqueeze(1)

        final_reward = self.get_reward_value(game_result)
        self.running_baseline = 0.99 * self.running_baseline + 0.01 * final_reward
        advantage = final_reward - self.running_baseline

        rewards = self.calculate_rewards(advantage, len(self.action_log))

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

        # Track Gradient Norm before step
        grad_norm = sum(p.grad.detach().data.norm(2).item() ** 2 for p in self.nn.parameters()) ** 0.5

        self.optimizer.step()

        if self.writer:
            if self.game_number % 500 == 0:
                self.nn.log_weights(self.writer, self.name, self.game_number)

            self.writer.add_scalar(f'{self.name}/Training_Loss', total_loss.item(), self.game_number)
            self.writer.add_scalar(f'{self.name}/Running_Baseline', self.running_baseline, self.game_number)
            self.writer.add_scalar(f'{self.name}/Policy_Loss', policy_loss.item(), self.game_number)
            self.writer.add_scalar(f'{self.name}/L2_Regularization_Loss', (self.beta * l2_reg).item(), self.game_number)
            self.writer.add_scalar(f'{self.name}/Gradient_Norm', grad_norm, self.game_number)
            self.writer.add_scalar(
                f'{self.name}/Random_Move_Probability',
                self.random_move_prob,
                self.game_number
            )
