from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tic_tac_toe.Board import Board, BOARD_SIZE, GameResult
from tic_tac_toe.ReplayMemory import ReplayMemory
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

    def __init__(self, name: str = "DirectPolicyAgent", reward_discount: float = 0.1, learning_rate: float = 0.0005,
                 win_value: float = 1.0, loss_value: float = 0.0, draw_value: float = 0.5,
                 training: bool = True, random_move_prob: float = 0.9,
                 beta: float = 0.000002, random_move_decrease: float = 0.9997,
                 pre_training_games: int = 500, batch_size: int = 60,
                 buffer_size: int = 3000,
                 device: torch.device = torch.device("cpu")):
        super().__init__(name, reward_discount, win_value, draw_value, loss_value, learning_rate,
                         training, device)
        self.writer = None

        self.training = training
        self.batch_size = batch_size
        self.beta = beta
        self.random_move_prob = random_move_prob
        self.random_move_decrease = random_move_decrease
        self.pre_training_games = pre_training_games

        # Network setup

        self.optimizer = optim.Adam(self.nn.parameters(), lr=learning_rate)

        # Integration with ReplayMemory
        self.memory = ReplayMemory(buffer_size=buffer_size)

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

        if self.training and (self.game_number < self.pre_training_games):
            move = board.random_empty_spot()  #
        else:
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
        final_reward = self.get_reward_value(result)

        rewards = self.calculate_rewards(final_reward, len(self.action_log))

        # Push pre-computed tensors to memory
        for i in range(len(self.action_log)):
            experience = (self.state_log[i], self.action_log[i], rewards[i])
            self.memory.push(experience, result.value-1)

        if self.training and (self.game_number > self.pre_training_games):
            self.train_network()
            self.random_move_prob *= self.random_move_decrease

    def train_network(self):
        self.nn.train()
        train_batch = self.memory.sample(self.batch_size)  #
        if not train_batch: return

        # Prepare tensors
        states = torch.stack([x[0] for x in train_batch])
        actions = torch.tensor([x[1] for x in train_batch], device=self.device)  # Shape: [batch]
        rewards = torch.stack([x[2] for x in train_batch])  # Shape: [batch]

        self.optimizer.zero_grad()

        # 1. Get raw logits from the model
        logits = self.nn(states)

        # 2. Compute Log-Softmax (Numerically Stable)
        log_probs = F.log_softmax(logits, dim=-1)

        # 3. Use NLLLoss with reduction='none' to get individual log-probabilities
        # NLLLoss(log_probs, actions) = -log_probs[actions]
        criterion = nn.NLLLoss(reduction='none')

        # 4. Multiply by rewards (Policy Gradient formula)
        # Note: criterion already applies the negative sign
        policy_loss = (criterion(log_probs, actions) * rewards).mean()

        # 5. Add L2 Regularization (Beta * sum of squared weights)
        l2_reg = sum(p.pow(2.0).sum() for p in self.nn.parameters())
        total_loss = policy_loss + (self.beta * l2_reg)

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.nn.parameters(), max_norm=1.0)
        # Track Gradient Norm before step
        grad_norm = sum(p.grad.detach().data.norm(2).item() ** 2 for p in self.nn.parameters()) ** 0.5

        self.optimizer.step()

        if self.writer:
            if self.game_number % 500 == 0:
                self.nn.log_weights(self.writer, self.name, self.game_number)

            self.writer.add_scalar(f'{self.name}/Training_Loss', total_loss.item(), self.game_number)
            self.writer.add_scalar(f'{self.name}/Policy_Loss', policy_loss.item(), self.game_number)
            self.writer.add_scalar(f'{self.name}/L2_Regularization_Loss', (self.beta * l2_reg).item(), self.game_number)
            self.writer.add_scalar(f'{self.name}/Gradient_Norm', grad_norm, self.game_number)
            self.writer.add_scalar(
                f'{self.name}/Random_Move_Probability',
                self.random_move_prob,
                self.game_number
            )
