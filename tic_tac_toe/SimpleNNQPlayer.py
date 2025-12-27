import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tic_tac_toe.Board import Board, BOARD_SIZE, EMPTY, CROSS, NAUGHT
from tic_tac_toe.Player import Player, GameResult

class QNetwork(nn.Module):
    """
    PyTorch version of the Q-network.
    """

    def __init__(self, learning_rate: float, device: torch.device):
        super().__init__()
        self.device = device
        self.model = nn.Sequential(
            nn.Linear(BOARD_SIZE * 3, BOARD_SIZE * 3 * 9),
            nn.ReLU(),
            nn.Linear(BOARD_SIZE * 3 * 9, BOARD_SIZE)
        ).to(device)

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
                 training: bool = True, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device
        self.name = name
        self.reward_discount = reward_discount
        self.win_value = win_value
        self.draw_value = draw_value
        self.loss_value = loss_value
        self.training = training

        self.side = None
        self.state_log = []
        self.action_log = []
        self.next_value_log = []
        self.q_log = []

        self.nn = QNetwork(learning_rate, device)

    def new_game(self, side: int):
        self.side = side
        self.state_log.clear()
        self.action_log.clear()
        self.next_value_log.clear()
        self.q_log.clear()

    def move(self, board: Board):
        state_tensor = self.board_state_to_nn_input(board.state)
        self.state_log.append(state_tensor)

        # Forward pass (training version with gradients)
        q_training = self.nn(state_tensor.unsqueeze(0))[0]
        self.q_log.append(q_training)

        # Forward pass without gradients for action selection
        with torch.no_grad():
            qvalues = q_training.clone()
            probs = torch.softmax(qvalues, dim=0)

            # OPTIMIZATION 3: Vectorized masking.
            # Instead of a python loop calling is_legal(i) 9 times, we use a boolean mask.
            # board.state != EMPTY implies the spot is occupied (illegal).
            occupied_mask = torch.tensor(board.state != EMPTY, device=self.device, dtype=torch.bool)

            # Set occupied spots to -1.0 so argmax avoids them
            probs[occupied_mask] = -1.0

            move = int(torch.argmax(probs).item())

        # Log the next state's Q max for Q-learning update
        if self.action_log:
            self.next_value_log.append(qvalues[move].item())

        self.action_log.append(move)

        _, res, finished = board.move(move, self.side)
        return res, finished

    def final_result(self, result: GameResult):
        if not self.training:
            return

        if (result == GameResult.NAUGHT_WIN and self.side == NAUGHT) or \
           (result == GameResult.CROSS_WIN and self.side == CROSS):
            reward = self.win_value
        elif result == GameResult.DRAW:
            reward = self.draw_value
        else:
            reward = self.loss_value

        self.next_value_log.append(reward)

        # Convert stored data into tensors
        # Note: Since we are already on CPU, this is just a stack operation
        states = torch.stack(self.state_log)
        q_pred = torch.stack(self.q_log)
        targets = q_pred.clone().detach()

        # Apply Q-learning update: batch
        actions = torch.tensor(self.action_log, device=self.device)
        next_vals = torch.tensor(self.next_value_log, device=self.device)

        # targets[i, action_i] = gamma * next_value[i]
        targets[torch.arange(len(actions)), actions] = \
            self.reward_discount * next_vals

        self.nn.train_batch(states, targets)
