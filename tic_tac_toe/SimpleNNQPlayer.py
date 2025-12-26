import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tic_tac_toe.Board import Board, BOARD_SIZE, EMPTY, CROSS, NAUGHT
from tic_tac_toe.Player import Player, GameResult


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    """
    PyTorch version of the Q-network.
    """

    def __init__(self, learning_rate: float):
        super().__init__()

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
        res = np.array([
            (state == self.side).astype(np.float32),
            (state == Board.other_side(self.side)).astype(np.float32),
            (state == EMPTY).astype(np.float32)
        ])
        return torch.tensor(res.reshape(-1), dtype=torch.float32, device=device)

    def __init__(self, name: str, reward_discount: float = 0.95,
                 win_value: float = 1.0, draw_value: float = 0.0,
                 loss_value: float = -1.0, learning_rate: float = 0.01,
                 training: bool = True):
        super().__init__()
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

        self.nn = QNetwork(learning_rate)

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

            # Mask illegal moves in probabilities only
            for i in range(BOARD_SIZE):
                if not board.is_legal(i):
                    probs[i] = -1.0

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
        states = torch.stack(self.state_log).to(device)
        q_pred = torch.stack(self.q_log).to(device)
        targets = q_pred.clone().detach()

        # Apply Q-learning update: batch
        actions = torch.tensor(self.action_log, device=device)
        next_vals = torch.tensor(self.next_value_log, device=device)

        # targets[i, action_i] = gamma * next_value[i]
        targets[torch.arange(len(actions)), actions] = \
            self.reward_discount * next_vals

        self.nn.train_batch(states, targets)
