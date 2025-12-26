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

    def get_probs(self, nn_input: torch.Tensor):
        with torch.no_grad():
            qvalues = self.nn(nn_input.unsqueeze(0))[0]
            probs = torch.softmax(qvalues, dim=0)
        return probs, qvalues

    def move(self, board: Board):
        state_tensor = self.board_state_to_nn_input(board.state)
        self.state_log.append(state_tensor)

        probs, qvalues = self.get_probs(state_tensor)
        q_copy = qvalues.clone()

        for index in range(len(qvalues)):
            if not board.is_legal(index):
                probs[index] = -1.0

        move = int(torch.argmax(probs).item())

        if self.action_log:
            self.next_value_log.append(q_copy[move].item())

        self.action_log.append(move)
        self.q_log.append(q_copy)

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

        targets = []
        for i, qvalues in enumerate(self.q_log):
            target = qvalues.clone().detach()
            target[self.action_log[i]] = self.reward_discount * self.next_value_log[i]
            targets.append(target)

        inputs = torch.stack(self.state_log).to(device)
        targets = torch.stack(targets).to(device)

        self.nn.train_batch(inputs, targets)
