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

    def __init__(self, learning_rate: float):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(BOARD_SIZE * 3, BOARD_SIZE * 3 * 9),
            nn.ReLU(),
            nn.Linear(BOARD_SIZE * 3 * 9, BOARD_SIZE)
        )

        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def train_step(self, inputs, targets):
        self.optimizer.zero_grad()
        q_pred = self.forward(inputs)
        loss = self.loss_fn(q_pred, targets)
        loss.backward()
        self.optimizer.step()


class NNQPlayer(Player):
    """
    Neural network Q-learning Tic Tac Toe player rewritten for PyTorch.
    """

    def board_state_to_nn_input(self, state: np.ndarray) -> np.ndarray:
        res = np.array([
            (state == self.side).astype(int),
            (state == Board.other_side(self.side)).astype(int),
            (state == EMPTY).astype(int)
        ])
        return res.reshape(-1)

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
        self.next_max_log = []
        self.q_log = []

        self.nn = QNetwork(learning_rate)

    def new_game(self, side: int):
        self.side = side
        self.state_log.clear()
        self.action_log.clear()
        self.next_max_log.clear()
        self.q_log.clear()

    def get_probs(self, input_pos: np.ndarray):
        with torch.no_grad():
            x = torch.tensor(input_pos, dtype=torch.float32).unsqueeze(0)
            qvals = self.nn(x)[0].cpu().numpy()

        probs = torch.softmax(torch.tensor(qvals), dim=0).numpy()
        return probs, qvals

    def move(self, board: Board):
        self.state_log.append(board.state.copy())

        nn_input = self.board_state_to_nn_input(board.state)
        probs, qvalues = self.get_probs(nn_input)
        qvalues = np.copy(qvalues)

        for index in range(len(qvalues)):
            if not board.is_legal(index):
                probs[index] = -1  # remove illegal moves from selection

        move = int(np.argmax(probs))

        if len(self.action_log) > 0:
            self.next_max_log.append(qvalues[move])

        self.action_log.append(move)
        self.q_log.append(qvalues)

        _, res, finished = board.move(move, self.side)
        return res, finished

    def final_result(self, result: GameResult):
        if (result == GameResult.NAUGHT_WIN and self.side == NAUGHT) or \
           (result == GameResult.CROSS_WIN and self.side == CROSS):
            reward = self.win_value
        elif (result == GameResult.NAUGHT_WIN and self.side == CROSS) or \
             (result == GameResult.CROSS_WIN and self.side == NAUGHT):
            reward = self.loss_value
        elif result == GameResult.DRAW:
            reward = self.draw_value
        else:
            raise ValueError("Unexpected game result: {}".format(result))

        self.next_max_log.append(reward)

        if not self.training:
            return

        inputs = np.array([self.board_state_to_nn_input(x) for x in self.state_log],
                          dtype=np.float32)
        targets = self.calculate_targets()

        inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
        targets_tensor = torch.tensor(targets, dtype=torch.float32)

        self.nn.train_step(inputs_tensor, targets_tensor)

    def calculate_targets(self):
        game_length = len(self.action_log)
        targets = []

        for i in range(game_length):
            target = np.copy(self.q_log[i])
            target[self.action_log[i]] = self.reward_discount * self.next_max_log[i]
            targets.append(target)

        return np.array(targets, dtype=np.float32)

