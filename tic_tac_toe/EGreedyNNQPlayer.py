from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter  # Added

from tic_tac_toe.Board import Board, BOARD_SIZE, EMPTY, CROSS, NAUGHT
from tic_tac_toe.Player import Player, GameResult
from tic_tac_toe.SimpleNNQPlayer import NNQPlayer

class EGreedyNNQPlayer(NNQPlayer):
    """
    Neural network Q-learning Tic-tac-toe player rewritten for PyTorch.
    """
    def __init__(self, name: str = "EGreedyNNQPlayer",
                 random_move_prob: float = 0.95, random_move_decrease: float = 0.995, random_min_prob: float = 0.0, **kwargs):  # Added writer
        super().__init__(name, **kwargs)

        self.random_move_prob = random_move_prob
        self.random_min_prob = random_min_prob
        self.random_move_decrease =random_move_decrease

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
                if self.writer and self.training and self.move_step % 500 == 0:
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

            if (self.training is True) and (np.random.rand(1) < self.random_move_prob):
                move = board.random_empty_spot()
            else:
                move = int(torch.argmax(logits).item())

            if self.training:
                if self.action_log: # Skip on first move, ie when action_log is empty
                    self.next_value_log.append(torch.max(q_values).item())
                self.action_log.append(move)

            _, res, finished = board.move(move, self.side)
            return res, finished

    def final_result(self, result: GameResult):
        if not self.training:
            return
        super().final_result(result)
        # Decrease random move probability
        self.random_move_prob *= self.random_move_decrease
        if self.random_move_prob < self.random_min_prob:
            self.random_move_prob = self.random_min_prob

        if self.writer:
            self.writer.add_scalar(f'{self.name}/Random_Move_Prob', self.random_move_prob, self.game_number)