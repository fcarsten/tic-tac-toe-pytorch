import random
import torch
import numpy as np
from collections import deque
from tic_tac_toe.Board import Board
from tic_tac_toe.Player import GameResult
from tic_tac_toe.EGreedyNNQPlayer import EGreedyNNQPlayer


class ReplayNNQPlayer(EGreedyNNQPlayer):
    """
    Extends EGreedyNNQPlayer with an Experience Replay Buffer.
    Instead of training on a single game at a time, it samples random
    batches of moves from past games to stabilize learning.
    """

    def __init__(self, name: str = "ReplayNNQPlayer", reward_discount: float = 0.95,
                 win_value: float = 1.0, draw_value: float = 0.0,
                 loss_value: float = -1.0, learning_rate: float = 0.0005,
                 random_move_prob: float = 0.95, random_move_decrease: float = 0.995,
                 batch_size: int = 64, buffer_size: int = 10000,
                 training: bool = True, device: torch.device = torch.device("cpu")):

        super().__init__(name, reward_discount, win_value, draw_value,
                         loss_value, learning_rate, random_move_prob,
                         random_move_decrease, training, device)

        # Experience Replay Buffer: Stores (state, action, next_max_q, reward, is_terminal)
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def final_result(self, result: GameResult):
        if not self.training:
            return

        # 1. Determine the final reward for the game
        if (result == GameResult.NAUGHT_WIN and self.side == 1) or \
                (result == GameResult.CROSS_WIN and self.side == 2):
            reward = self.win_value
        elif result == GameResult.DRAW:
            reward = self.draw_value
        else:
            reward = self.loss_value

        # 2. Store this game's transitions into the replay buffer
        # We zip logs to create (S, A, S_next_value) tuples
        for i in range(len(self.action_log)):
            state = self.state_log[i]
            action = self.action_log[i]

            # The 'next_value' in our existing logic is already the max_q of the next state
            # recorded during the move() function.
            next_val = self.next_value_log[i] if i < len(self.next_value_log) else reward
            is_terminal = (i == len(self.action_log) - 1)

            self.memory.append((state, action, next_val, is_terminal))

        # 3. Perform a Training Step using a Random Batch from Memory
        if len(self.memory) > self.batch_size:
            self._train_from_replay()

        # 4. Decay exploration probability
        self.random_move_prob *= self.random_move_decrease
        if self.writer:
            self.writer.add_scalar(f'{self.name}/Random_Move_Prob', self.random_move_prob, self.game_number)

    def _train_from_replay(self):
        """Samples a random batch from memory and updates the neural network."""
        batch = random.sample(self.memory, self.batch_size)

        states, actions, next_vals, terminals = zip(*batch)

        states_tensor = torch.stack(states).to(self.device)
        actions_tensor = torch.tensor(actions, device=self.device)
        next_vals_tensor = torch.tensor(next_vals, device=self.device)

        # Get current Q-values for the batch
        with torch.no_grad():
            targets = self.nn(states_tensor)

            # Bellman Equation: R + gamma * maxQ(S')
        # Note: If terminal, next_val is just the reward (no discount)
        updated_q_values = next_vals_tensor.clone()

        # Apply discount to non-terminal transitions
        # In our logic, next_vals already contains the max_q of the next state for non-terminals
        # and the raw reward for terminals.
        mask = torch.tensor([not t for t in terminals], device=self.device)
        updated_q_values[mask] = self.reward_discount * next_vals_tensor[mask]

        # Update specific actions in the target tensor
        row_indices = torch.arange(self.batch_size, device=self.device)
        targets[row_indices, actions_tensor] = updated_q_values

        # Backpropagate
        loss = self.nn.train_batch(states_tensor, targets, writer=self.writer,
                                   name=self.name, game_number=self.game_number)

        if self.writer:
            self.writer.add_scalar(f'{self.name}/Training_Loss', loss, self.game_number)