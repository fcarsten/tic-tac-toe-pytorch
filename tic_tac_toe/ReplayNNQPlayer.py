import random
import torch
import numpy as np
from collections import deque
from tic_tac_toe.Board import Board, EMPTY
from tic_tac_toe.Player import GameResult
from tic_tac_toe.EGreedyNNQPlayer import EGreedyNNQPlayer


class ReplayNNQPlayer(EGreedyNNQPlayer):
    def __init__(self, name: str = "ReplayNNQPlayer", batch_size: int = 64,
                 buffer_size: int = 10000, **kwargs):
        super().__init__(name, **kwargs)
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def move(self, board: Board):
        self.move_step += 1
        # 1. Get current Q-values
        nn_input = self.board_state_to_nn_input(board.state)
        with torch.no_grad():
            q_values = self.nn(nn_input)

        # 2. Your Alignment Logic: Skip first move to get S_{t+1}
        if self.training and self.action_log:
            # The current state is the "Next State" for the previous move
            self.next_state_log.append(nn_input)
            # Move-level logging (use q_values for logging)
            if self.writer and self.training and self.move_step % 500 == 0:
                # self.log_q_heatmap(q_values, self.move_step)
                self.writer.add_histogram(f'{self.name}/Action_Q_Distribution', q_values, self.move_step)
                max_q = float(torch.max(q_values).item())
                avg_q = float(torch.mean(q_values).item())
                self.writer.add_scalar(f'{self.name}/Max_Q_Value', max_q, self.move_step)
                self.writer.add_scalar(f'{self.name}/Average_Q_In_Game', avg_q, self.move_step)
                self.writer.add_scalar(f'{self.name}/Move_Confidence', max_q - avg_q, self.move_step)

            # 3. E-Greedy Selection (Same as your EGreedyNNQPlayer)
        occupied_mask = torch.as_tensor(board.state != EMPTY, device=self.device, dtype=torch.bool)
        logits = q_values.clone()
        logits[occupied_mask] = -float('inf')

        if (self.training) and (np.random.rand(1) < self.random_move_prob):
            move = board.random_empty_spot()
        else:
            move = int(torch.argmax(logits).item())

        if self.training:
            self.state_log.append(nn_input)
            self.action_log.append(move)

        _, res, finished = board.move(move, self.side)
        return res, finished

    def new_game(self, side: int):
        super().new_game(side)
        self.next_state_log = []  # Track the actual tensors

    def final_result(self, result: GameResult):
        if not self.training: return

        reward = self.get_reward_value(result)  # Logic to get 1.0, 0.0, -1.0

        # Store transitions in memory
        for i in range(len(self.action_log)):
            s = self.state_log[i]
            a = self.action_log[i]

            if i < len(self.next_state_log):
                s_next = self.next_state_log[i]
                done = False
            else:
                s_next = None  # Terminal
                done = True

            self.memory.append((s, a, reward, s_next, done))

        # Sample and Train
        if len(self.memory) >= self.batch_size:
            self._train_from_replay()

        self.random_move_prob *= self.random_move_decrease
        self.writer.add_scalar(f'{self.name}/Random_Move_Prob', self.random_move_prob, self.game_number)

    def _train_from_replay(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_v = torch.stack(states).to(self.device)
        actions_v = torch.tensor(actions, device=self.device)
        rewards_v = torch.tensor(rewards, device=self.device)

        # Calculate Targets
        with torch.no_grad():
            # Get current brain's opinion on ALL states in batch
            current_qs = self.nn(states_v)

            # For next states, find the Max Q
            next_q_max = torch.zeros(self.batch_size, device=self.device)
            for j, ns in enumerate(next_states):
                if ns is not None:
                    next_q_max[j] = torch.max(self.nn(ns.to(self.device)))

        # Bellman Equation: r + gamma * maxQ(s')
        targets = current_qs.clone()
        for j in range(self.batch_size):
            if dones[j]:
                targets[j, actions_v[j]] = rewards_v[j]
            else:
                targets[j, actions_v[j]] = rewards_v[j] + self.reward_discount * next_q_max[j]

        self.nn.train_batch(states_v, targets, writer=self.writer,
                            name=self.name, game_number=self.game_number)

