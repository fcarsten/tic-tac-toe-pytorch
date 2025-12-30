import torch
import random

from tic_tac_toe.DQNPlayer import DQNPlayer


class DoubleDQNPlayer(DQNPlayer):
    def __init__(self, name: str = "DoubleDQNPlayer", **kwargs):
        super().__init__(name, **kwargs)

    def _train_from_replay(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_v = torch.stack(states).to(self.device)
        actions_v = torch.tensor(actions, device=self.device)
        rewards_v = torch.tensor(rewards, device=self.device, dtype=torch.float)

        # 1. Get current Q values from the Policy Network
        current_qs = self.nn(states_v)

        # 2. Double DQN Logic for Next-State Values
        with torch.no_grad():
            next_q_values = torch.zeros(self.batch_size, device=self.device)

            # Mask for non-terminal states
            non_final_mask = torch.tensor([ns is not None for ns in next_states], device=self.device)

            if non_final_mask.any():
                non_final_next_states = torch.stack([ns for ns in next_states if ns is not None]).to(self.device)

                target_q_estimates = self.target_nn(non_final_next_states)

                # --- DOUBLE DQN STEP ---
                # A. Select the best action using the ONLINE Policy Network (self.nn)
                # This prevents picking the "luckiest" value from the target network.
                best_actions = self.nn(non_final_next_states).argmax(dim=1, keepdim=True)

                # B. Evaluate that specific action using the TARGET Network (self.target_nn)
                # Extract the Q-values for the 'best_actions' found by the online network
                next_q_values[non_final_mask] = target_q_estimates.gather(1, best_actions).squeeze(1)

        # 3. Bellman Equation: r + gamma * Q_target(s', argmax Q_policy(s', a))
        targets = current_qs.clone()
        expected_q = rewards_v + (self.reward_discount * next_q_values)
        targets[torch.arange(self.batch_size), actions_v] = expected_q

        # 4. Perform optimization
        self.nn.train_batch(states_v, targets, writer=self.writer,
                            name=self.name, game_number=self.game_number)

        if self.writer:
            td_errors = torch.abs(expected_q - current_qs[torch.arange(self.batch_size), actions_v])
            self.writer.add_scalar(f'{self.name}/Mean_TD_Error', td_errors.mean(), self.move_step)

        # 5. Corrected Synchronization: Use inherited self.move_step
        if self.move_step % self.target_update_freq == 0:
            self._update_target_network()
            if self.writer:
                # Log that a sync happened
                self.writer.add_scalar(f'{self.name}/Target_Sync_Event', 1, self.move_step)