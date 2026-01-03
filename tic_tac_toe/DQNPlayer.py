import copy
import torch

from tic_tac_toe.ReplayNNQPlayer import ReplayNNQPlayer


class DQNPlayer(ReplayNNQPlayer):
    def __init__(self, name: str = "DQNPlayer", target_update_freq: int = 500, **kwargs):
        super().__init__(name, **kwargs)
        self._create_target_network()
        self.target_update_freq = target_update_freq

    def _create_target_network(self):
        self.target_nn = copy.deepcopy(self.nn)
        self.target_nn.to(self.device)
        self.target_nn.eval()

    def _train_from_replay(self):
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_v = torch.stack(states).to(self.device)
        actions_v = torch.tensor(actions, device=self.device)
        rewards_v = torch.tensor(rewards, device=self.device, dtype=torch.float)

        # 1. Get current Q values from the Policy Network
        current_qs = self.nn(states_v)

        # 2. Get Next-State values from the TARGET Network
        with torch.no_grad():
            next_q_max = torch.zeros(self.batch_size, device=self.device)

            # Vectorized evaluation of non-terminal next states
            non_final_mask = torch.tensor([ns is not None for ns in next_states], device=self.device)

            if non_final_mask.any():
                non_final_next_states = torch.stack([ns for ns in next_states if ns is not None]).to(self.device)

                target_q_estimates = self.target_nn(non_final_next_states)

                # Use target_nn for the 'max Q' part of the Bellman equation

                next_q_max[non_final_mask] = torch.max(target_q_estimates, dim=1)[0]

        # 3. Bellman Equation: r + gamma * max_Q_target(s')
        targets = current_qs.clone()
        expected_q = rewards_v + (self.reward_discount * next_q_max)
        targets[torch.arange(self.batch_size), actions_v] = expected_q

        # 4. Perform optimization
        self.nn.train_batch(states_v, targets, writer=self.writer,
                               name=self.name, game_number=self.game_number)

        if self.writer:
            td_errors = torch.abs(expected_q - current_qs[torch.arange(self.batch_size), actions_v])
            self.writer.add_scalar(f'{self.name}/Mean_TD_Error', td_errors.mean(), self.move_step)

        # 5. Corrected Synchronization: Use inherited self.move_step
        if self.move_step % self.target_update_freq == 0:
            self.target_nn.load_state_dict(self.nn.state_dict())