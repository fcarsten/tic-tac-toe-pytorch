import math
import random
from collections import deque
from typing import Any, List, Optional


class ReplayMemory:
    def __init__(self,
                 buffer_size: int,
                 prioritized: bool = False,
                 alpha: float = 0.6,
                 epsilon: float = 1e-6):
        """
        buffer_size: total memory capacity split evenly across 3 buffers.
        prioritized: if True, use prioritized sampling weights.
        alpha: exponent controlling prioritization degree.
        epsilon: small value to ensure nonzero priority.
        """

        self.n_buffers = 3
        self.buffer_size = buffer_size
        self.prioritized = prioritized
        self.alpha = alpha
        self.epsilon = epsilon

        per_buffer = buffer_size // self.n_buffers
        self.memory = [deque(maxlen=per_buffer) for _ in range(self.n_buffers)]
        self.priorities = [deque(maxlen=per_buffer) for _ in range(self.n_buffers)]


    def __len__(self):
        """Total number of stored transitions."""
        return sum(len(d) for d in self.memory)


    def push(self, transition: Any, buffer_idx: int, priority: Optional[float] = None):
        """
        Add a transition to a specific buffer.

        transition: any object (tuple, dict, custom class)
        buffer_idx: 0, 1, or 2 for which deque to insert into
        priority: optional value; if missing, use max priority of that deque or 1.0
        """

        assert 0 <= buffer_idx < self.n_buffers, "buffer_idx must be 0, 1, or 2"

        if priority is None:
            if len(self.priorities[buffer_idx]) > 0:
                priority = max(self.priorities[buffer_idx])
            else:
                priority = 1.0  # fallback

        self.memory[buffer_idx].append(transition)
        self.priorities[buffer_idx].append(priority + self.epsilon)


    def _balanced_allocation(self, batch_size: int) -> List[int]:
        """Compute how many samples to take from each buffer, evenly and fairly."""
        sizes = [len(d) for d in self.memory]
        n = self.n_buffers

        # If we do not have enough total samples, reduce batch_size
        total_available = sum(sizes)
        batch_size = min(batch_size, total_available)
        if batch_size == 0:
            return [0] * n

        base = batch_size // n
        remainder = batch_size % n
        allocation = [base + (1 if i < remainder else 0) for i in range(n)]

        # Compute deficit and clamp to available sizes
        deficit = 0
        for i in range(n):
            if allocation[i] > sizes[i]:
                deficit += allocation[i] - sizes[i]
                allocation[i] = sizes[i]

        # Redistribute deficit fairly across buffers with surplus
        while deficit > 0:
            candidates = [i for i in range(n) if allocation[i] < sizes[i]]
            if not candidates:
                break

            take_each = math.ceil(deficit / len(candidates))
            for i in candidates:
                if deficit == 0:
                    break
                possible = sizes[i] - allocation[i]
                take = min(take_each, possible, deficit)
                allocation[i] += take
                deficit -= take

        return allocation


    def _sample_from_buffer(self, buffer_idx: int, k: int) -> List[Any]:
        """Sample k elements from one buffer using weighted or uniform sampling."""
        if k == 0:
            return []

        if not self.prioritized:
            return random.sample(self.memory[buffer_idx], k)

        # Prioritized sampling
        priorities = self.priorities[buffer_idx]
        weights = [p ** self.alpha for p in priorities]
        total_weight = sum(weights)
        # Weighted sampling without replacement
        sampled = random.choices(self.memory[buffer_idx],
                                weights=weights,
                                k=k)
        # random.choices samples with replacement, so ensure uniqueness if possible
        if len(set(sampled)) < k <= len(self.memory[buffer_idx]):
            sampled = random.sample(self.memory[buffer_idx], k)

        return sampled


    def sample(self, batch_size: int) -> List[Any]:
        """
        Perform balanced sampling from the 3 buffers.
        Automatically scales if not enough total samples are available.
        Supports uniform and prioritized sampling.
        """
        allocation = self._balanced_allocation(batch_size)

        result = []
        for idx, k in enumerate(allocation):
            result.extend(self._sample_from_buffer(idx, k))

        return result
