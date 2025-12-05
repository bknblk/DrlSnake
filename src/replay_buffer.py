from collections import deque
import random
import numpy as np
from dataclasses import dataclass

@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity = 10000):
        self.buffer = deque(maxlen = capacity)

    def push(self, state, action, reward, next_state, done):
        experience = Experience(
                state = state,
                action = action,
                reward = reward,
                next_state = next_state,
                done = done
        )
        self.buffer.append(experience)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            raise ValueError(f"not enough samples")

        experiences = random.sample(self.buffer, batch_size)

        states = np.array([exp.state for exp in experiences])
        actions = np.array([exp.action for exp in experiences])
        rewards = np.array([exp.reward for exp in experiences])
        next_states = np.array([exp.next_state for exp in experiences])
        dones = np.array([exp.done for exp in experiences])

        states = states[..., np.newaxis]
        next_states = next_states[...,np.newaxis]

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

