""" Data structure for implementing experience replay.

This is based on https://github.com/pemami4911/deep-rl/tree/master/ddpg.

"""
from collections import deque
import numpy as np
import random


class ReplayBuffer(object):
    """Replay buffer for DDPG."""

    def __init__(self, buffer_size, random_seed=123):
        """The right side of the deque contains the most recent experiences."""
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        """Add experience to the buffer."""
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    @property
    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        """Sample a batch of experience."""
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        i_batch = np.array([_[0][0] for _ in batch])
        s_batch = np.array([_[0][1] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        i2_batch = np.array([_[4][0] for _ in batch])
        s2_batch = np.array([_[4][1] for _ in batch])

        return (i_batch, s_batch), a_batch, r_batch, t_batch, (i2_batch, s2_batch)

    def clear(self):
        """Clear buffer."""
        self.buffer.clear()
        self.count = 0