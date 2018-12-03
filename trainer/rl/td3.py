"""A TD3 Agent.

Implementation of the TD3 agent from the following paper.
https://arxiv.org/pdf/1802.09477.pdf

"""
from agent import Agent
import numpy as np


class TD3Agent(Agent):
    """Implementation of the TD3 agent from the following paper.

    The implementation is based on https://arxiv.org/pdf/1802.09477.pdf.
    The agent is an actor-critic model, and has 2 critics and 1 actor.

    """
    def __init__(self,
                 critic1,
                 critic2,
                 actor,
                 replay_buffer,
                 rand_process,
                 warmup_size=10000,
                 batch_size=64,
                 gamma=0.99,
                 sigma=0.01,
                 smoothing_regularization_cap=0.05,
                 update_interval=2):
        """Initialization. """
        self.critic1 = critic1
        self.critic2 = critic2
        self.actor = actor
        self.replay_buffer = replay_buffer
        self.rand_process = rand_process
        self.warmup_size = warmup_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.sigma = sigma
        self.c = smoothing_regularization_cap
        self.d = update_interval

    def action(self, sess, observation):
        """Return an action according to the agent's internal policy."""
        return self.actor.get_action(sess, observation)

    def action_with_noise(self, sess, observation):
        """Return a noisy action."""
        action = self.action(sess, observation)
        noise = self.rand_process.sample()
        action_with_noise = action + noise
        return (np.clip(action_with_noise,
                        self.actor.a_low,
                        self.actor.a_high),
                action, noise)

    def store_experience(self, s0, a, r, t, s1):
        """Store experience in the replay buffer."""
        self.replay_buffer.add(s0, a, r, t, s1)

    def train(self, sess, global_step):
        """Improve the agent's policy once."""
        if self.replay_buffer.size > self.warmup_size:
            s0, a, r, t, s1 = self.replay_buffer.sample_batch(self.batch_size)
            epsilon = np.clip(
                np.random.randn(self.batch_size, self.actor.a_dim),
                -self.c, self.c)
            target_actions = self.actor.get_target_action(sess, s1) + epsilon
            target_actions = np.clip(target_actions,
                                     self.actor.a_low,
                                     self.actor.a_high)
            target_qval = self.get_target_qval(sess, s1, target_actions)
            t = t.astype(dtype=int)
            y = r + self.gamma * target_qval * (1 - t)
            self.critic1.train(sess, s0, a, y)
            self.critic2.train(sess, s0, a, y)
            if global_step % self.d == 0:
                actions = self.actor.get_action(sess, s0)
                grads = self.critic1.get_action_gradients(sess, s0, actions)
                self.actor.train(sess, s0, grads[0])
                self.update_targets(sess)
            return s0[0]

    def update_targets(self, sess):
        """Update all target networks."""
        self.critic1.update_target(sess)
        self.critic2.update_target(sess)
        self.actor.update_target(sess)

    def get_target_qval(self, sess, observation, action):
        """Get target Q-val."""
        target_qval1 = self.critic1.get_target_qval(sess, observation, action)
        target_qval2 = self.critic2.get_target_qval(sess, observation, action)
        return np.minimum(target_qval1, target_qval2)

    def get_qval(self, sess, observation, action):
        """Get target Q-val."""
        qval1 = self.critic1.get_qval(sess, observation, action)
        qval2 = self.critic2.get_qval(sess, observation, action)
        return np.minimum(qval1, qval2)

    def initialize(self, sess):
        """Sync network weights."""
        self.update_targets(sess)
