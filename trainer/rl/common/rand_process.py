"""Random processes for action exploration."""

import numpy as np


class OrnsteinUhlenbeck(object):
    """Ornstein Uhlenbeck Process, implementation follows wikipedia."""

    def __init__(self,
                 action_dim,
                 theta,
                 sigma,
                 init_val=0.0,
                 scale_min=0.0,
                 annealing_steps=0):
        self.action_dim = action_dim
        self.theta = theta
        self.sigma = sigma

        self.scale = 1.0
        self.scale_min = scale_min
        if annealing_steps > 0:
            self.scale_delta = (self.scale - scale_min) / annealing_steps
        else:
            self.scale_delta = 0.0

        # initialize x0
        self.xt = np.ones(action_dim) * init_val

    def sample(self):
        """Sample noise."""
        delta_xt = (self.theta * (-1.0 * self.xt) +
                    self.sigma * np.random.randn(self.action_dim))
        self.xt += delta_xt
        noise = self.scale * self.xt
        self.scale = max(self.scale - self.scale_delta, self.scale_min)
        return noise


class GaussianNoise(object):
    """Gaussian noise."""

    def __init__(self,
                 action_dim,
                 mu,
                 sigma,
                 sigma_min=0.0,
                 annealing_steps=0):
        self.action_dim = action_dim
        self.mu = mu
        self.sigma = sigma
        self.sigma_min = sigma_min
        if annealing_steps > 0:
            self.sigma_delta = (self.sigma - sigma_min) / annealing_steps
        else:
            self.sigma_delta = 0.0

    def sample(self):
        """Sample from normal distribution."""
        noise = self.mu + np.random.randn(self.action_dim) * self.sigma
        self.sigma = max(self.sigma - self.sigma_delta, self.sigma_min)
        return noise
