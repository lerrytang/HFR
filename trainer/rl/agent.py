"""General interface of an RL agent.

The classes implements this class need to support the following interfaces:
1. random_action(observation), given an observation return a random action.
2. action(observation), given an observation return an action from the agent's policy.
3. train(global_step), improves the agents internal policy once.

"""


class Agent(object):
    """General interface of an RL agent."""

    def initialize(self, sess):
        """All initialization work goes here.

        This function provides the agent a chance to do all pre-learning tasks.
        E.g. sync weights of networks for a DDPG agent.

        Args:
            sess: tf.Session object.
        """
        pass

    def action(self, sess, observation):
        """Return an action according to the agent's internal policy.

        Given an observation return an action according to the agent's
        internal policy. Specifications of the action space should be
        given/initialized when the agent is initialized.

        Args:
            sess: tf.Session object.
            observation: object, observations from the env.
        Returns:
            numpy.array, represent an action.
        """
        raise NotImplementedError('Not implemented')

    def action_with_noise(self, sess, observation):
        """Return a noisy action.

        Given an observation return a noisy action according to the agent's
        internal policy and exploration noise process.
        Specifications of the action space should be given/initialized
        when the agent is initialized.

        Args:
            sess: tf.Session object.
            observation: object, observations from the env.
        Returns:
            numpy.array, represent an action.
        """
        raise NotImplementedError('Not implemented')

    def train(self, sess, global_step):
        """Improve the agent's policy once.

        Train the agent and improve its internal policy once.

        Args:
            sess: tf.Session object.
            global_step: int, global step count.
        Returns:
            object, represent training metrics.
        """
        raise NotImplementedError('Not implemented')
