"""Implementation of a DDPG agent.

Implementation of DDPG - Deep Deterministic Policy Gradient
Algorithm and hyperparameter details can be found here:
    http://arxiv.org/pdf/1509.02971v2.pdf

This implementation is based on https://github.com/pemami4911/deep-rl/tree/master/ddpg

"""
import numpy as np
import tensorflow as tf
import tflearn


def create_conv_part(net_inputs):
    """Creates an input stream from depth image."""
    net = tflearn.conv_2d(incoming=net_inputs,
                          nb_filter=32,
                          filter_size=5,
                          strides=5,
                          padding='valid',
                          activation='relu')
    net = tflearn.max_pool_2d(incoming=net, kernel_size=2, strides=2)
    net = tflearn.conv_2d(incoming=net,
                          nb_filter=32,
                          filter_size=2,
                          strides=2,
                          padding='valid',
                          activation='relu')
    net = tflearn.max_pool_2d(incoming=net, kernel_size=2, strides=2)
    net = tflearn.conv_2d(incoming=net,
                          nb_filter=64,
                          filter_size=2,
                          strides=2,
                          padding='valid',
                          activation='relu')
    net = tflearn.max_pool_2d(incoming=net, kernel_size=2, strides=2)
    net = tflearn.flatten(incoming=net)
    return net


def create_fc_part(net_inputs):
    """Construct the FC part of the net."""
    net = tflearn.fully_connected(incoming=net_inputs,
                                  n_units=512,
                                  activation='relu')
    net = tflearn.fully_connected(incoming=net,
                                  n_units=256,
                                  activation='relu')
    return net


def _print_var_info(var_list, title):
    print(title)
    for v in var_list:
        print(v)
    print('-' * 50)


class ImageEncoder(object):
    """This class learns an embedding of images."""

    def __init__(self, img_h, img_w):
        """Initialization."""
        self.img_h = img_h
        self.img_w = img_w
        with tf.variable_scope('image_encoder'):
            self.inputs, self.embedding = self.create_embedding_network()
        self.network_params = tf.trainable_variables(scope='image_encoder')

    def create_embedding_network(self):
        """Create the embedding learning part."""
        i_inputs = tflearn.input_data(shape=[None, self.img_h, self.img_w, 1])
        embedding = create_conv_part(i_inputs)
        return i_inputs, embedding


class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.
    """

    def __init__(self,
                 sess,
                 image_encoder,
                 state_dim,
                 action_dim,
                 learning_rate,
                 tau,
                 batch_size):
        self.sess = sess
        self._image_encoder = image_encoder
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        # Actor Network
        self.actor_i = self._image_encoder.inputs
        with tf.variable_scope('actor'):
            self.actor_s, self.actor_out = self.create_actor_network()
        self.network_params = tf.trainable_variables(scope='actor')
        _print_var_info(self.network_params, 'actor')

        # Target Network
        self.target_i = self._image_encoder.inputs
        with tf.variable_scope('actor_target'):
            self.target_s, self.target_out = self.create_actor_network()
        self.target_network_params = tf.trainable_variables(scope='actor_target')
        _print_var_info(self.target_network_params, 'actor_target')

        # Op for periodically updating target network with online network weights
        self.update_target_network_params = [
            self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                 tf.multiply(self.target_network_params[i], 1. - self.tau))
            for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        actor_gradients = tf.gradients(
            self.actor_out, self.network_params, -self.action_gradient)
        self.actor_gradients = map(
            lambda x: tf.clip_by_norm(tf.div(x, self.batch_size), 5.0),
            actor_gradients)

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(
            zip(self.actor_gradients, self.network_params))

    def create_actor_network(self):
        """The actor has 2 flows, depth image and vector of state."""
        s_inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.merge([self._image_encoder.embedding, s_inputs],
                            mode='concat',
                            axis=1)
        net = create_fc_part(net)
        out = tflearn.fully_connected(incoming=net,
                                      n_units=self.a_dim,
                                      activation='tanh')
        return s_inputs, out

    def train(self, inputs, a_gradient):
        i_inputs, s_inputs = inputs
        i_inputs = np.expand_dims(i_inputs, -1)
        self.sess.run(self.optimize, feed_dict={
            self.actor_i: i_inputs,
            self.actor_s: s_inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        i_inputs, s_inputs = inputs
        i_inputs = np.expand_dims(i_inputs, -1)
        if np.ndim(i_inputs) < 4:
            i_inputs = np.expand_dims(i_inputs, 0)
        if np.ndim(s_inputs) < 2:
            s_inputs = np.expand_dims(s_inputs, 0)
        return self.sess.run(self.actor_out, feed_dict={
            self.actor_i: i_inputs,
            self.actor_s: s_inputs
        })

    def predict_target(self, inputs):
        i_inputs, s_inputs = inputs
        i_inputs = np.expand_dims(i_inputs, -1)
        return self.sess.run(self.target_out, feed_dict={
            self.target_i: i_inputs,
            self.target_s: s_inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """

    def __init__(self,
                 sess,
                 image_encoder,
                 state_dim,
                 action_dim,
                 learning_rate,
                 tau,
                 gamma):
        self.sess = sess
        self._image_encoder = image_encoder
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        # Create the critic network
        self.critic_i = self._image_encoder.inputs
        with tf.variable_scope('critic'):
            self.critic_s, self.critic_action, self.critic_out = (
                self.create_critic_network())
        self.network_params = tf.trainable_variables(scope='critic')
        _print_var_info(self.network_params, 'critic')

        # Target Network
        self.target_i = self._image_encoder.inputs
        with tf.variable_scope('critic_target'):
            self.target_s, self.target_action, self.target_out = (
                self.create_critic_network())
        self.target_network_params = tf.trainable_variables(scope='critic_target')
        _print_var_info(self.target_network_params, 'critic_target')

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
                                                  + tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        var_list = self._image_encoder.network_params + self.network_params
        self.loss = tflearn.mean_square(self.predicted_q_value, self.critic_out)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss, var_list=var_list)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.critic_out, self.critic_action)

    def create_critic_network(self):
        """The critic has 2 flows, depth image and vector of state."""
        s_inputs = tflearn.input_data(shape=[None, self.s_dim])
        observation = tflearn.merge([self._image_encoder.embedding, s_inputs],
                                    mode='concat',
                                    axis=1)

        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.merge([observation, action], mode='concat', axis=1)

        net = create_fc_part(net)
        out = tflearn.fully_connected(incoming=net, n_units=1)

        return s_inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        i_input, s_input = inputs
        i_input = np.expand_dims(i_input, -1)
        return self.sess.run([self.critic_out, self.optimize], feed_dict={
            self.critic_i: i_input,
            self.critic_s: s_input,
            self.critic_action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        i_inputs, s_inputs = inputs
        i_inputs = np.expand_dims(i_inputs, -1)
        if np.ndim(i_inputs) < 4:
            i_inputs = np.expand_dims(i_inputs, 0)
        if np.ndim(s_inputs) < 2:
            s_inputs = np.expand_dims(s_inputs, 0)
        return self.sess.run(self.critic_out, feed_dict={
            self.critic_i: i_inputs,
            self.critic_s: s_inputs,
            self.critic_action: action
        })

    def predict_target(self, inputs, action):
        i_inputs, s_inputs = inputs
        i_inputs = np.expand_dims(i_inputs, -1)
        return self.sess.run(self.target_out, feed_dict={
            self.target_i: i_inputs,
            self.target_s: s_inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        i_input, s_input = inputs
        i_input = np.expand_dims(i_input, -1)
        return self.sess.run(self.action_grads, feed_dict={
            self.critic_i: i_input,
            self.critic_s: s_input,
            self.critic_action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)


class OUPfromWiki:
    """
    Ornstein Uhlenbeck Process whose implementation follows wikipedia
    """

    def __init__(self, action_dim, theta, sigma, init_val=0.0, scale_min=0, annealing_steps=0):
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
        delta_xt = self.theta * (-1.0 * self.xt) + self.sigma * np.random.randn(self.action_dim)
        self.xt += delta_xt
        noise = self.scale * self.xt
        self.scale = max(self.scale - self.scale_delta, self.scale_min)
        return noise
