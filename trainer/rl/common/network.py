import numpy as np
import tensorflow as tf


def create_conv_net(inputs):
    """Create an input stream from depth image."""
    x = tf.layers.conv2d(inputs=inputs,
                         kernel_size=5,
                         filters=32,
                         strides=5,
                         padding='valid',
                         activation=tf.nn.relu)
    x = tf.layers.max_pooling2d(inputs=x,
                                pool_size=2,
                                strides=2)
    x = tf.layers.conv2d(inputs=x,
                         kernel_size=2,
                         filters=32,
                         strides=2,
                         padding='valid',
                         activation=tf.nn.relu)
    x = tf.layers.max_pooling2d(inputs=x,
                                pool_size=2,
                                strides=2)
    x = tf.layers.conv2d(inputs=x,
                         kernel_size=2,
                         filters=64,
                         strides=2,
                         padding='valid',
                         activation=tf.nn.relu)
    x = tf.layers.max_pooling2d(inputs=x,
                                pool_size=2,
                                strides=2)
    outputs = tf.layers.flatten(inputs=x)
    return outputs


def create_fc_net(inputs):
    """Construct the FC part of the net."""
    x = tf.layers.dense(inputs=inputs,
                        units=512,
                        activation=tf.nn.relu)
    outputs = tf.layers.dense(inputs=x,
                              units=256,
                              activation=tf.nn.relu)
    return outputs


def print_var_info(var_list, title):
    tf.logging.info('-' * 50)
    tf.logging.info(title)
    for v in var_list:
        tf.logging.info(v)
    tf.logging.info('-' * 50)


def preprocess_observation(observation):
    image, state = observation
    image = np.expand_dims(image, -1)
    if np.ndim(image) == 3:
        image = np.expand_dims(image, 0)
    if np.ndim(state) == 1:
        state = np.expand_dims(state, 0)
    return image, state


class ImageEncoder(object):
    """This class learns an embedding of images."""

    def __init__(self, img_h, img_w, name=None):
        """Initialization."""
        self.img_h = img_h
        self.img_w = img_w
        scope_name = 'image_encoder' if name is None else name
        with tf.variable_scope(scope_name):
            self.inputs, self.embedding = self.create_embedding_network()
        self.params = tf.trainable_variables(scope=scope_name)

    def create_embedding_network(self):
        """Create the embedding learning part."""
        inputs = tf.placeholder(dtype=tf.float32,
                                shape=[None, self.img_h, self.img_w, 1])
        embedding = create_conv_net(inputs)
        return inputs, embedding


class Actor(object):
    """Actor network."""

    def __init__(self,
                 state_dim,
                 action_dim,
                 action_high,
                 action_low,
                 learning_rate,
                 tau,
                 batch_size,
                 grad_norm_clip,
                 img_encoder,
                 target_img_encoder,
                 train_img_encoder=True,
                 share_img_encoder=False,
                 name=None):
        """Actor initialization."""

        # member initialization
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.a_high = action_high
        self.a_low = action_low
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size
        self.grad_norm_clip = grad_norm_clip
        self.train_img_encoder = train_img_encoder
        self.share_img_encoder = share_img_encoder
        self.img_encoder = img_encoder
        self.target_img_encoder = target_img_encoder

        # build actor and target networks
        scope_name = 'actor' if name is None else name
        with tf.variable_scope(scope_name):
            self.ob_state, self.action = self._build_network(
                self.img_encoder)
        self.params = tf.trainable_variables(scope=scope_name)
        print_var_info(self.params + self.img_encoder.params, scope_name)
        scope_name += '_target'
        with tf.variable_scope(scope_name):
            self.target_ob_state, self.target_action = self._build_network(
                self.target_img_encoder)
        self.target_params = tf.trainable_variables(scope=scope_name)
        print_var_info(
            self.target_params + self.target_img_encoder.params, scope_name)

        # build operators
        (self.update_target_op,
         self.update_target_img_encoder_op,
         self.action_gradient,
         self.train_op) = self._build_ops()

    def _build_network(self, im_encoder):
        """Build network for the actor part."""
        ob_state = tf.placeholder(dtype=tf.float32, shape=[None, self.s_dim])
        observation = tf.concat(values=[im_encoder.embedding, ob_state], axis=1)
        x = create_fc_net(observation)
        action = tf.layers.dense(inputs=x,
                                 units=self.a_dim,
                                 activation=tf.nn.sigmoid)
        scaled_action = action * (self.a_high - self.a_low) + self.a_low
        return ob_state, scaled_action

    def _build_ops(self):
        """Build tf operators."""
        target_update_op = tf.group([
            x.assign(tf.multiply(y, self.tau) + tf.multiply(x, 1. - self.tau))
            for x, y in zip(self.target_params, self.params)])
        target_im_encoder_update_op = tf.group([
            x.assign(tf.multiply(y, self.tau) + tf.multiply(x, 1. - self.tau))
            for x, y in zip(self.target_img_encoder.params,
                            self.img_encoder.params)])
        grad_ph = tf.placeholder(tf.float32, [None, self.a_dim])
        if self.train_img_encoder:
            params = self.params + self.img_encoder.params
        else:
            params = self.params
        print_var_info(params, 'Actor_to_optimize')
        actor_gradients = tf.gradients(self.action, params, -grad_ph)
        clipped_grad = [
            tf.clip_by_norm(tf.div(x, self.batch_size), self.grad_norm_clip)
            for x in actor_gradients]
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_op = optimizer.apply_gradients(zip(clipped_grad, params))
        return target_update_op, target_im_encoder_update_op, grad_ph, train_op

    def get_action(self, sess, observation):
        """Return action based on its policy."""
        image, state = preprocess_observation(observation)
        return sess.run(self.action,
                        feed_dict={self.img_encoder.inputs: image,
                                   self.ob_state: state})

    def get_target_action(self, sess, observation):
        """Return action based on its target policy."""
        image, state = preprocess_observation(observation)
        return sess.run(self.target_action,
                        feed_dict={self.target_img_encoder.inputs: image,
                                   self.target_ob_state: state})

    def update_target(self, sess):
        """Update the target network's weights."""
        sess.run(self.update_target_op)
        if not self.share_img_encoder:
            sess.run(self.update_target_img_encoder_op)

    def train(self, sess, observation, critic_action_grads):
        """Train the actor once."""
        image, state = preprocess_observation(observation)
        sess.run(self.train_op,
                 feed_dict={self.img_encoder.inputs: image,
                            self.ob_state: state,
                            self.action_gradient: critic_action_grads})


class Critic(object):
    """Critic network."""

    def __init__(self,
                 state_dim,
                 action_dim,
                 learning_rate,
                 tau,
                 img_encoder,
                 target_img_encoder,
                 train_img_encoder=True,
                 share_img_encoder=False,
                 name=None):
        """Critic initialization."""

        # member initialization
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.train_img_encoder = train_img_encoder
        self.share_img_encoder = share_img_encoder
        self.img_encoder = img_encoder
        self.target_img_encoder = target_img_encoder

        # build the critic and the target
        scope_name = 'critic' if name is None else name
        with tf.variable_scope(scope_name):
            (self.ob_state,
             self.action,
             self.q_value) = self._build_network(self.img_encoder)
        self.params = tf.trainable_variables(scope=scope_name)
        print_var_info(self.params + self.img_encoder.params, scope_name)
        scope_name += '_target'
        with tf.variable_scope(scope_name):
            (self.target_ob_state,
             self.target_action,
             self.target_q_value) = self._build_network(self.target_img_encoder)
        self.target_params = tf.trainable_variables(scope=scope_name)
        print_var_info(
            self.target_params + self.target_img_encoder.params, scope_name)

        # build operators
        (self.update_target_op,
         self.update_target_img_encoder_op,
         self.y,
         self.train_op,
         self.action_grad) = self._build_ops()

    def _build_network(self, im_encoder):
        """Build network for the critic part."""
        ob_state = tf.placeholder(dtype=tf.float32, shape=[None, self.s_dim])
        observation = tf.concat(values=[im_encoder.embedding, ob_state], axis=1)
        action_ph = tf.placeholder(dtype=tf.float32,
                                   shape=[None, self.a_dim])
        x = tf.concat(values=[observation, action_ph], axis=1)
        x = create_fc_net(x)
        q_value = tf.squeeze(tf.layers.dense(inputs=x, units=1))
        return ob_state, action_ph, q_value

    def _build_ops(self):
        """Build all related tf operations."""
        target_update_op = tf.group([
            x.assign(tf.multiply(y, self.tau) + tf.multiply(x, 1. - self.tau))
            for x, y in zip(self.target_params, self.params)])
        target_im_encoder_update_op = tf.group([
            x.assign(tf.multiply(y, self.tau) + tf.multiply(x, 1. - self.tau))
            for x, y in zip(self.target_img_encoder.params,
                            self.img_encoder.params)])
        y = tf.placeholder(dtype=tf.float32, shape=[None])
        loss = tf.losses.mean_squared_error(y, self.q_value)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        if self.train_img_encoder:
            params = self.params + self.img_encoder.params
        else:
            params = self.params
        print_var_info(params, 'Critic_to_optimize')
        train_op = optimizer.minimize(loss, var_list=params)
        grads = tf.gradients(self.q_value, self.action)
        return target_update_op, target_im_encoder_update_op, y, train_op, grads

    def get_qval(self, sess, observation, action):
        """Get Q-val."""
        image, state = preprocess_observation(observation)
        if np.ndim(action) == 1:
            action = np.expand_dims(action, axis=0)
        return sess.run(self.q_value,
                        feed_dict={self.img_encoder.inputs: image,
                                   self.ob_state: state,
                                   self.action: action})

    def get_target_qval(self, sess, observation, action):
        """Get target network's Q-val."""
        image, state = preprocess_observation(observation)
        if np.ndim(action) == 1:
            action = np.expand_dims(action, axis=0)
        return sess.run(self.target_q_value,
                        feed_dict={self.target_img_encoder.inputs: image,
                                   self.target_ob_state: state,
                                   self.target_action: action})

    def update_target(self, sess):
        """Update the target network's weights."""
        sess.run(self.update_target_op)
        if not self.share_img_encoder:
            sess.run(self.update_target_img_encoder_op)

    def train(self, sess, observation, action, y):
        """Train/Improve critic once."""
        image, state = preprocess_observation(observation)
        sess.run(self.train_op,
                 feed_dict={self.img_encoder.inputs: image,
                            self.ob_state: state,
                            self.action: action,
                            self.y: y})

    def get_action_gradients(self, sess, observation, actions):
        image, state = preprocess_observation(observation)
        return sess.run(self.action_grad,
                        feed_dict={self.img_encoder.inputs: image,
                                   self.ob_state: state,
                                   self.action: actions})
