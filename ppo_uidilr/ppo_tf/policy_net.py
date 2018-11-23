import gym
import numpy as np
import tensorflow as tf
from tensorflow import nn


class Policy_net:
    def __init__(self, name: str, env, temp=0.1):
        """
        :param name: string
        :param env: gym env
        :param temp: temperature of boltzmann distribution
        """

        ob_space = env.observation_space
        act_space = env.action_space

        with tf.variable_scope(name):
            self.obs = tf.placeholder(dtype=tf.float32, shape=[None] + list(ob_space.shape), name='obs')

            with tf.variable_scope('policy_net'):
                layer_1 = tf.layers.dense(inputs=self.obs, units=20, activation=tf.tanh)
                layer_2 = tf.layers.dense(inputs=layer_1, units=20, activation=tf.tanh)
                layer_3 = tf.layers.dense(inputs=layer_2, units=act_space.n, activation=tf.tanh)
                self.act_probs = tf.layers.dense(inputs=tf.divide(layer_3, temp), units=act_space.n, activation=tf.nn.softmax)

            with tf.variable_scope('value_net'):
                layer_1 = tf.layers.dense(inputs=self.obs, units=20, activation=tf.tanh)
                layer_2 = tf.layers.dense(inputs=layer_1, units=20, activation=tf.tanh)
                self.v_preds = tf.layers.dense(inputs=layer_2, units=1, activation=None)

            self.act_stochastic = tf.multinomial(tf.log(self.act_probs), num_samples=1)
            self.act_stochastic = tf.reshape(self.act_stochastic, shape=[-1])

            self.act_deterministic = tf.argmax(self.act_probs, axis=1)

            self.scope = tf.get_variable_scope().name

    def act(self, obs, stochastic=True):
        if stochastic:
            return tf.get_default_session().run([self.act_stochastic, self.v_preds], feed_dict={self.obs: obs})
        else:
            return tf.get_default_session().run([self.act_deterministic, self.v_preds], feed_dict={self.obs: obs})

    def get_action_prob(self, obs):
        return tf.get_default_session().run(self.act_probs, feed_dict={self.obs: obs})

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)


class PolicyGRUNet:
    def __init__(self, name: str, env, temp=0.1):
        """
        :param name: string
        :param env: gym env
        :param temp: temperature of boltzmann distribution
        """

        ob_space = env.observation_space
        act_space = env.action_space
        # policy_state = tf.placeholder(tf.float32, [1, 256], name='pi_state')
        # value_state = tf.placeholder(tf.float32, [1, 256],name='v_state')

        with tf.variable_scope(name):
            self.obs = tf.placeholder(dtype=tf.float32, shape=[None] + list(ob_space.shape), name='obs')
            rnn_in = tf.expand_dims(self.obs, [0])

            with tf.variable_scope('policy_net'):
                gru_cell = nn.rnn_cell.GRUCell(256, activation=nn.relu, kernel_initializer=tf.initializers.orthogonal(), bias_initializer=tf.zeros_initializer())
                outputs, states = nn.dynamic_rnn(gru_cell, inputs=rnn_in, dtype=tf.float32)
                outputs = tf.reshape(outputs, [-1, 256])
                self.act_probs = tf.layers.dense(outputs, activation=None, units=act_space.n, kernel_initializer=tf.glorot_normal_initializer(), bias_initializer=tf.zeros_initializer(),)
                self.policy_states = states

            with tf.variable_scope('value_net'):
                gru_cell = nn.rnn_cell.GRUCell(256, activation=nn.relu, kernel_initializer=tf.initializers.orthogonal(), bias_initializer=tf.zeros_initializer())
                outputs, states = nn.dynamic_rnn(gru_cell, inputs=rnn_in, dtype=tf.float32)
                outputs = tf.reshape(outputs, [-1, 256])
                self.v_preds = tf.layers.dense(outputs, units=1, activation=None, kernel_initializer=tf.glorot_normal_initializer(), bias_initializer=tf.zeros_initializer())
                self.value_states = states

            self.act_stochastic = tf.multinomial(tf.nn.log_softmax(self.act_probs), num_samples=1)
            self.act_stochastic = tf.reshape(self.act_stochastic, shape=[-1])

            self.act_deterministic = tf.argmax(self.act_probs, axis=1)

            self.scope = tf.get_variable_scope().name

    def act(self, obs, stochastic=True):
        if stochastic:
            return tf.get_default_session().run([self.act_stochastic, self.v_preds], feed_dict={self.obs: obs})
        else:
            return tf.get_default_session().run([self.act_deterministic, self.v_preds], feed_dict={self.obs: obs})

    def get_action_prob(self, obs):
        return tf.get_default_session().run(self.act_probs, feed_dict={self.obs: obs})

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
