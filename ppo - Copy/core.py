import numpy as np
import tensorflow as tf
import scipy.signal
from gym.spaces import Box, Discrete

EPS = 1e-8

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=combined_shape(None,dim))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def placeholder_from_space(space):
    if isinstance(space, Box):
        return placeholder(space.shape)
    elif isinstance(space, Discrete):
        return tf.placeholder(dtype=tf.int32, shape=(None,), name='discrete_space_ph')
    raise NotImplementedError

def placeholders_from_spaces(*args):
    return [placeholder_from_space(space) for space in args]

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def gru(x, a, r, num_gru_units, num_out, activation=None, kernel_initializer=None, output_activation=None):
    rnn_in = tf.concat([x,a,r], 1)
    rnn_in = tf.expand_dims(rnn_in, 0)
    sequence_length = tf.shape(x)[:1]
    gru_layer = tf.contrib.rnn.GRUCell(num_gru_units, activation=activation, kernel_initializer=kernel_initializer)
    gru_output, gru_state = tf.nn.dynamic_rnn(gru_layer, rnn_in, sequence_length = sequence_length, dtype=tf.float32)
    gru_output = tf.reshape(gru_output, [-1, num_gru_units])
    return tf.layers.dense(gru_output, units=num_out, activation=output_activation)


def get_vars(scope=''):
    return [x for x in tf.trainable_variables() if scope in x.name]

def count_vars(scope=''):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


"""
Policies
"""

def mlp_categorical_policy(x, a, hidden_sizes, activation, output_activation, action_space):
    act_dim = action_space.n
    logits = mlp(x, list(hidden_sizes)+[act_dim], activation, None)
    logp_all = tf.nn.log_softmax(logits)
    pi = tf.squeeze(tf.multinomial(logits,1), axis=1)
    logp = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logp_all, axis=1)
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)
    return pi, logp, logp_pi

def gru_categorical_policy(x, a, r, num_gru_units, activation, output_activation, action_space):
    '''

    :param x: input
    :param a: actions from last time
    :param hidden_sizes: list of hidden layer sizes
    :param activation: activation func
    :param output_activation: output layer activation func
    :param action_space: action space as in gym
    :return: (current actions, sum of log probabilities, sum of log probabilities on old actions)
    '''

    act_dim = action_space.n
    logits = gru(x, a, r, num_gru_units, act_dim, activation=activation, kernel_initializer=tf.initializers.orthogonal, output_activation=output_activation)
    logp_all = tf.nn.log_softmax(logits)
    pi = tf.squeeze(tf.multinomial(logits,1, output_dtype=tf.int32), axis=1)
    logp = tf.reduce_sum(a * logp_all, axis=1)
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)
    return pi, logp, logp_pi

def gru_gaussian_policy(x, a, r, num_gru_units, activation, output_activation, action_space):
    raise NotImplementedError

    


def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation, action_space):
    act_dim = a.shape.as_list()[-1]
    mu = mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)
    log_std = tf.get_variable(name='log_std', initializer=-0.5*np.ones(act_dim, dtype=np.float32))
    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp = gaussian_likelihood(a, mu, log_std)
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return pi, logp, logp_pi


"""
Actor-Critics
"""
def mlp_actor_critic(x, a, hidden_sizes=(64,64), activation=tf.tanh, 
                     output_activation=None, policy=None, action_space=None):

    # default policy builder depends on action space
    if policy is None and isinstance(action_space, Box):
        policy = mlp_gaussian_policy
    elif policy is None and isinstance(action_space, Discrete):
        policy = mlp_categorical_policy

    with tf.variable_scope('pi'):
        pi, logp, logp_pi = policy(x, a, hidden_sizes, activation, output_activation, action_space)
    with tf.variable_scope('v'):
        v = tf.squeeze(mlp(x, list(hidden_sizes)+[1], activation, None), axis=1)
    return pi, logp, logp_pi, v

def gru_actor_critic(x, a, r, num_gru_units=256, activation=tf.nn.relu, output_activation=tf.nn.relu, policy=None, action_space=None):
    if policy is None and isinstance(action_space, Box):
        policy = gru_gaussian_policy
    elif policy is None and isinstance(action_space, Discrete):
        policy = gru_categorical_policy
    a = tf.one_hot(a, depth=action_space.n)

    with tf.variable_scope('pi'):
        pi, logp, logp_pi = policy(x, a, r, num_gru_units, activation, output_activation, action_space)
    with tf.variable_scope('v'):
        v = tf.squeeze(gru(x, a, r, num_gru_units, 1, activation, tf.initializers.orthogonal, output_activation=None), axis=1)
        #v = tf.squeeze(mlp(x, [256]+[1], activation, None), axis=1)

        # v = tf.squeeze(gru(x, a, r, num_gru_units, 1, activation, tf.initializers.orthogonal, output_activation=None), axis=1)
    return pi, logp, logp_pi, v
    #def gru(x, n_hidden=256, activation=tf.nn.relu, output_activation=None):
    #
    #def gru_categorical_policy(x, a, hidden_sizes, activation, output_activation, action_space):
    #    act_dim = action_space.n
    #    logits = mlp(x, list(hidden_sizes)+[act_dim], activation, None)
    #    logp_all = tf.nn.log_softmax(logits)
    #    pi = tf.squeeze(tf.multinomial(logits,1), axis=1)
    #    logp = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logp_all, axis=1)
    #    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)
    #    return pi, logp, logp_pi
    #
    #def gru_actor_critic(x, a, hidden_sizes=(64,64), activation=tf.tanh,
    #                     output_activation=None, policy=None, action_space=None):
    #
    #    if policy is None and isinstance(action_space, Discrete):
    #        policy = gru_categorical_policy
    #
    #    with tf.variable_scope('pi'):
    #        pi, logp, logp_pi = policy(x, a, hidden_sizes, activation, output_activation, action_space)
    #    with tf.variable_scope('v'):
    #        v = tf.squeeze(mlp(x, list(hidden_sizes)+[1], activation, None), axis=1)
    #    return pi, logp, logp_pi, v