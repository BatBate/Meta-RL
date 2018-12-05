import numpy as np
import tensorflow as tf
import scipy.signal
from gym.spaces import Box, Discrete
from gru import GRU
from block import WeightedNormGRUCell, dense

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
        return tf.placeholder(dtype=tf.int32, shape=(None,))
    raise NotImplementedError

def placeholders_from_spaces(*args):
    return [placeholder_from_space(space) for space in args]

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)       

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

def gru(x, a, rew, rnn_state, n_hidden, max_sequence_length, activation, output_size, seq_len, action_encoding_matrix):
    # Determine the sequence length dynamically based on where x starts to be all zeros

    # First dimension is the batch size
    # tmp_var = tf.reshape(x, shape=[-1, max_sequence_length, tf.shape(x)[1]])
    tmp_var = tf.reshape(x, shape=[-1, max_sequence_length, x.get_shape()[1]])

    # Determine length; assuming 0 padding
    # https://danijar.com/variable-sequence-lengths-in-tensorflow/
    # used = tf.sign(tf.reduce_max(tf.abs(tmp_var), 2))
    # length = tf.reduce_sum(used, 1)
    # seq_length_vec = tf.cast(length, tf.int32)
    # hidden = tf.concat([x, a, rew], 1)
    # hidden = tf.concat([1 * x], 1)
    action_embedded = tf.matmul(a, action_encoding_matrix)
    seq_length_vec = seq_len
    # hidden = tf.concat([x, action_embedded, rew], axis=1)
    hidden = x

    # use layer normalization for gru
    gru_cell = WeightedNormGRUCell(n_hidden, activation=activation)
    # gru_cell = tf.nn.rnn_cell.GRUCell(n_hidden, activation=activation, kernel_initializer=tf.initializers.orthogonal(), bias_initializer=tf.initializers.zeros())

    # rnn_in = tf.reshape(hidden, [-1, max_sequence_length, tf.shape(hidden)[1]])
    # rnn_in = tf.expand_dims(hidden, [0])
    rnn_in = tf.reshape(hidden, shape=[-1, max_sequence_length, hidden.get_shape()[1]])
    # step_size = tf.minimum(tf.shape(rew)[:1], max_sequence_length)
    gru_outputs, gru_state = tf.nn.dynamic_rnn(
        gru_cell, rnn_in, initial_state=rnn_state, sequence_length=seq_length_vec,
        time_major=False)
    state_out = gru_state[:1, :]
    rnn_out = tf.reshape(gru_outputs, [-1, n_hidden])

    # layer normalization for dense layer
    # norm_out = tf.contrib.layers.layer_norm(out)
    return rnn_out, state_out, seq_length_vec, tmp_var
    
def gru_categorical_policy(x, a, rew, rnn_state, n_hidden, max_sequence_length, activation, output_size, action_space,seq_len, action_encoding_matrix):
    act_dim = action_space.n
    rnn_out, state_out, seq_len_vec, tmp_var = gru(x, a, rew, rnn_state, n_hidden, max_sequence_length, activation, output_size, seq_len, action_encoding_matrix)
    logits = tf.layers.dense(rnn_out, units=output_size,
                          kernel_initializer=tf.initializers.glorot_uniform(),
                          bias_initializer=tf.zeros_initializer())
    # logits = dense(rnn_out, num_units=output_size, name='pi_out_layer')
    # logits = tf.contrib.layers.layer_norm(logits)
    # logits = pi_out

    v = tf.layers.dense(rnn_out, units=1,
                          kernel_initializer=tf.initializers.glorot_uniform(),
                          bias_initializer=tf.zeros_initializer(),)
    v = tf.squeeze(v, axis=1)
    # v = dense(rnn_out, num_units=1, name='v_out_layer')
    # v = tf.contrib.layers.layer_norm(v)
    # v = v_out

    logp_all = tf.nn.log_softmax(logits)
    pi = tf.squeeze(tf.multinomial(logits,1), axis=1)
    logp = tf.reduce_sum(a * logp_all, axis=1)
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)
    return pi, v, logp, logp_pi, state_out, logits, seq_len_vec, tmp_var

def gru_actor_critic(x, a, rew, rnn_state, n_hidden, max_sequence_length, action_encoding_matrix, seq_len, activation=tf.nn.relu,
                     output_activation=None, policy=None, action_space=None):
    # only consider discrete experiment now
    if policy is None and isinstance(action_space, Discrete):
        policy = gru_categorical_policy
    act_dim = action_space.n
    a = tf.one_hot(a, depth=act_dim)

    with tf.variable_scope('pi-v'):
        pi, v, logp, logp_pi, rnn_state, logits, seq_len_vector, tmp_vec = policy(x, a, rew,
                                                         rnn_state, n_hidden,
                                                         max_sequence_length, activation, act_dim,
                                                         action_space, seq_len, action_encoding_matrix)
    # with tf.variable_scope('v'):
    #     v, new_v_rnn_state = gru(x, a, rew, v_rnn_state, n_hidden, max_sequence_length, activation, 1, seq_len)
    #     v = tf.squeeze(v, axis=1)
    return pi, logp, logp_pi, v, rnn_state, logits, seq_len_vector, tmp_vec

def denseblock(x, a, rew, dilation_rate, num_filter, action_space):
    act_dim = action_space.n
    a = tf.one_hot(a, depth=act_dim)
    input = tf.concat([x, a, rew], 1)
    xf = tf.keras.layers.Conv1D(input.shape[-1], num_filter, dilation_rate=dilation_rate)
    xg = tf.keras.layers.Conv1D(input.shape[-1], num_filter, dilation_rate=dilation_rate)
    activations = tf.tanh(xf) * tf.sigmoid(xg)
    return tf.concat([input, activations], 1)

#def tcblock(x, a, rew, seq_length, num_filter):
    