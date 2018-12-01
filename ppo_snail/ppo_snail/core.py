import numpy as np
import tensorflow as tf
import scipy.signal
from gym.spaces import Box, Discrete
from gru import GRU
from CausalConv1D import CausalConv1D
import math

EPS = 1e-8

n = 100
MASK = np.array([[-float('inf') if i>j else 1 for i in range(n)] for j in range(n)])


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

def gru(x, a, rew, rnn_state, n_hidden, n, activation, output_size):
    hidden = tf.concat([x, a, rew], 1)
    # use layer normalization for gru
    gru_cell = GRU(n_hidden, activation=activation)
#    gru_cell = tf.nn.rnn_cell.GRUCell(n_hidden, activation=activation, kernel_initializer=tf.initializers.orthogonal(), bias_initializer=tf.initializers.zeros())
    rnn_in = tf.expand_dims(hidden, [0])
    step_size = tf.minimum(tf.shape(rew)[:1], n)
    gru_outputs, gru_state = tf.nn.dynamic_rnn(
        gru_cell, rnn_in, initial_state=rnn_state, sequence_length=step_size,
        time_major=False)
    state_out = gru_state[:1, :]
    rnn_out = tf.reshape(gru_outputs, [-1, n_hidden])
    out = tf.layers.dense(rnn_out, units=output_size, 
                          kernel_initializer=tf.initializers.glorot_normal(), 
                          bias_initializer=tf.zeros_initializer(),)
    # layer normalization for dense layer
    norm_out = tf.contrib.layers.layer_norm(out)
    return norm_out, state_out
    
def gru_categorical_policy(x, a, rew, rnn_state, n_hidden, n, activation, output_size, action_space):
    act_dim = action_space.n
    logits, state_out = gru(x, a, rew, rnn_state, n_hidden, n, activation, output_size)
    logp_all = tf.nn.log_softmax(logits)
    pi = tf.squeeze(tf.multinomial(logits,1), axis=1)
    logp = tf.reduce_sum(a * logp_all, axis=1)
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)
    return pi, logp, logp_pi, state_out

def gru_actor_critic(x, a, rew, pi_rnn_state, v_rnn_state, n_hidden, n, activation=tf.nn.relu, 
                     output_activation=None, policy=None, action_space=None):
    # only consider discrete experiment now
    if policy is None and isinstance(action_space, Discrete):
        policy = gru_categorical_policy
    act_dim = action_space.n
    a = tf.one_hot(a, depth=act_dim)

    with tf.variable_scope('pi'):
        pi, logp, logp_pi, new_pi_rnn_state = policy(x, a, rew, 
                                                     pi_rnn_state, n_hidden, 
                                                     n, activation, act_dim, 
                                                     action_space)
    with tf.variable_scope('v'):
        v, new_v_rnn_state = gru(x, a, rew, v_rnn_state, n_hidden, n, activation, 1)
        v = tf.squeeze(v, axis=1)
    return pi, logp, logp_pi, v, new_pi_rnn_state, new_v_rnn_state

def dense_block(inputs, dilation_rate, num_filter):
    kernel_size = 2
#    xf = CausalConv1D(inputs, num_filter, kernel_size, dilation_rate=dilation_rate)
#    xg = CausalConv1D(inputs, num_filter, kernel_size, dilation_rate=dilation_rate)
    xf = tf.keras.layers.Conv1D(num_filter, kernel_size, padding="causal", dilation_rate=dilation_rate)(inputs)
    xg = tf.keras.layers.Conv1D(num_filter, kernel_size, padding="causal", dilation_rate=dilation_rate)(inputs)
    activations = tf.multiply(tf.nn.tanh(xf), tf.nn.sigmoid(xg))
    return tf.concat([inputs, activations], 2)

def tcb_lock(inputs, seq_length, num_filter): 
    for i in range(int(math.ceil(math.log2(seq_length)))):
        inputs = dense_block(inputs, 2**(i+1), num_filter)
    return inputs
        
def attention_block(inputs, key_size, value_size):
    keys = tf.layers.dense(inputs, key_size)
    query = tf.layers.dense(inputs, key_size)
    print("shape of query:", query.shape)
    logits = tf.matmul(query, tf.transpose(keys, perm=[0, 2, 1]))
    print("shape of logits:", logits.shape)
    batch_mask = tf.cast(tf.tile(tf.expand_dims(MASK, 0), [tf.shape(inputs)[0], 1, 1]), tf.float32)
    # batch_mask = np.repeat(MASK[np.newaxis, :, :], keys.shape[0], axis=0)
    print("shape of batch_mask:", batch_mask.shape)
    print("type of batch_mask:", batch_mask.dtype)
    print("type of logits:", logits.dtype)
    mask_logits = tf.where(tf.equal(batch_mask, 1), logits, batch_mask)
    print("shape of mask_logits:", mask_logits.shape)
#    mask_logits = tf.keras.layers.Masking(mask_value=0)(logits)
    probs = tf.nn.softmax(mask_logits / math.sqrt(key_size))
    values = tf.layers.dense(inputs, value_size)
    read = tf.matmul(probs, values)
    return tf.concat([inputs, read], 2)

def snail_bandit(a, rew, seq_length, action_space):
    act_dim = action_space.n
    a = tf.one_hot(a, depth=act_dim)
    rew = tf.expand_dims(rew, -1)
    # inputs shape: batch_size * sequence_length * input_dimensionality
    inputs = tf.concat([rew, a], 2)
    print("shape of inputs:", inputs.shape)
    input_layer = tf.layers.dense(inputs, units=32, 
                          kernel_initializer=tf.initializers.glorot_normal(), 
                          bias_initializer=tf.zeros_initializer(),)
    print("shape of input_layer:", input_layer.shape)
    with tf.variable_scope('pi'):
        # num of filters = 32
        policy_net = tcb_lock(input_layer, seq_length, 32)
        policy_net = tcb_lock(policy_net, seq_length, 32)
        # key_size = 32 and value_size = 32
        policy_net = attention_block(policy_net, 32, 32)
        print("shaple of policy_net:", policy_net.shape)
        logits = tf.layers.dense(policy_net, act_dim)
        print("shaple of logits:", logits.shape)
        logits = tf.reshape(logits,[-1, act_dim])
        print("shaple of logits after reshape:", logits.shape)
        logp_all = tf.nn.log_softmax(logits)
        print("shaple of logp_all:", logp_all.shape)
        pi = tf.squeeze(tf.multinomial(logits, 1), axis=1)
        print("shaple of pi:", pi.shape)
        logp = tf.reduce_sum(tf.reshape(a, [-1, act_dim]) * logp_all, axis=1)
        print("shaple of logp:", logp.shape)
        print("shape of one hot pi:", tf.one_hot(pi, depth=act_dim).shape)
        logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)
        print("shaple of logp_pi:", logp_pi.shape)

    with tf.variable_scope('v'):
        value_net = tcb_lock(input_layer, seq_length, 16)
        value_net = tcb_lock(value_net, seq_length, 16)
        # key_size = 16 and value_size = 16
        value_net = attention_block(value_net, 16, 16)
        v = tf.layers.dense(value_net, 1)
        print("shape of v:", v.shape)
        v = tf.reshape(tf.squeeze(v, axis=2), [-1])
        print("shape of v after squeeze:", v.shape)
    return pi, logp, logp_pi, v

def snail_actor_critic(a, rew, seq_length, policy=None, action_space=None):
    # only consider discrete experiment now
    if policy is None and isinstance(action_space, Discrete):
        policy = snail_bandit
    pi, logp, logp_pi, v = policy(a, rew, seq_length, action_space)
    return pi, logp, logp_pi, v