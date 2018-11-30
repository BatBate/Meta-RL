from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def dense(x, num_units, nonlinearity=None, use_weight_normalization=True, name=''):

    with tf.variable_scope(name):
        V = tf.get_variable('V', shape=[int(x.get_shape()[1]),num_units], dtype=tf.float32,
                              initializer=tf.initializers.glorot_normal(), trainable=True)
        b = tf.get_variable('b', shape=[num_units], dtype=tf.float32,
                              initializer=tf.zeros_initializer(), trainable=True)

        if use_weight_normalization:
            g = tf.get_variable('g', shape=[num_units], dtype=tf.float32,
                              initializer=tf.zeros_initializer(), trainable=True)
            x = tf.matmul(x, V)
            scaler = g/tf.sqrt(tf.reduce_sum(tf.square(V),[0]))
            x = tf.reshape(scaler,[1,num_units])*x
            b = tf.reshape(b,[1,num_units])
            x = x + b

        # apply nonlinearity
        if nonlinearity is not None:
            x = nonlinearity(x)

        return x


class GRU(tf.contrib.rnn.RNNCell):

    def __init__(
            self, size, activation=tf.tanh, reuse=None,
            normalizer=tf.contrib.layers.layer_norm,
            initializer=tf.initializers.orthogonal()):
        super(GRU, self).__init__(_reuse=reuse)
        self._size = size
        self._activation = activation
        self._normalizer = normalizer
        self._initializer = initializer

    @property
    def state_size(self):
        return self._size

    @property
    def output_size(self):
        return self._size

    def call(self, input_, state):
        update, reset = tf.split(self._forward(
            'update_reset', [state, input_], 2 * self._size, tf.nn.sigmoid,
            bias_initializer=tf.constant_initializer(-1.)), 2, 1)
        candidate = self._forward(
            'candidate', [reset * state, input_], self._size, self._activation)
        state = (1 - update) * state + update * candidate
        return state, state

    def _forward(self, name, inputs, size, activation, **kwargs):
        with tf.variable_scope(name):
            return _forward(
                inputs, size, activation, normalizer=self._normalizer,
                weight_initializer=self._initializer, **kwargs)


class WeightedNormGRUCell(tf.contrib.rnn.RNNCell):

    def __init__(self,
                 num_units,
                 norm=True,
                 activation=None,
                 reuse=None,
                 name=None,
                 kernel_initializer=tf.initializers.orthogonal(),
                 bias_initializer=tf.initializers.zeros(),
                 dtype=None):
        super(WeightedNormGRUCell, self).__init__(_reuse=reuse, name=name, dtype=dtype)
        self._num_units = num_units
        if activation:
            self._activation = activation
        else:
            self._activation = tf.math.tanh()
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._norm = norm

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def weightbuild(self, inputs_shape, norm):
        if inputs_shape[-1] is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                           % str(inputs_shape))

        input_depth = inputs_shape[-1]
        with tf.variable_scope('update_reset'):
            self._gate_kernel = tf.get_variable(
                "gates_kernel",
                shape=[input_depth + self._num_units, 2 * self._num_units],
                initializer=self._kernel_initializer)
            if norm:
                gate_wn = [self._normalize(self._gate_kernel[0:input_depth, :],
                                          name='gate_norm_input'),
                            self._normalize(self._gate_kernel[input_depth:input_depth+self._num_units, :],
                                          name='gate_norm_state')]
                self._gate_kernel = tf.concat(gate_wn, axis=0)
            self._gate_bias = tf.get_variable(
                "gates_bias",
                shape=[2 * self._num_units],
                initializer=(
                    self._bias_initializer
                    if self._bias_initializer is not None
                    else tf.constant_initializer(1.0, dtype=self.dtype)))
        with tf.variable_scope('candidate'):
            self._candidate_kernel = tf.get_variable(
                "candidate_kernel",
                shape=[input_depth + self._num_units, self._num_units],
                initializer=self._kernel_initializer)
            if norm:
                candidate_wn = [self._normalize(self._candidate_kernel[0:input_depth, :],
                                                name='candidate_norm_input'),
                                self._normalize(self._candidate_kernel[input_depth:input_depth+self._num_units, :],
                                                name='candidate_norm_state')]
                self._candidate_kernel = tf.concat(candidate_wn, axis=0)
            self._candidate_bias = tf.get_variable(
                "candidate_bias",
                shape=[self._num_units],
                initializer=(
                    self._bias_initializer
                    if self._bias_initializer is not None
                    else tf.zeros_initializer(dtype=self.dtype)))

        self.built = True

    def _normalize(self, weights, name):
        output_size = weights.get_shape().as_list()[1]
        g = tf.get_variable(name, [output_size], dtype=weights.dtype)
        return tf.nn.l2_normalize(weights, axis=0) * g

    def call(self, inputs, state):
        self.weightbuild(inputs.get_shape().as_list(), self._norm)
        gate_inputs = tf.matmul(
            tf.concat([inputs, state], 1), self._gate_kernel)
        gate_inputs = tf.nn.bias_add(gate_inputs, self._gate_bias)

        value = tf.sigmoid(gate_inputs)
        r, u = tf.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state

        candidate = tf.matmul(
            tf.concat([inputs, r_state], 1), self._candidate_kernel)
        candidate = tf.nn.bias_add(candidate, self._candidate_bias)

        c = self._activation(candidate)
        new_h = u * state + (1 - u) * c
        return new_h, new_h


def _forward(
        inputs, size, activation, normalizer=tf.contrib.layers.layer_norm,
        weight_initializer=tf.initializers.orthogonal(),
        bias_initializer=tf.initializers.zeros()):
    if not isinstance(inputs, (tuple, list)):
        inputs = (inputs,)
    shapes = []
    outputs = []
    # Map each input to individually normalize their outputs.
    for index, input_ in enumerate(inputs):
        shapes.append(input_.shape[1: -1].as_list())
        input_ = tf.contrib.layers.flatten(input_)
        weight = tf.get_variable(
            'weight_{}'.format(index + 1), (int(input_.shape[1]), size),
            tf.float32, weight_initializer)
        output = tf.matmul(input_, weight)
        if normalizer:
            output = normalizer(output)
        outputs.append(output)
    output = tf.reduce_mean(outputs, 0)
    # Add bias after normalization.
    bias = tf.get_variable(
        'weight', (size,), tf.float32, bias_initializer)
    output += bias
    # Activation function.
    if activation:
        output = activation(output)
    # Restore shape dimensions that are consistent among inputs.
    min_dim = min(len(shape[1:]) for shape in shapes)
    dim_shapes = [[shape[dim] for shape in shapes] for dim in range(min_dim)]
    matching_dims = ''.join('NY'[len(set(x)) == 1] for x in dim_shapes) + 'N'
    agreement = matching_dims.index('N')
    remaining = sum(np.prod(shape[agreement:]) for shape in shapes)
    if agreement:
        batch_size = output.shape[0].value or -1
        shape = [batch_size] + shapes[:agreement] + [remaining]
        output = tf.reshape(output, shape)
    return output
