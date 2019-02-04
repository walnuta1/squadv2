"""
Implementation of Highway layers. Reference:
https://arxiv.org/pdf/1505.00387.pdf
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import utils

def highway(
        x_in, net_size, activation_fn=utils.get_activation("relu"),
        carry_offset=-1.0, initializer_range=0.2, layer_id=None):
    """To build a highway layer"""
    with tf.variable_scope("highway" + ("_" + str(layer_id) if layer_id else "")):
        h_val = tf.layers.dense(
            x_in,
            net_size,
            activation=activation_fn,
            name="transform",
            kernel_initializer=tf.truncated_normal_initializer(stddev=initializer_range)
        )
        t_val = tf.layers.dense(
            x_in,
            net_size,
            activation=utils.get_activation("sigmoid"),
            name="gate",
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.constant_initializer(carry_offset)
        )
        c_val = tf.math.subtract(1.0, t_val, name="carry")
        return x_in * c_val + h_val * t_val

def multi_highway(layer_count, x_in, net_size,
                  activation_fn=utils.get_activation("relu"),
                  carry_offset=-1.0, initializer_range=0.2):
    """Construct multiple highway layers"""
    y_out = x_in
    for lid in range(layer_count):
        y_out = highway(
            y_out, net_size, activation_fn=activation_fn, carry_offset=carry_offset,
            initializer_range=initializer_range, layer_id=lid)
    return y_out
