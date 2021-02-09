# Copyright (c) 2020, Ioana Bica

import numpy as np
import tensorflow as tf


def equivariant_layer(x, h_dim, layer_id, treatment_id):
    xm = tf.reduce_sum(x, axis=1, keepdims=True)

    l_gamma = tf.layers.dense(x, h_dim, activation=None,
                              name='eqv_%s_treatment_%s_gamma' % (str(layer_id), str(treatment_id)),
                              reuse=tf.AUTO_REUSE)
    l_lambda = tf.layers.dense(xm, h_dim, activation=None, use_bias=False,
                               name='eqv_%s_treatment_%s_lambda' % (str(layer_id), str(treatment_id)),
                               reuse=tf.AUTO_REUSE)
    out = l_gamma - l_lambda
    return out


def invariant_layer(x, h_dim, treatment_id):
    rep_layer_1 = tf.layers.dense(x, h_dim, activation=tf.nn.elu,
                                  name='inv_treatment_%s' % str(treatment_id),
                                  reuse=tf.AUTO_REUSE)
    rep_sum = tf.reduce_sum(rep_layer_1, axis=1)

    return rep_sum


def sample_Z(m, n):
    return np.random.uniform(0, 1., size=[m, n])


def sample_X(X, size):
    start_idx = np.random.randint(0, X.shape[0], size)
    return start_idx


def sample_dosages(batch_size, num_treatments, num_dosages):
    dosage_samples = np.random.uniform(0., 1., size=[batch_size, num_treatments, num_dosages])
    return dosage_samples
