import tensorflow as tf

import os,\
    sys,\
    inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from global_config.global_config import config
cfg = config()
weight_decay = cfg.weight_decay

def weight_variable(name, shape, init=None):
    if init is not None:
        initial = tf.get_variable(name, shape,
                                  initializer = tf.constant_initializer(value = init),
                                  regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
    else:
        initial = tf.get_variable(name = name,shape=shape,
           initializer=tf.contrib.layers.xavier_initializer())
    return tf.Variable(initial)

def bias_variable(name, shape, init = None):
    if init is not None:
        initial = tf.get_variable(name, shape=shape,
                              initializer=tf.constant_initializer(value = init))
    else:
        initial = initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv1d(inputs, filter_shape, name):

    output = tf.nn.conv1d(inputs,
                          weight_variable('w_'+name, filter_shape),
                          1,
                          'SAME')
    output = tf.nn.relu(output+bias_variable('b_'+name,
                                             [filter_shape[-1]]))
    return output


def max_pooling1d(inputs):

    return tf.layers.max_pooling1d(inputs,
                                   2,
                                   2,
                                   padding='same')


def fc(inputs, shape, name):

    inputs = tf.nn.relu(tf.matmul(inputs, weight_variable('w_'+name, shape))
                       + bias_variable('b_'+name, [shape[-1]]))
    inputs = tf.nn.dropout(inputs, keep_prob=cfg.dropout)

    return inputs


def batch_norm(x, n_out):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)
        """
        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        """
        def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = mean_var_with_update()
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed
