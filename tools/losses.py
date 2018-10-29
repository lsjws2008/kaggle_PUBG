import tensorflow as tf


def mse_loss(input, target):
    target = tf.expand_dims(target, 1)
    loss = tf.losses.mean_squared_error(target,
                                        input)
    return loss
