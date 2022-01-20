import tensorflow as tf


def lasso(alpha, beta, family):
    return tf.reduce_sum(tf.abs(beta))


# TODO: add network fusion penalty
