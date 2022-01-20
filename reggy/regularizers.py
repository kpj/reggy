import tensorflow as tf


def ridge(alpha, beta, family):
    return tf.reduce_sum(tf.square(beta))


def lasso(alpha, beta, family):
    return tf.reduce_sum(tf.abs(beta))


def network_fusion_x(graph):
    graph = tf.cast(graph, tf.float32)

    def tmp(alpha, beta, family):
        return tf.linalg.trace(tf.matmul(tf.transpose(beta), tf.matmul(graph, beta)))

    return tmp


def network_fusion_y(graph):
    graph = tf.cast(graph, tf.float32)

    def tmp(alpha, beta, family):
        return tf.linalg.trace(tf.matmul(beta, tf.matmul(graph, tf.transpose(beta))))

    return tmp
