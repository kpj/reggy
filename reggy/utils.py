import tensorflow as tf


def init_matrix(m, n):
    initializer = tf.keras.initializers.glorot_normal(42)
    return tf.Variable(initializer(shape=(m, n)), trainable=True)


def init_vector(m):
    initializer = tf.keras.initializers.glorot_normal(42)
    return tf.Variable(initializer(shape=(m, 1)), trainable=True)
