import tensorflow as tf

from .utils import init_vector, init_matrix


class Model(tf.keras.Model):
    def __init__(self, p, q, family, regularizers):
        super(Model, self).__init__(p, q, family)

        self.alpha = init_vector(q)
        self.beta = init_matrix(p, q)
        self.family = family

        self.regularizers = regularizers or []

    def call(self, x):
        # forward pass
        eta = self._linear_predictor(self.alpha, self.beta, x)

        # apply regularization terms
        for reg in self.regularizers:
            # TODO: add regularization parameters
            self.add_loss(reg(self.alpha, self.beta, self.family))

        return self.family["linkinv"](eta)

    def _linear_predictor(self, alpha, beta, x):
        eta = tf.matmul(x, beta) + alpha
        return eta
