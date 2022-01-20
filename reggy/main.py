import tensorflow as tf

from .model import Model
from .families import gaussian_family


class RegReg:
    def __init__(self, x, y, family=gaussian_family, regularizers=None):
        self.x = x
        self.y = y

        self.model = Model(
            self.x.shape[1], self.y.shape[1], family, regularizers=regularizers
        )
        self.model.compile(
            optimizer="adam",
            loss=lambda y_true, y_pred: self._loss_wrapper(
                y_true, y_pred, family["loss"]
            ),
        )

    def fit(self, epochs=100000):
        es = tf.keras.callbacks.EarlyStopping(
            monitor="loss", mode="min", verbose=1, patience=50
        )
        return self.model.fit(self.x, self.y, epochs=epochs, callbacks=[es])

    def _loss_wrapper(self, y_true, y_pred, loss_func):
        return loss_func(y_true, y_pred)

    @property
    def intercept_(self):
        return self.model.alpha.numpy()

    @property
    def coef_(self):
        return self.model.beta.numpy()
