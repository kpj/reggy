import tensorflow as tf
import tensorflow_probability as tfp


def gaussian_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


gaussian_family = {
    "link": lambda x: x,
    "linkinv": lambda x: x,
    "loss": gaussian_loss,
}


def binomial_loss(y_true, y_pred):
    obj = 0
    for j in range(y_true.shape[1]):
        prob = tfp.distributions.Bernoulli(probs=y_pred[:, j], validate_args=True)
        obj += tf.reduce_sum(prob.log_prob(y_true[:, j]))
    return -obj


binomial_family = {
    "link": lambda p: tf.math.log(p / (1 - p)),  # logit
    "linkinv": lambda x: 1 / (1 + tf.exp(-x)),  # logistic
    "loss": binomial_loss,
}
