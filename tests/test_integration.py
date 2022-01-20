import random

import numpy as np
import tensorflow as tf

import pytest

import reggy


@pytest.fixture
def seed_rng():
    random.seed(42)
    np.random.seed(42)  # deprecated?
    tf.random.set_seed(42)


def test_simple_case(seed_rng):
    alpha = 0.3
    beta = 1.7

    X = np.random.normal(size=(100, 1))
    y = np.random.normal(X * beta + alpha, size=(100, 1))

    model = reggy.RegReg(X, y)
    fit = model.fit()

    np.testing.assert_allclose(model.intercept_, alpha, rtol=1e-1)
    np.testing.assert_allclose(model.coef_, beta, rtol=1e-1)


def test_gaussian_example(seed_rng):
    # generate data
    N = 1000  # sample count
    K = 1  # covariate count
    x = np.random.normal(size=(N, K))
    alpha_true = np.random.normal()
    beta_true = np.random.normal(size=(K, 1))
    mu_true = x @ beta_true + alpha_true
    y = np.random.normal(mu_true, size=(N, 1))

    # fit model
    model = reggy.RegReg(x, y)
    fit = model.fit()

    # check results
    np.testing.assert_allclose(model.intercept_, alpha_true, rtol=1e-1)
    np.testing.assert_allclose(model.coef_, beta_true, rtol=1e-1)


def test_regularization(seed_rng):
    alpha = 0.3
    beta = np.array([[1.7, -0.2, 5.5, -1.3]]).T

    X = np.random.normal(size=(1000, 4))
    y = np.random.normal(X @ beta + alpha, size=(1000, 1))
    similarity_graph = np.random.random((4, 4))

    model = reggy.RegReg(
        X,
        y,
        family=reggy.gaussian_family,
        regularizers=[
            (0.5, reggy.lasso),
            (0.5, reggy.network_fusion_x(similarity_graph)),
        ],
    )
    fit = model.fit()
