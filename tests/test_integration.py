import numpy as np

import reggy


def test_simple_case():
    alpha = 0.3
    beta = 1.7

    X = np.random.normal(size=(1000, 1))
    y = np.random.normal(X * beta + alpha, size=(1000, 1))

    model = reggy.RegReg(X, y)
    fit = model.fit()

    np.testing.assert_allclose(model.model.alpha, alpha, rtol=1e-1)
    np.testing.assert_allclose(model.model.beta, beta, rtol=1e-1)


def test_gaussian_example():
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
    np.testing.assert_allclose(model.model.alpha, alpha_true, rtol=1e-1)
    np.testing.assert_allclose(model.model.beta, beta_true, rtol=1e-1)


def test_regularization():
    alpha = 0.3
    beta = 1.7

    X = np.random.normal(size=(1000, 1))
    y = np.random.normal(X * beta + alpha, size=(1000, 1))

    model = reggy.RegReg(X, y, family=reggy.gaussian_family, regularizers=[reggy.lasso])
    fit = model.fit()
