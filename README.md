# reggy

[![PyPI](https://img.shields.io/pypi/v/reggy.svg?style=flat)](https://pypi.python.org/pypi/reggy)
[![Tests](https://github.com/kpj/reggy/workflows/Tests/badge.svg)](https://github.com/kpj/reggy/actions)

Regressions with arbitrarily complex regularization terms.

Currently supported regularization terms:
* LASSO


## Installation

```bash
$ pip install reggy
```


## Usage

A simple example with LASSO regularization:
```python
import reggy
import numpy as np


alpha = 0.3
beta = 1.7

X = np.random.normal(size=(1000, 1))
y = np.random.normal(X * beta + alpha, size=(1000, 1))

model = reggy.RegReg(X, y, regularizers=[reggy.lasso])
model.fit()

print(model.coef())
## (array([[0.27395004]], dtype=float32), array([[1.2682909]], dtype=float32))
```
