# regreg

[![PyPI](https://img.shields.io/pypi/v/regreg.svg?style=flat)](https://pypi.python.org/pypi/regreg)
[![Tests](https://github.com/kpj/regreg/workflows/Tests/badge.svg)](https://github.com/kpj/regreg/actions)

Regressions with arbitrarily complex regularization terms.

Currently supported regularization terms:
* LASSO


## Installation

```bash
$ pip install regreg
```


## Usage

A simple example with LASSO regularization:
```python
import regreg
import numpy as np


alpha = 0.3
beta = 1.7

X = np.random.normal(size=(1000, 1))
y = np.random.normal(X * beta + alpha, size=(1000, 1))

model = regreg.RegReg(X, y, regularizers=[regreg.lasso])
model.fit()

print(model.coef())
## (array([[0.27395004]], dtype=float32), array([[1.2682909]], dtype=float32))
```
