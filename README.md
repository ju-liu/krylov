<p align="center">
  <a href="https://github.com/nschloe/krylov"><img alt="krylov" src="https://nschloe.github.io/krylov/logo.svg" width="60%"></a>
  <p align="center">Krylov subspace methods.</p>
</p>

[![PyPi Version](https://img.shields.io/pypi/v/krylov.svg?style=flat-square)](https://pypi.org/project/krylov/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/krylov.svg?style=flat-square)](https://pypi.org/project/krylov/)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/krylov.svg?logo=github&label=Stars&logoColor=white&style=flat-square)](https://github.com/nschloe/krylov)
[![PyPi downloads](https://img.shields.io/pypi/dm/krylov.svg?style=flat-square)](https://pypistats.org/packages/krylov)

[![Discord](https://img.shields.io/static/v1?logo=discord&label=chat&message=on%20discord&color=7289da&style=flat-square)](https://discord.gg/hnTJ5MRX2Y)

[![gh-actions](https://img.shields.io/github/workflow/status/nschloe/krylov/ci?style=flat-square)](https://github.com/nschloe/krylov/actions?query=workflow%3Aci)
[![codecov](https://img.shields.io/codecov/c/github/nschloe/krylov.svg?style=flat-square)](https://app.codecov.io/gh/nschloe/krylov)
[![LGTM](https://img.shields.io/lgtm/grade/python/github/nschloe/krylov.svg?style=flat-square)](https://lgtm.com/projects/g/nschloe/krylov)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)

Forked from André Gaul's [KryPy](https://github.com/andrenarchy/krypy), _krylov_
provides Krylov subspace methods for Python. It is more advanced than [SciPy's linalg
methods](https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html) in that it
 
 * return many interesting statistics, e.g., the residual norms,
 * lets the user adjust more parameters, e.g., the orthogonalization method in GMRES,
 * allows arbitrary inner products, and
 * is fully blocked/vectorized.

_krylov_ is written completely in Python, so methods can be easily adapted.

Install with
```
pip install krylov
```
and use as
```python
import numpy
import krylov

A = numpy.diag([1.0e-3] + list(range(2, 101)))
b = numpy.ones(100)

# sol, info = krylov.cg(A, b)
# sol, info = krylov.minres(A, b)
sol, info = krylov.gmres(A, b)

# sol is None if no solution has been found
# info.resnorms contains the relative residual norms and some more data

# plot residuals
import matplotlib.pyplot as plt

plt.semilogy(info.resnorms)
plt.show()
```

### Advantages over SciPy

 * Works with SciPy sparse matrices, NumPy arrays, custom LinearOperators
 * Supports custom inner products
 * Returns residual norms and other interesting data
 * Full Python implementation
 * Fully vectorized, can deal with multiple right-hand sides
 * Supports floats and complex data


### Testing

To run the krylov unit tests, check out this repository and type, e.g.,
```
tox -- --maxfail=1
```

### Further reading

 * [R. Barrett et al., Templates for the Solution of Linear Systems: Building Blocks for
   Iterative Methods](https://www.netlib.org/templates/)

### License
This software is published under the [MIT license](https://en.wikipedia.org/wiki/MIT_License).
