<p align="center">
  <a href="https://github.com/nschloe/krylov"><img alt="krylov" src="https://nschloe.github.io/krylov/logo.svg" width="60%"></a>
  <p align="center">Iterative methods for linear equation systems.</p>
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

### Team Krylov

#### [Aleksey N. Krylov](https://en.wikipedia.org/wiki/Aleksey_Krylov)

<img src="https://nschloe.github.io/krylov/krylov.jpg" align="left" width="200px"/>
Алексе́й Никола́евич Крыло́в (1863-1945).

#### [J. Wallace Givens](https://en.wikipedia.org/wiki/Wallace_Givens)

<img src="https://nschloe.github.io/krylov/givens.jpg" align="left" width="200px"/>
(1910-1993)

[Givens rotation](https://en.wikipedia.org/wiki/Givens_rotation)

#### [Walter Edwin Arnoldi](https://en.wikipedia.org/wiki/Walter_Edwin_Arnoldi)
(1917-1995)  [Arnoldi iteration](https://en.wikipedia.org/wiki/Arnoldi_iteration)

#### [Alston Scott Householder](https://en.wikipedia.org/wiki/Alston_Scott_Householder)

<img src="https://nschloe.github.io/krylov/householder.jpg" align="left" width="200px"/>
(1904-1993)

Householder Symposium

  Inventor of the [Householder
  transformation](https://en.wikipedia.org/wiki/Householder_transformation) and of
  [Householder's method](https://en.wikipedia.org/wiki/Householder%27s_method).


#### [Yussef Saad](https://en.wikipedia.org/wiki/Yousef_Saad) (1950-)
  GCR, GMRES

#### [Magnus Hestenes](https://en.wikipedia.org/wiki/Magnus_Hestenes) (1906-1991)
 <img src="https://nschloe.github.io/krylov/hestenes.jpg" align="left" width="200px"/>
  CG

#### [Eduard Stiefel](https://en.wikipedia.org/wiki/Eduard_Stiefel) (1909-1978)
 <img src="https://nschloe.github.io/krylov/stiefel.jpg" align="left" width="200px"/>
  CG

#### [Cornelius Lanczos](https://en.wikipedia.org/wiki/Cornelius_Lanczos) (1893-1974)
 <img src="https://nschloe.github.io/krylov/lanczos.jpg" align="left" width="200px"/>
  Foundations of FFT, Chebyshev polynomials, [Lanczos
  algorithm](https://en.wikipedia.org/wiki/Lanczos_algorithm) for eigenvalues, CG
  [Lanczos about his life](https://youtu.be/PO6xtSxB5Vg)
  [Lanczos about mathematics](https://youtu.be/avSHHi9QCjA)


#### [Christopher Paige](https://www.cs.mcgill.ca/~chris/)
 <img src="https://nschloe.github.io/krylov/paige.jpg" align="left" width="200px"/>
  MINRES

#### [Michael Saunders](https://web.stanford.edu/~saunders/)
 <img src="https://nschloe.github.io/krylov/saunders.jpg" align="left" width="200px"/>
  MINRES

#### [Henk van der Vorst](https://en.wikipedia.org/wiki/Henk_van_der_Vorst) (1994-)
 <img src="https://nschloe.github.io/krylov/vandervorst.jpg" align="left" width="200px"/>
  BiCGSTAB, GMRESR, Jacobi-Davidson method.

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
