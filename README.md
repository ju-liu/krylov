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

Some important figures in Krylov subspace method, ordered by date of birth.

#### [Aleksey N. Krylov](https://en.wikipedia.org/wiki/Aleksey_Krylov) (1863–1945)
<img src="https://nschloe.github.io/krylov/krylov.jpg" align="left" width="200px"/>

Алексе́й Никола́евич Крыло́в, Russian naval engineer, applied mathematician and memoirist.
Krylov wrote about 300 papers and books. They span a wide range of topics, including
shipbuilding, magnetism, artillery, mathematics, astronomy, and geodesy. In 1904 he
built the first machine in Russia for integrating Ordinary differential equations. In
1931 (aged 68) he published a paper on what is now called the [Krylov
subspace](https://en.wikipedia.org/wiki/Krylov_subspace) and Krylov subspace methods.
<br clear="left"/>


#### [Cornelius Lanczos](https://en.wikipedia.org/wiki/Cornelius_Lanczos) (1893–1974)
<img src="https://nschloe.github.io/krylov/lanczos.jpg" align="left" width="200px"/>

Extensive contributions all over numerical analysis and physics. For example, Lanczos
pioneered [FFT](https://en.wikipedia.org/wiki/Fast_Fourier_transform), introduced
[Chebyshev polynomials](https://en.wikipedia.org/wiki/Chebyshev_polynomials) to
numerics, and devised the [Lanczos
algorithm](https://en.wikipedia.org/wiki/Lanczos_algorithm) for eigenvalue computations.

Assistant to Einstein in Berlin. Later one of the "Martians", a group of Hungarian
scientists who emigrated to the US in the first half the 20th century.

In 1972, aged 79, Lanczos gave two extensive interviews which are recorded on video and
can be watched on YouTube:

* [Lanczos about his life](https://youtu.be/PO6xtSxB5Vg)
* [Lanczos about mathematics](https://youtu.be/avSHHi9QCjA)

<br clear="left"/>


#### [Alston Scott Householder](https://en.wikipedia.org/wiki/Alston_Scott_Householder) (1904–1993)

<img src="https://nschloe.github.io/krylov/householder.jpg" align="left" width="200px"/>

Discoveror of the [Householder
transformation](https://en.wikipedia.org/wiki/Householder_transformation) and of
[Householder's method](https://en.wikipedia.org/wiki/Householder%27s_method).
[SIAM](https://www.siam.org/) and [ACM](https://www.acm.org/) president, organizer of
the Gatlinburg conferences, now named Householder Symposia.

<br clear="left"/>


#### [Magnus Hestenes](https://en.wikipedia.org/wiki/Magnus_Hestenes) (1906–1991)
<img src="https://nschloe.github.io/krylov/hestenes.jpg" align="left" width="200px"/>

Best known for his contributions to calculus of variations and optimal control, pioneer
in in computer science.  Co-discoverer (with Lanczos and Stiefel) of the conjugate
gradient method (CG).

[AMS](https://www.ams.org/home/page) vice president.
<br clear="left"/>


#### [Eduard Stiefel](https://en.wikipedia.org/wiki/Eduard_Stiefel) (1909–1978)
<img src="https://nschloe.github.io/krylov/stiefel.jpg" align="left" width="200px"/>

Ground-breaking contributions in the fields of algebraic topology. Co-discoverer (with
Lanczos and Hestenes) of CG.

Also active as a military officer, rising to the rank of colonel in the Swiss army
during World War II.
<br clear="left"/>


#### [J. Wallace Givens](https://en.wikipedia.org/wiki/Wallace_Givens) (1910–1993)
<img src="https://nschloe.github.io/krylov/givens.jpg" align="left" width="200px"/>

Pioneer in computer science. Director of the Division of Applied Mathematics at the
Argonne National Labs. SIAM president. [Givens
rotations](https://en.wikipedia.org/wiki/Givens_rotation) are named after him.
<br clear="left"/>


#### [Walter Edwin Arnoldi](https://en.wikipedia.org/wiki/Walter_Edwin_Arnoldi) (1917–1995)

American engineer, mainly known for the
[discovery](https://www.jstor.org/stable/43633863) of the [Arnoldi
iteration](https://en.wikipedia.org/wiki/Arnoldi_iteration) which for Hermitian
matrices, reduces to the [Lanczos
algorithm](https://en.wikipedia.org/wiki/Lanczos_algorithm).

<br clear="left"/>

#### [Henk van der Vorst](https://en.wikipedia.org/wiki/Henk_van_der_Vorst) (1944–)
<img src="https://nschloe.github.io/krylov/vandervorst.jpg" align="left" width="200px"/>

Various important contributions in Krylov subspace methods and related topics.
Discoverer of the BiCGSTAB method and the Jacobi-Davidson method.

Since 2006, van der Vorst [works as an artist](http://www.henkvandervorst.nl/).

YouTube: [Galerie Laimböck -- Henk van der Vorst ‘Geometrische Origami’
(Dutch)](https://youtu.be/q6cReoxEvDk)
<br clear="left"/>


#### [Yousef Saad](https://en.wikipedia.org/wiki/Yousef_Saad) (1950–)
<img src="https://nschloe.github.io/krylov/saad.png" align="left" width="200px"/>

Various important contributions in Krylov subspace methods and related topics.
Discoverer of GCR and [GMRES](https://doi.org/10.1137/0907058) (with Martin H. Schultz).

YouTube: [Filtering techniques for eigenvalue problems, Saad](https://youtu.be/ck660Jd7wrI)
<br clear="left"/>


#### [Christopher Paige](https://www.cs.mcgill.ca/~chris/)
<img src="https://nschloe.github.io/krylov/paige.jpg" align="left" width="200px"/>

Discoverer of [MINRES, SYMMLQ](https://www.jstor.org/stable/2156178) (with Saunders).

<br clear="left"/>


#### [Michael Saunders](https://web.stanford.edu/~saunders/)
 <img src="https://nschloe.github.io/krylov/saunders.jpg" align="left" width="200px"/>

Discoverer of [MINRES, SYMMLQ](https://www.jstor.org/stable/2156178) (with Paige).
<br clear="left"/>


### Books

* [Yousef Saad, Iterative Krylov Methods for Large Linear Systems](https://doi.org/10.1017/CBO9780511615115)
* [Henk A. van der Vorst, Iterative Methods for Sparse Linear Systems](https://doi.org/10.1137/1.9780898718003)
* [Jörg Liesen and Zdeněk Strakoš, Krylov Subspace Methods](https://global.oup.com/academic/product/krylov-subspace-methods-9780199655410)

* [R. Barrett et al., Templates for the Solution of Linear Systems: Building Blocks for
  Iterative Methods](https://www.netlib.org/templates/)

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

### License
This software is published under the [MIT license](https://en.wikipedia.org/wiki/MIT_License).
