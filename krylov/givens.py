import numpy
import scipy.linalg.blas as blas

from .errors import ArgumentError


def givens(x):
    """Compute Givens rotation for provided vector x.

    Computes Givens rotation
    :math:`G=\\begin{bmatrix}c&s\\\\-\\overline{s}&c\\end{bmatrix}`
    such that
    :math:`Gx=\\begin{bmatrix}r\\\\0\\end{bmatrix}`.
    """
    # make sure that x is a vector ;)
    if x.shape != (2,):
        raise ArgumentError("x is not a vector of shape (2,)")

    if numpy.isreal(x).all():
        # real vector
        c, s = blas.drotg(*numpy.real(x))
    else:
        # complex vector
        c, s = blas.zrotg(*x)

    G = numpy.array([[c, s], [-numpy.conj(s), c]])
    # r = c * x[0] + s * x[1]
    return G
