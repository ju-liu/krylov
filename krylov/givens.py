import numpy
import scipy.linalg.blas as blas


def givens(x):
    """Compute Givens rotation for provided vector x.

    Computes Givens rotation
    :math:`G=\\begin{bmatrix}c&s\\\\-\\overline{s}&c\\end{bmatrix}`
    such that
    :math:`Gx=\\begin{bmatrix}r\\\\0\\end{bmatrix}`.
    """
    if numpy.isreal(x).all():
        # real vector
        c, s = blas.drotg(*numpy.real(x))
    else:
        # complex vector
        c, s = blas.zrotg(*x)

    G = numpy.array([[c, s], [-numpy.conj(s), c]])
    # r = c * x[0] + s * x[1]
    return G
