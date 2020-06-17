import numpy
import scipy.linalg.blas as blas

from .errors import ArgumentError


class Givens:
    def __init__(self, x):
        """Compute Givens rotation for provided vector x.

        Computes Givens rotation
        :math:`G=\\begin{bmatrix}c&s\\\\-\\overline{s}&c\\end{bmatrix}`
        such that
        :math:`Gx=\\begin{bmatrix}r\\\\0\\end{bmatrix}`.
        """
        # make sure that x is a vector ;)
        if x.shape != (2, 1):
            raise ArgumentError("x is not a vector of shape (2,1)")

        a = x[0].item()
        b = x[1].item()
        # real vector
        if numpy.isreal(x).all():
            a = numpy.real(a)
            b = numpy.real(b)
            c, s = blas.drotg(a, b)
        # complex vector
        else:
            c, s = blas.zrotg(a, b)

        self.c = c
        self.s = s
        self.r = c * a + s * b
        self.G = numpy.array([[c, s], [-numpy.conj(s), c]])

    def apply(self, x):
        """Apply Givens rotation to vector x."""
        return numpy.dot(self.G, x)
