import numpy
from scipy.linalg import lapack


def givens(X):
    """Compute Givens rotation for provided vectors X.

    Computes Givens rotation
    :math:`G=\\begin{bmatrix}c&s\\\\-\\overline{s}&c\\end{bmatrix}`
    such that
    :math:`Gx=\\begin{bmatrix}r\\\\0\\end{bmatrix}`.
    """
    assert X.shape[0] == 2
    X_shape = X.shape
    X = X.reshape(X.shape[0], -1)

    # This was previously done with the BLAS routines *rotg.
    # A more fitting alternative are LAPACK's *lartg functions. See
    # <https://www.cs.cornell.edu/~bindel/papers/2002-toms.pdf>.
    if numpy.isreal(X).all():
        # real vector
        X = numpy.real(X)
        fun2 = lapack.dlartg
    else:
        # complex vector
        fun2 = lapack.zlartg

    G = []
    for k in range(X.shape[1]):
        c, s, _ = fun2(*X[:, k])
        G.append(numpy.array([[c, s], [-numpy.conj(s), c]]))

    G = numpy.array(G)
    G = numpy.moveaxis(G, 0, -1).reshape(2, 2, *X_shape[1:])
    return G
