import numpy
import scipy.linalg.blas as blas


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

    if numpy.isreal(X).all():
        # real vector
        X = numpy.real(X)
        fun = blas.drotg
    else:
        # complex vector
        fun = blas.zrotg

    G = []
    for k in range(X.shape[1]):
        c, s = fun(*X[:, k])
        G.append(numpy.array([[c, s], [-numpy.conj(s), c]]))
        # r = c * x[0] + s * x[1]

    G = numpy.array(G)
    G = numpy.moveaxis(G, 0, -1).reshape(2, 2, *X_shape[1:])
    return G
