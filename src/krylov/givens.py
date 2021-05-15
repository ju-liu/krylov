import numpy as np
from scipy.linalg import lapack


def givens(X):
    """Compute Givens rotation for provided vectors X.

    Computes Givens rotation
    :math:`G=\\begin{bmatrix}c&s\\\\-\\overline{s}&c\\end{bmatrix}` such that
    :math:`Gx=\\begin{bmatrix}r\\\\0\\end{bmatrix}`.
    """
    assert X.shape[0] == 2
    X_shape = X.shape
    X = X.reshape(X.shape[0], -1)

    # This was previously done with the BLAS routines *rotg.
    # A more fitting alternative are LAPACK's *lartg functions. See
    # <https://www.cs.cornell.edu/~bindel/papers/2002-toms.pdf>.
    lartg = lapack.get_lapack_funcs("lartg", (X,))

    G = []
    for k in range(X.shape[1]):
        c, s, _ = lartg(*X[:, k])
        G.append(np.array([[c, s], [-np.conj(s), c]]))

    G = np.array(G)
    G = np.moveaxis(G, 0, -1).reshape(2, 2, *X_shape[1:])
    return G
