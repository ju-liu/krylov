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
    #
    # From the docs:
    # DLARTG generate a plane rotation so that
    #
    #     [  CS  SN  ]  .  [ F ]  =  [ R ]   where CS**2 + SN**2 = 1.
    #     [ -SN  CS  ]     [ G ]     [ 0 ]
    #
    # This is a slower, more accurate version of the BLAS1 routine DROTG,
    # with the following other differences:
    #    F and G are unchanged on return.
    #    If G=0, then CS=1 and SN=0.
    #    If F=0 and (G .ne. 0), then CS=0 and SN=1 without doing any
    #       floating point operations (saves work in DBDSQR when
    #       there are zeros on the diagonal).
    #
    # If F exceeds G in magnitude, CS will be positive.
    lartg = lapack.get_lapack_funcs("lartg", (X,))

    G = []
    R = []
    for k in range(X.shape[1]):
        c, s, r = lartg(*X[:, k])
        G.append(np.array([[c, s], [-np.conj(s), c]]))
        R.append(r)

    G = np.array(G)
    G = np.moveaxis(G, 0, -1).reshape(2, 2, *X_shape[1:])
    R = np.array(R)
    return G, R
