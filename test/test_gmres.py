import numpy as np
import scipy.sparse

import krylov


def test_gmres_minres_equivalence():
    """GMRES and MINRES do the same thing when applied to symmetric matrices.  In SciPy,
    that's not the case, and it's unclear why. Keep an eye on
    <https://github.com/scipy/scipy/issues/13937>.
    """

    # construct a symmetric matrix
    n = 100
    data = -np.ones((3, n))
    data[1] = 2.0
    A = scipy.sparse.spdiags(data, [-1, 0, 1], n, n)

    x0 = np.zeros(n)
    b = np.ones(n)

    _, info_minres = krylov.minres(A, b, x0=x0, maxiter=3, tol=0.0)
    _, info_gmres = krylov.minres(A, b, x0=x0, maxiter=3, tol=0.0)

    res1 = np.asarray(info_minres.resnorms)
    res2 = np.asarray(info_gmres.resnorms)

    tol = 1.0e-13
    assert np.all(np.abs(res1 - res2) < tol * (1 + res2))
