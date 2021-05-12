import numpy as np
import pytest
import scipy.sparse

import krylov

from .helpers import assert_consistent
from .linear_problems import (
    complex_unsymmetric,
    hermitian_indefinite,
    hpd,
    real_unsymmetric,
)
from .linear_problems import spd_dense as spd
from .linear_problems import spd_rhs_0, spd_rhs_0sol0, symmetric_indefinite


@pytest.mark.parametrize(
    "A_b",
    [
        spd((5,)),
        spd((5, 1)),
        spd((5, 3)),
        spd_rhs_0((5,)),
        spd_rhs_0sol0(),
        hpd(),
        symmetric_indefinite(),
        hermitian_indefinite(),
        real_unsymmetric(),
        complex_unsymmetric(),
    ],
)
@pytest.mark.parametrize(
    "ortho",
    ["mgs", "dmgs"],
)
def test_gmres(A_b, ortho):
    A, b = A_b
    A_dense = A if isinstance(A, np.ndarray) else A.toarray()
    sol = np.linalg.solve(A_dense, b)

    sol, info = krylov.gmres(A, b, tol=1.0e-7, ortho=ortho, exact_solution=sol)
    assert_consistent(A, b, info, sol, 1.0e-7)


# TODO lanczos doesn't work for unsymmetric
@pytest.mark.parametrize(
    "A_b",
    [
        spd((5,)),
        spd((5, 1)),
        spd((5, 3)),
        spd_rhs_0sol0(),
        hpd(),
        symmetric_indefinite(),
        hermitian_indefinite(),
    ],
)
def test_gmres_lanczos(A_b, ortho="lanczos"):
    A, b = A_b
    A_dense = A if isinstance(A, np.ndarray) else A.toarray()
    sol = np.linalg.solve(A_dense, b)

    sol, info = krylov.gmres(A, b, tol=1.0e-7, ortho=ortho, exact_solution=sol)
    assert_consistent(A, b, info, sol, 1.0e-7)


def test_gmres_minres_equivalence():
    """GMRES and MINRES do the same thing when applied to symmetric matrices. In SciPy,
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
