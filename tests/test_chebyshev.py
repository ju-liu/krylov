import numpy as np
import pytest

import krylov

from .helpers import assert_consistent
from .linear_problems import spd_dense, spd_rhs_0, spd_rhs_0sol0, spd_sparse


@pytest.mark.parametrize(
    "A_b",
    [
        spd_dense((5,)),
        spd_sparse((5,)),
        spd_dense((5, 1)),
        spd_dense((5, 3)),
        spd_rhs_0((5,)),
        spd_rhs_0sol0(),
        # hpd(),
        # symmetric_indefinite(),
        # hermitian_indefinite(),
        # real_unsymmetric(),
        # complex_unsymmetric(),
    ],
)
def test_chebyshev(A_b):
    A, b = A_b
    print("A:")
    print(A)
    print()
    print("b:")
    print(b)
    print()
    A_dense = A if isinstance(A, np.ndarray) else A.toarray()
    sol = np.linalg.solve(A_dense, b)
    eigenvalue_estimates = (1.0e-2, 1.75)
    tol = 1.0e-3
    sol, info = krylov.chebyshev(
        A, b, eigenvalue_estimates, tol=tol, maxiter=100, exact_solution=sol
    )
    print("info:")
    print(info)
    print()
    print(info.resnorms)
    assert_consistent(A, b, info, sol, tol)
