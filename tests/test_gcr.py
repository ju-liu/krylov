import numpy as np
import pytest

import krylov

from .helpers import assert_consistent
from .linear_problems import (
    hermitian_indefinite,
    hpd,
    spd_dense,
    spd_rhs_0,
    spd_rhs_0sol0,
    spd_sparse,
    symmetric_indefinite,
)


def test_ref():
    A, b = spd_dense((5,))
    _, info = krylov.gcr(A, b, maxiter=5)

    ref = np.array(
        [
            2.23606797749979,
            1.06995241076096,
            0.9872554076721121,
            0.9709417754217194,
            0.7291872161218861,
            6.377745716588144e-16,
        ]
    )
    print(info.resnorms)
    assert np.all(np.abs(info.resnorms - ref) < 1.0e-13 * (1.0 + np.abs(ref)))


@pytest.mark.parametrize(
    "A_b",
    [
        spd_dense((5,)),
        spd_sparse((5,)),
        spd_sparse((5, 1)),
        spd_sparse((5, 3)),
        spd_rhs_0((5,)),
        spd_rhs_0sol0(),
        hpd(),
        # gcr also works for symmetric indefinite problems
        symmetric_indefinite(),
        hermitian_indefinite(),
    ],
)
def test_gcr(A_b):
    A, b = A_b
    print("A:")
    print(A)
    print()
    print("b:")
    print(b)
    print()
    A_dense = A if isinstance(A, np.ndarray) else A.toarray()
    sol = np.linalg.solve(A_dense, b)
    sol, info = krylov.gcr(A, b, tol=1.0e-7, maxiter=10, exact_solution=sol)
    print("info:")
    print(info)
    assert info.success
    assert_consistent(A, b, info, sol, 1.0e-7)
