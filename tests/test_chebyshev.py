import numpy as np
import pytest

import krylov

from .helpers import assert_consistent
from .linear_problems import spd_dense, spd_rhs_0, spd_rhs_0sol0, spd_sparse


def test_reference():
    A, b = spd_dense((5,))
    eigenvalue_estimates = (1.0e-2, 1.75)
    _, info = krylov.chebyshev(A, b, eigenvalue_estimates, tol=1.0e-5, maxiter=5)
    ref = np.array(
        [
            2.23606797749979,
            1.626826691029081,
            1.744954212067044,
            1.7113839589143471,
            1.6298632096913288,
            1.4593167230617032,
        ]
    )
    assert np.all(np.abs(info.resnorms - ref) < 1.0e-12 * (1.0 + ref))


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
    eigenvalue_estimates = (1.0e-2, 1.75)
    tol = 1.0e-3
    callback_counter = 0

    def callback(x, r):
        nonlocal callback_counter
        callback_counter += 1

    sol, info = krylov.chebyshev(
        A, b, eigenvalue_estimates, tol=tol, maxiter=100, callback=callback
    )
    print("info:")
    print(info)
    print()
    print(info.resnorms)
    assert callback_counter == info.numsteps + 1
    assert info.success
    assert_consistent(A, b, info, sol, tol)
