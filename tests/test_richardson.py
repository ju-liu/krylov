import numpy as np
import pytest

import krylov

from .helpers import assert_consistent
from .linear_problems import (
    complex_unsymmetric,
    hermitian_indefinite,
    hpd,
    real_unsymmetric,
    spd_dense,
    spd_rhs_0,
    spd_rhs_0sol0,
    spd_sparse,
    symmetric_indefinite,
)


def test_reference():
    A, b = spd_dense((5,))
    _, info = krylov.richardson(A, b, tol=1.0e-5, maxiter=5)
    ref = np.array(
        [
            2.23606797749979,
            1.3620205578477882,
            1.1590550073227759,
            1.0655176233413504,
            1.0133010995040146,
            0.9806493287065431,
        ]
    )
    print(info.resnorms)
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
        hpd(),
        symmetric_indefinite(),
        hermitian_indefinite(),
        real_unsymmetric(),
        complex_unsymmetric(),
    ],
)
def test_richardson(A_b):
    A, b = A_b
    print("A:")
    print(A)
    print()
    print("b:")
    print(b)
    print()
    tol = 1.0e-3
    sol, info = krylov.richardson(A, b, tol=tol, maxiter=10)
    print("info:")
    print(info)
    print()
    print(info.resnorms)
    assert_consistent(A, b, info, sol, tol)
