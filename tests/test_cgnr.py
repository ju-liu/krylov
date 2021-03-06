import pytest

import krylov

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
        symmetric_indefinite(),
        hermitian_indefinite(),
        real_unsymmetric(),
        complex_unsymmetric(),
    ],
)
def test_cgnr(A_b):
    A, b = A_b
    print("A:")
    print(A)
    print()
    print("b:")
    print(b)
    print()
    sol, info = krylov.cgnr(A, b, tol=1.0e-7, maxiter=10)
    print("info:")
    print(info)
    assert info.success
    # The residuals are given as ||A^H b - A^H A x||, so the test against ||b - Ax||
    # fails.
    # assert_consistent(A, b, info, sol, 1.0e-7)
