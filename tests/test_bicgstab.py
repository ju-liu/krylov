import numpy as np
import pytest
import scipyx as spx

import krylov

from .helpers import assert_consistent
from .linear_problems import (
    complex_unsymmetric,
    hermitian_indefinite,
    hpd,
    real_unsymmetric,
    spd,
    spd_rhs_0,
    spd_rhs_0sol0,
    spd_sparse,
    symmetric_indefinite,
)


@pytest.mark.parametrize(
    "A_b",
    [
        spd((5,)),
        spd_sparse((5,)),
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
def test_bicgstab(A_b):
    A, b = A_b
    print("A:")
    print(A)
    print()
    print("b:")
    print(b)
    print()
    sol, info = krylov.bicgstab(A, b, tol=1.0e-7, maxiter=10)
    print("info:")
    print(info)
    assert_consistent(A, b, info, sol, 1.0e-7)


@pytest.mark.parametrize(
    "A_b",
    [
        spd((5,)),
        spd((5, 1)),
        symmetric_indefinite(),
        real_unsymmetric(),
        hpd(),
        hermitian_indefinite(),
        complex_unsymmetric(),
    ],
)
@pytest.mark.parametrize("with_prec", [False, True])
@pytest.mark.parametrize("use_explicit_residual", [False, True])
def test_compare_scipy(A_b, with_prec, use_explicit_residual, tol=1.0e-12):
    A, b = A_b
    print()
    print("A:")
    print(A)
    print()
    print("b:")
    print(b)

    if with_prec:
        n = A.shape[0]
        M = np.diag(np.full(n, 2.0))
    else:
        M = None

    x0 = np.zeros_like(b)

    print()
    print("M:")
    print(M)

    _, info_sp = spx.bicgstab(A, b, x0, M=M, maxiter=5, atol=1.0e-15)
    _, info_kry = krylov.bicgstab(
        A, b, Ml=M, maxiter=5, atol=1.0e-15, use_explicit_residual=use_explicit_residual
    )

    print()
    print("scipy.info ", info_sp.resnorms)
    print()
    print("krylov.info", info_kry.resnorms)
    ref = np.asarray(info_sp.resnorms)
    assert np.all(np.abs(ref - info_kry.resnorms) < tol * (1.0 + ref))
