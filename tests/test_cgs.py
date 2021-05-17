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
def test_cgs(A_b):
    A, b = A_b
    print("A:")
    print(A)
    print()
    print("b:")
    print(b)
    print()
    callback_counter = 0

    def callback(x, r):
        nonlocal callback_counter
        callback_counter += 1

    sol, info = krylov.cgs(A, b, tol=1.0e-7, maxiter=10, callback=callback)
    print("info:")
    print(info)
    assert callback_counter == info.numsteps + 1
    assert info.success
    assert_consistent(A, b, info, sol, 1.0e-7)


@pytest.mark.parametrize(
    "A_b",
    [
        spd_dense((5,)),
        spd_dense((5, 1)),
        symmetric_indefinite(),
        real_unsymmetric(),
        hpd(),
        # For some reason, not _exactly_ like scipy:
        # hermitian_indefinite(),
        # complex_unsymmetric(),
    ],
)
@pytest.mark.parametrize("with_prec", [False, True])
def test_compare_scipy(A_b, with_prec, tol=1.0e-10):
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

    _, info_sp = spx.cgs(A, b, x0, M=M, maxiter=5, atol=1.0e-15)
    _, info_kry = krylov.cgs(A, b, M=M, maxiter=5, atol=1.0e-15)

    print()
    print("scipy.info ", info_sp.resnorms)
    print()
    print("krylov.info", info_kry.resnorms)
    ref = np.asarray(info_sp.resnorms)
    assert np.all(np.abs(ref - info_kry.resnorms) < tol * (1.0 + ref))
