import numpy as np
import pytest
import scipyx as spx
from scipy.sparse import spdiags

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
def test_qmr(A_b):
    A, b = A_b
    print("A:")
    print(A)
    print()
    print("b:")
    print(b)
    print()
    sol, info = krylov.qmr(A, b, tol=1.0e-7, maxiter=10)
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
        # For some reason, not _exactly_ like scipy:
        # hermitian_indefinite(),
        # complex_unsymmetric(),
    ],
)
@pytest.mark.parametrize("with_M1", [False, True])
@pytest.mark.parametrize("with_M2", [False, True])
@pytest.mark.parametrize("use_explicit_residual", [False, True])
def test_compare_scipy(A_b, with_M1, with_M2, use_explicit_residual, tol=1.0e-10):
    A, b = A_b
    print()
    print("A:")
    print(A)
    print()
    print("b:")
    print(b)

    if with_M1:
        n = A.shape[0]
        M1 = spdiags(np.full(n, 2.0), [0], n, n)
    else:
        M1 = None

    if with_M2:
        n = A.shape[0]
        M2 = spdiags(np.full(n, 2.0), [0], n, n)
    else:
        M2 = None

    x0 = np.zeros_like(b)

    print()
    print("M1:")
    print(M1)
    print()
    print("M2:")
    print(M2)

    _, info_sp = spx.qmr(A, b, x0, M1=M1, M2=M2, maxiter=5, atol=1.0e-15)

    _, info_kry = krylov.qmr(
        A,
        b,
        Ml=M1,
        Mr=M2,
        maxiter=5,
        atol=1.0e-15,
        use_explicit_residual=use_explicit_residual,
    )

    print()
    print("scipy.info ", info_sp.resnorms)
    print()
    print("krylov.info", info_kry.resnorms)
    ref = np.asarray(info_sp.resnorms)
    assert np.all(np.abs(ref - info_kry.resnorms) < tol * (1.0 + ref))
