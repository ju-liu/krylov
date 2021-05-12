import numpy as np
import pytest
import scipyx as spx
from scipy.sparse.linalg import LinearOperator

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
def test_qmr(A_b):
    A, b = A_b
    print("A:")
    print(A)
    print()
    print("b:")
    print(b)
    print()
    A_dense = A if isinstance(A, np.ndarray) else A.toarray()
    sol = np.linalg.solve(A_dense, b)
    sol, info = krylov.qmr(A, b, tol=1.0e-7, maxiter=10, exact_solution=sol)
    print("info:")
    print(info)
    assert_consistent(A, b, info, sol, 1.0e-7)


@pytest.mark.parametrize(
    "A_b",
    [
        spd_sparse((5,)),
        spd_sparse((5, 1)),
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

    # scipy admits both precondioners or none, bug report:
    # <https://github.com/scipy/scipy/issues/14027>
    if with_prec:
        n = A.shape[0]
        M1 = LinearOperator(
            (n, n),
            matvec=lambda x: 2.0 * x,
            rmatvec=lambda x: 2.0 * x,
            dtype=float,
        )
        M2 = LinearOperator(
            (n, n),
            matvec=lambda x: 3.0 * x,
            rmatvec=lambda x: 3.0 * x,
            dtype=float,
        )
    else:
        M1 = None
        M2 = None

    x0 = np.zeros_like(b)

    print()
    print("M1:")
    print(M1)
    print()
    print("M2:")
    print(M2)
    print()
    print("use_explicit_residual", use_explicit_residual)

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
