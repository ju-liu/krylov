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
    symmetric_indefinite,
)


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
def test_bicg(A_b):
    A, b = A_b
    sol, info = krylov.bicg(A, b, tol=1.0e-7, maxiter=10)
    assert_consistent(A, b, info, sol, 1.0e-7)


@pytest.mark.parametrize(
    "A_b",
    [
        spd((5,)),
        spd((5, 1)),
        symmetric_indefinite(),
        hpd(),
        hermitian_indefinite(),
        real_unsymmetric(),
        complex_unsymmetric(),
    ],
)
def test_compare_scipy(A_b, tol=1.0e-12):
    A, b = A_b
    print(f"{A = }")
    print(f"{b = }")
    x0 = np.zeros_like(b)

    _, info_sp = spx.bicg(A, b, x0, maxiter=5, atol=1.0e-15)
    _, info_kry = krylov.bicg(A, b, maxiter=5, atol=1.0e-15)

    print()
    print(f"{info_sp.resnorms  = }")
    print(f"{info_kry.resnorms = }")
    ref = np.asarray(info_sp.resnorms)
    assert np.all(np.abs(ref - info_kry.resnorms) < tol * (1.0 + ref))
