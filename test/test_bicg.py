import numpy as np
import pytest
import scipyx as spx

import krylov

from .helpers import assert_correct
from .linear_problems import (
    complex_unsymmetric,
    hermitian_indefinite,
    hpd,
    real_unsymmetric,
    spd,
    spd_funny_rhs,
    symmetric_indefinite,
)


def test_compare_scipy(tol=1.0e-13):
    n = 5
    A, b = spd((n,))
    x0 = np.zeros(n)

    _, info_sp = spx.bicg(A, b, x0, maxiter=10, atol=1.0e-15)
    _, info_kry = krylov.bicg(A, b, maxiter=10, atol=1.0e-15)

    ref = np.asarray(info_sp.resnorms)
    assert np.all(np.abs(ref - info_kry.resnorms) < tol * (1.0 + ref))
    exit(1)


@pytest.mark.parametrize(
    "A_b",
    [
        spd((5,)),
        spd((5, 1)),
        spd((5, 3)),
        spd_funny_rhs(),
        hpd(),
        symmetric_indefinite(),
        hermitian_indefinite(),
        real_unsymmetric(),
        complex_unsymmetric(),
    ],
)
def test_bicg(A_b):
    A, b = A_b
    sol, info = krylov.bicg(A, b, tol=1.0e-7)
    assert_correct(A, b, info, sol, 1.0e-7)
