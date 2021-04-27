import pytest

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
