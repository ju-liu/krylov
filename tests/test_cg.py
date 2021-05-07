import pytest

import krylov

from .helpers import assert_consistent
from .linear_problems import (
    hermitian_indefinite,
    hpd,
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
        # no idea why this works:
        symmetric_indefinite(),
        hermitian_indefinite(),
    ],
)
def test_cg(A_b):
    A, b = A_b
    sol, info = krylov.cg(A, b, tol=1.0e-7)
    assert_consistent(A, b, info, sol, 1.0e-7)
