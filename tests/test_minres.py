import pytest

import krylov

from .helpers import assert_consistent
from .linear_problems import hermitian_indefinite, hpd
from .linear_problems import spd_dense as spd
from .linear_problems import spd_rhs_0, spd_rhs_0sol0, symmetric_indefinite


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
    ],
)
@pytest.mark.parametrize(
    "ortho",
    ["mgs", "dmgs", "lanczos"],
)
def test_minres(A_b, ortho):
    A, b = A_b

    callback_counter = 0

    def callback(x, r):
        nonlocal callback_counter
        callback_counter += 1

    sol, info = krylov.minres(A, b, tol=1.0e-7, ortho=ortho, callback=callback)
    assert callback_counter == info.numsteps + 1
    assert info.success
    assert_consistent(A, b, info, sol, 1.0e-7)
