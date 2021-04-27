from math import ceil

import pytest
from numpy.testing import assert_almost_equal

import krylov

from .helpers import assert_correct
from .linear_problems import (
    hermitian_indefinite,
    hpd,
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
    ],
)
@pytest.mark.parametrize(
    "ortho",
    ["mgs", "dmgs", "lanczos"],
)
def test_minres(A_b, ortho):
    A, b = A_b
    sol, info = krylov.minres(A, b, tol=1.0e-7, ortho=ortho)
    assert_correct(A, b, info, sol, 1.0e-7)


def test_BoundMinres():
    b = krylov.BoundMinres([-1, 1, 2])
    assert_almost_equal(b.eval_step(8), 0.0017331035544401801)
    assert ceil(b.get_step(2e-3)) == 8

    b = krylov.BoundMinres(
        krylov.utils.Intervals(
            [krylov.utils.Interval(-2, -1), krylov.utils.Interval(2)]
        )
    )
    assert_almost_equal(b.eval_step(8), 0.0017331035544401801)
    assert ceil(b.get_step(2e-3)) == 8
