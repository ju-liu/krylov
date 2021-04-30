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
        # no idea why this works:
        symmetric_indefinite(),
        hermitian_indefinite(),
    ],
)
def test_cg(A_b):
    A, b = A_b
    sol, info = krylov.cg(A, b, tol=1.0e-7)
    assert_correct(A, b, info, sol, 1.0e-7)

    assert info.numsteps + 1 == len(info.resnorms)


def test_BoundCG():
    b = krylov.BoundCG([1, 2])
    assert_almost_equal(b.eval_step(8), 1.5018239652065932e-06)
    assert ceil(b.get_step(1e-6)) == 9

    b = krylov.BoundCG(
        krylov.utils.Intervals(
            [krylov.utils.Interval(1, 1.2), krylov.utils.Interval(2)]
        )
    )
    assert_almost_equal(b.eval_step(8), 1.5018239652065932e-06)
    assert ceil(b.get_step(1e-6)) == 9
