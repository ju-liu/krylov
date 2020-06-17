from math import ceil

from numpy.testing import assert_almost_equal

import krylov


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
