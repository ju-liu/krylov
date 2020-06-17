from math import ceil

from numpy.testing import assert_almost_equal

import krylov


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
