import numpy
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal

import krylov
from helpers import get_ip_Bs


@pytest.mark.parametrize(
    "X",
    [
        numpy.eye(10, 1),
        numpy.eye(10, 5),
        numpy.eye(10, 5) + 1e-1 * numpy.ones((10, 5)),
        numpy.eye(10),
        numpy.zeros((10, 0)),
    ],
)
@pytest.mark.parametrize("Ys", [None, 0, 1])
@pytest.mark.parametrize("ip_B", get_ip_Bs())
@pytest.mark.parametrize("iterations", [1, 2, 3])
def test_projection(X, Ys, ip_B, iterations):
    Y = None if Ys is None else X + Ys

    P = krylov.Projection(X, Y, ip_B=ip_B, iterations=iterations)

    (N, k) = X.shape
    I = numpy.eye(N)
    z = numpy.ones((10, 1))
    z /= numpy.linalg.norm(z, 2)

    # check that it's a projection, i.e. P^2=P
    assert_almost_equal(numpy.linalg.norm(P.apply(I - P.apply(I)), 2), 0, 14)
    if k > 0:
        # check that the range is X, i.e. the kernel of I-P is X
        assert_almost_equal(numpy.linalg.norm(X - P.apply(X), 2), 0, 14)

        # check that the kernel is Y^perp, i.e. the range of I-P is orthogonal
        # to Y
        assert_almost_equal(
            numpy.linalg.norm(
                krylov.utils.inner(X if Y is None else Y, I - P.apply(I), ip_B=ip_B), 2
            ),
            0,
            13,
        )
    else:
        assert_equal(numpy.linalg.norm(P.apply(I)), 0)

    # check that the complementary projection is correct
    assert_almost_equal(
        numpy.linalg.norm(I - P.apply(I) - P.apply_complement(I), 2), 0, 14
    )

    # check that operator()*z executes apply(z)
    assert numpy.linalg.norm(P.operator() * z - P.apply(z)) == 0

    # check that operator_complement()*z executes apply_complement(z)
    assert numpy.linalg.norm(P.operator_complement() * z - P.apply_complement(z)) == 0

    # check that the matrix representation is correct
    assert_almost_equal(numpy.linalg.norm(P.matrix() - P.apply(I), 2), 0, 14)

    # check that <Y,a> is returned correctly with return_Ya=True
    a = numpy.ones((N, 1))
    _, Ya = P.apply(a, return_Ya=True)
    assert_array_almost_equal(
        Ya, krylov.utils.inner(X if Y is None else Y, a, ip_B=ip_B)
    )

    # same check for apply_complement
    _, Ya = P.apply_complement(a, return_Ya=True)
    assert_array_almost_equal(
        Ya, krylov.utils.inner(X if Y is None else Y, a, ip_B=ip_B)
    )
