import numpy
import pytest

import krylov

_factors = [0.0, 1.0, 1.0j, 1.0 + 1.0j, 1e8, 1e-8]


@pytest.mark.parametrize("a", _factors)
@pytest.mark.parametrize("b", _factors)
def test_givens(a, b):
    x = numpy.array([[a], [b]])

    G = krylov.Givens(x)
    y = G.apply(x)

    eye = numpy.eye(2)
    # check that G.G is unitary
    assert numpy.linalg.norm(eye - numpy.dot(G.G.T.conj(), G.G), 2) <= 1e-14
    # check that absolute value of y[0] equals norm(x)
    assert numpy.abs(
        numpy.linalg.norm(x, 2) - numpy.abs(y[0])
    ) <= 1e-14 * numpy.linalg.norm(x, 2)
    # check that y[0] == 0
    assert numpy.linalg.norm(y[1], 2) <= 1e-14 * numpy.linalg.norm(x, 2)
