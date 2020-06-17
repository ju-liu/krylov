import numpy
import pytest

import krylov

_factors = [0.0, 1.0, 1.0j, 1.0 + 1.0j, 1e8, 1e-8]


@pytest.mark.parametrize("a", _factors)
@pytest.mark.parametrize("b", _factors)
def test_givens(a, b):
    x = numpy.array([a, b])

    G = krylov.givens(x)

    # check that G.G is unitary
    I = numpy.eye(2)
    assert numpy.linalg.norm(I - numpy.dot(G.T.conj(), G), 2) <= 1e-14

    # check that absolute value of y[0] equals norm(x)
    y = G @ x
    ref = numpy.linalg.norm(x, 2)
    assert numpy.abs(ref - numpy.abs(y[0])) <= 1e-14 * ref
    # check that y[1] == 0
    assert numpy.abs(y[1]) <= 1e-14 * ref
