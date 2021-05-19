import numpy as np
import pytest

import krylov

_factors = [0.0, 1.0, 1.0j, 1.0 + 1.0j, 1e8, 1e-8]


@pytest.mark.parametrize("a", _factors)
@pytest.mark.parametrize("b", _factors)
def test_givens(a, b):
    x = np.array([a, b])

    G, _ = krylov.givens(x)

    # check that G.G is unitary
    I = np.eye(2)
    assert np.linalg.norm(I - np.dot(G.T.conj(), G), 2) <= 1e-14

    # check that absolute value of y[0] equals norm(x)
    y = G @ x
    ref = np.linalg.norm(x, 2)
    assert np.abs(ref - np.abs(y[0])) <= 1e-14 * ref
    # check that y[1] == 0
    assert np.abs(y[1]) <= 1e-14 * ref
