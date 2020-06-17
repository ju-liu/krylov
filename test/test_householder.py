import numpy
import pytest

import krylov

_factors = [0.0, 1.0, 1.0j, 1.0 + 1.0j, 1e8, 1e-8]


@pytest.mark.parametrize("a", _factors)
@pytest.mark.parametrize("b", _factors)
@pytest.mark.parametrize("length", [10, 1])
def test_house(a, b, length):
    x = numpy.ones((length, 1), dtype=numpy.array(a).dtype) * b
    x[0] = a

    H = krylov.Householder(x)
    y = H.apply(x)

    eye = numpy.eye(len(x))
    # check that H.matrix() equals to H.apply(I)
    HI = H.apply(eye)
    Hm = H.matrix()
    assert numpy.linalg.norm(HI - Hm, 2) <= 1e-14

    # check that H.matrix() is Hermitian
    assert numpy.linalg.norm(Hm - Hm.T.conj(), 2) <= 1e-14
    # check that H.matrix() is unitary/involutory
    assert numpy.linalg.norm(eye - numpy.dot(Hm.T.conj(), Hm), 2) <= 1e-14
    # check that absolute value of y[0] equals norm(x)
    assert numpy.abs(
        numpy.linalg.norm(x, 2) - numpy.abs(y[0])
    ) <= 1e-14 * numpy.linalg.norm(x, 2)
    # check that abs(alpha)=1
    assert numpy.abs(1 - numpy.abs(H.alpha)) <= 1e-14
    # check that y[0] = alpha*norm(x)
    assert numpy.abs(y[0] - H.alpha * H.xnorm) <= 1e-14 * numpy.linalg.norm(x, 2)
    if y.shape[0] > 1:
        # check that all elements of r except the first are zero
        assert numpy.linalg.norm(y[1:], 2) <= 1e-14 * numpy.linalg.norm(x, 2)
