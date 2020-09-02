import numpy
import pytest

import krylov

_factors = [0.0, 1.0, 1.0j, 1.0 + 1.0j, 1e8, 1e-8]


@pytest.mark.parametrize("a", _factors)
@pytest.mark.parametrize("b", _factors)
@pytest.mark.parametrize("shape", [(10,), (10, 1), (1,)])
def test_house(a, b, shape):
    dtype = numpy.asarray(a * b).dtype
    x = numpy.full(shape, b, dtype=dtype)
    x[0] = a
    n = shape[0]

    H = krylov.Householder(x)
    y = H.apply(x)

    HI = []
    for k in range(n):
        e = numpy.zeros(shape)
        e[k] = 1.0
        HI.append(H.apply(e))
        assert HI[-1].shape == e.shape

    HI = numpy.array(HI)
    # "transpose" in the leading two dimensions
    HI = numpy.moveaxis(HI, 0, 1)
    Hm = H.matrix()

    # check that H.matrix() equals to H.apply(I)
    assert numpy.max(numpy.abs(HI - Hm)) <= 1e-14

    # check that H.matrix() is Hermitian
    HmH = numpy.moveaxis(Hm, 0, 1).conj()
    assert numpy.max(Hm - HmH) <= 1e-14
    # check that H.matrix() is unitary/involutory
    # create identity matrix
    eye = numpy.zeros([n, n] + list(x.shape[1:]))
    idx = numpy.arange(n)
    eye[idx, idx] = 1.0
    HmH_dot_Hm = numpy.einsum("ij...,jk...->ik...", HmH, Hm)
    assert numpy.max(numpy.abs(eye - HmH_dot_Hm)) <= 1e-14
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
