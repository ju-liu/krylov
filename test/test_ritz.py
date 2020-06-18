import numpy
import pytest
import scipy

import krylov
from helpers import _matrices_herm, _matrices_nonherm

B = numpy.diag(numpy.linspace(1, 5, 10))

_inners = [
    lambda x, y: numpy.dot(x.T.conj(), y),
    lambda x, y: numpy.dot(x.T.conj(), numpy.dot(B, y)),
]


@pytest.mark.parametrize("A", _matrices_herm + _matrices_nonherm)
@pytest.mark.parametrize("v", [numpy.ones((10, 1)), numpy.eye(10, 1)])
@pytest.mark.parametrize("maxiter", [1, 5, 9, 10])
@pytest.mark.parametrize("inner", _inners)
@pytest.mark.parametrize("with_V", [True, False])
@pytest.mark.parametrize("type", ["ritz", "harmonic", "harmonic_improved"])
def test_ritz(A, v, maxiter, inner, with_V, type):
    is_hermitian = any(A is x for x in _matrices_herm)
    eig = scipy.linalg.eigh if is_hermitian else scipy.linalg.eig
    Aevals, _ = eig(A)
    An = numpy.linalg.norm(A, 2)

    ortho = "house" if inner is None else "dmgs"
    V, H = krylov.arnoldi(A, v, maxiter=maxiter, ortho=ortho, inner=inner)
    N = v.shape[0]
    n = H.shape[1]

    Z = None
    if with_V:
        theta, U, resnorm, Z = krylov.ritz(H, V=V, hermitian=is_hermitian, type=type)
    else:
        theta, U, resnorm = krylov.ritz(H, hermitian=is_hermitian, type=type)

    V = V.reshape(V.shape[:2]).T  # TODO remove

    # check Z
    if Z is None:
        Z = numpy.dot(V[:, :n], U)

    assert numpy.linalg.norm(numpy.dot(V[:, :n], U) - Z, 2) <= 1e-14

    # check shapes
    assert theta.shape == (n,)
    assert U.shape == (n, n)
    assert resnorm.shape == (n,)
    assert Z.shape == (N, n)
    # check norm of Ritz vectors
    for i in range(n):
        assert numpy.abs(numpy.linalg.norm(U[:, i], 2) - 1) <= 1e-14
    # check residuals
    R = A @ Z - Z @ numpy.diag(theta)
    for i in range(n):
        rnorm = krylov.utils.norm(R[:, [i]], inner=inner)
        assert numpy.abs(rnorm - resnorm[i]) <= 1e-14 * An
    # check Ritz projection property
    if type == "ritz":
        assert numpy.linalg.norm(inner(V[:, :n], R), 2) <= 1e-14 * An
    elif type == "harmonic":
        AVortho = scipy.linalg.orth(A @ V[:, :n])
        assert numpy.linalg.norm(inner(AVortho, R), 2) <= 1e-12 * An
    else:
        pass

    # compare Ritz values with eigenvalues if n==N
    if n == N:
        Aevals_sort = numpy.argsort(numpy.abs(Aevals))
        theta_sort = numpy.argsort(numpy.abs(theta))
        assert (numpy.abs(Aevals[Aevals_sort] - theta[theta_sort]) <= 5e-14 * An).all()
