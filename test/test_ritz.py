import numpy
import pytest
import scipy

import krylov
from helpers import _matrices_herm, _matrices_nonherm, get_ip_Bs


@pytest.mark.parametrize("matrix", _matrices_herm + _matrices_nonherm)
@pytest.mark.parametrize(
    "get_operator", [lambda A: A, lambda A: krylov.utils.MatrixLinearOperator(A)]
)
@pytest.mark.parametrize("v", [numpy.ones((10, 1)), numpy.eye(10, 1)])
@pytest.mark.parametrize("maxiter", [1, 5, 9, 10])
@pytest.mark.parametrize("ip_B", get_ip_Bs())
@pytest.mark.parametrize("with_V", [True, False])
@pytest.mark.parametrize("type", ["ritz", "harmonic", "harmonic_improved"])
def test_ritz(matrix, get_operator, v, maxiter, ip_B, with_V, type):
    is_hermitian = any(matrix is x for x in _matrices_herm)
    eig = scipy.linalg.eigh if is_hermitian else scipy.linalg.eig
    Aevals, _ = eig(matrix)
    An = numpy.linalg.norm(matrix, 2)

    A = get_operator(matrix)

    ortho = "house" if ip_B is None else "dmgs"
    V, H = krylov.utils.arnoldi(A, v, maxiter=maxiter, ortho=ortho, ip_B=ip_B)
    N = v.shape[0]
    n = H.shape[1]
    A = krylov.utils.get_linearoperator((N, N), A)

    Z = None
    if with_V:
        theta, U, resnorm, Z = krylov.ritz(H, V=V, hermitian=is_hermitian, type=type)
    else:
        theta, U, resnorm = krylov.ritz(H, hermitian=is_hermitian, type=type)
    # check Z
    if Z is not None:
        assert numpy.linalg.norm(numpy.dot(V[:, :n], U) - Z, 2) <= 1e-14
    else:
        Z = numpy.dot(V[:, :n], U)

    # check shapes
    assert theta.shape == (n,)
    assert U.shape == (n, n)
    assert resnorm.shape == (n,)
    assert Z.shape == (N, n)
    # check norm of Ritz vectors
    for i in range(n):
        assert numpy.abs(numpy.linalg.norm(U[:, i], 2) - 1) <= 1e-14
    # check residuals
    R = A * Z - numpy.dot(Z, numpy.diag(theta))
    for i in range(n):
        rnorm = krylov.utils.norm(R[:, [i]], ip_B=ip_B)
        assert numpy.abs(rnorm - resnorm[i]) <= 1e-14 * An
    # check Ritz projection property
    if type == "ritz":
        assert (
            numpy.linalg.norm(krylov.utils.inner(V[:, :n], R, ip_B=ip_B), 2)
            <= 1e-14 * An
        )
    elif type == "harmonic":
        AVortho = scipy.linalg.orth(A * V[:, :n])
        assert (
            numpy.linalg.norm(krylov.utils.inner(AVortho, R, ip_B=ip_B), 2)
            <= 1e-12 * An
        )
    else:
        pass

    # compare Ritz values with eigenvalues if n==N
    if n == N:
        Aevals_sort = numpy.argsort(numpy.abs(Aevals))
        theta_sort = numpy.argsort(numpy.abs(theta))
        assert (numpy.abs(Aevals[Aevals_sort] - theta[theta_sort]) <= 5e-14 * An).all()
