import numpy
import pytest
import scipy

import krylov
from helpers import get_matrix_comp_nonsymm  # get_matrix_herm_indef,
from helpers import (
    get_matrix_hpd,
    get_matrix_nonsymm,
    get_matrix_spd,
    get_matrix_symm_indef,
)

_B = numpy.diag(numpy.linspace(1.0, 5.0, 10))


@pytest.mark.parametrize(
    "A",
    [
        get_matrix_spd(),
        get_matrix_hpd(),
        get_matrix_symm_indef(),
        # TODO activate
        # get_matrix_herm_indef(),
        get_matrix_nonsymm(),
        get_matrix_comp_nonsymm(),
    ],
)
@pytest.mark.parametrize(
    "get_operator", [lambda A: A, lambda A: krylov.MatrixLinearOperator(A)]
)
@pytest.mark.parametrize("v", [numpy.ones((10, 1)), numpy.eye(10, 1)])
@pytest.mark.parametrize("maxiter", [1, 5, 9, 10])
@pytest.mark.parametrize("ortho", ["mgs", "dmgs", "house"])
@pytest.mark.parametrize("M", [None, _B])
@pytest.mark.parametrize(
    "ip_B", [lambda x, y: x.T.conj().dot(y), lambda x, y: x.T.conj().dot(_B.dot(y))],
)
def test_arnoldi(A, get_operator, v, maxiter, ortho, M, ip_B):
    An = numpy.linalg.norm(A, 2)

    if ortho == "house" and (ip_B is not None or M is not None):
        return

    if ip_B is None:
        res = krylov.arnoldi(A, v, maxiter=maxiter, ortho=ortho, M=M)
    else:
        res = krylov.arnoldi(A, v, maxiter=maxiter, ortho=ortho, M=M, ip_B=ip_B)

    if M is not None:
        V, H, P = res
    else:
        V, H = res
        P = None
    assert_arnoldi(A, v, V, H, P, maxiter, ortho, M, ip_B, An=An)


@pytest.mark.parametrize(
    "A",
    [
        # TODO: reactivate the complex tests once travis-ci uses newer
        #       numpy/scipy versions.
        get_matrix_spd(),
        # get_matrix_hpd(),
        get_matrix_symm_indef(),
        # get_matrix_herm_indef()
    ],
)
@pytest.mark.parametrize(
    "get_operator", [lambda A: A, lambda A: krylov.MatrixLinearOperator(A)]
)
@pytest.mark.parametrize("v", [numpy.ones((10, 1)), numpy.eye(10, 1)])
@pytest.mark.parametrize("maxiter", [1, 5, 9, 10])
@pytest.mark.parametrize("M", [None, _B])
@pytest.mark.parametrize(
    "ip_B", [lambda x, y: x.T.conj().dot(y), lambda x, y: x.T.conj().dot(_B.dot(y))],
)
def test_arnoldi_lanczos(A, get_operator, v, maxiter, M, ip_B):
    An = numpy.linalg.norm(A, 2)
    ortho = "lanczos"

    res = krylov.arnoldi(A, v, maxiter=maxiter, ortho=ortho, M=M, ip_B=ip_B)
    if M is not None:
        V, H, P = res
    else:
        V, H = res
        P = None
    assert_arnoldi(A, v, V, H, P, maxiter, ortho, M, ip_B, An=An)


def assert_arnoldi(
    A,
    v,
    V,
    H,
    P,
    maxiter,
    ortho,
    M,
    ip_B,
    lanczos=False,
    arnoldi_const=1,
    ortho_const=1,
    proj_const=10,
    An=None,
):
    # the checks in this function are based on the following literature:
    # [1] Drkosova, Greenbaum, Rozloznik and Strakos. Numerical Stability of
    #     GMRES. 1995. BIT.
    N = v.shape[0]
    if An is None:
        An = numpy.linalg.norm(A, 2)
    eps = numpy.finfo(numpy.double).eps

    k = H.shape[1]

    # maxiter respected?
    assert k <= maxiter

    invariant = H.shape[0] == k
    # check shapes of V and H
    assert V.shape[0] == H.shape[0]

    if P is None:
        P = V

    # check that the initial vector is correct
    Mv = v if M is None else M @ v
    print(ip_B)
    v1n = numpy.sqrt(ip_B(v, Mv))
    assert numpy.linalg.norm(P[0] - v / v1n) <= 1e-14

    # check if H is Hessenberg
    assert numpy.linalg.norm(numpy.tril(H, -2)) == 0
    if lanczos:
        # check if H is Hermitian
        assert numpy.linalg.norm(H - H.T.conj()) == 0
        # check if H is real
        assert numpy.isreal(H).all()

    # check if subdiagonal-elements are real and non-negative
    d = numpy.diag(H[1:, :])
    assert (numpy.abs(d.imag) < 1.0e-15).all()
    assert (d >= 0).all()

    V = V.reshape(V.shape[:2]).T  # TODO remove
    if P is not None:
        P = P.reshape(P.shape[:2]).T  # TODO remove

    # check Arnoldi residual \| A*V_k - V_{k+1} H \|
    AV = A @ V if invariant else A @ V[:, :-1]
    MAV = AV if M is None else M @ AV
    arnoldi_res = MAV - numpy.dot(V, H)
    arnoldi_resn = krylov.utils.norm(arnoldi_res, ip_B=ip_B)
    # inequality (2.3) in [1]
    arnoldi_tol = arnoldi_const * k * (N ** 1.5) * eps * An
    assert arnoldi_resn <= arnoldi_tol

    # check orthogonality by measuring \| I - <V,V> \|_2
    ortho_res = numpy.eye(V.shape[1]) - ip_B(V, P)

    ortho_resn = numpy.linalg.norm(ortho_res, 2)
    if ortho == "house":
        ortho_tol = ortho_const * (k ** 1.5) * N * eps  # inequality (2.4) in [1]
    else:
        vAV_singvals = scipy.linalg.svd(
            numpy.column_stack([V[:, [0]], (MAV[:, :-1] if invariant else MAV)]),
            compute_uv=False,
        )
        if vAV_singvals[-1] == 0:
            ortho_tol = numpy.inf
        else:
            # inequality (2.5) in [1]
            ortho_tol = (
                ortho_const * (k ** 2) * N * eps * vAV_singvals[0] / vAV_singvals[-1]
            )
    # mgs or lanczos is not able to detect an invariant subspace reliably
    if (ortho != "mgs" or N != k) and ortho != "lanczos":
        assert ortho_resn <= ortho_tol

    # check projection residual \| <V_k, A*V_k> - H_k \|
    proj_res = ip_B(P, MAV) - H
    proj_tol = proj_const * (
        ortho_resn * An + arnoldi_resn * krylov.utils.norm(V, ip_B=ip_B)
    )
    assert numpy.linalg.norm(proj_res, 2) <= proj_tol
