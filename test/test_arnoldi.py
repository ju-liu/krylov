import numpy as np
import pytest
import scipy

import krylov

from .helpers import get_matrix_comp_nonsymm  # get_matrix_herm_indef,
from .helpers import (
    get_matrix_hpd,
    get_matrix_nonsymm,
    get_matrix_spd,
    get_matrix_symm_indef,
)

_B = np.diag(np.linspace(1.0, 5.0, 10))


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
@pytest.mark.parametrize("v", [np.ones(10), np.eye(10)[0]])
@pytest.mark.parametrize("maxiter", [1, 5, 9, 10])
@pytest.mark.parametrize("ortho", ["mgs", "dmgs", "house"])
@pytest.mark.parametrize("M", [None, _B])
@pytest.mark.parametrize(
    "inner",
    [lambda x, y: x.T.conj().dot(y), lambda x, y: x.T.conj().dot(_B.dot(y))],
)
def test_arnoldi(A, v, maxiter, ortho, M, inner):
    An = np.linalg.norm(A, 2)

    if ortho == "house" and (inner is not None or M is not None):
        return

    V, H, P = krylov.arnoldi(A, v, maxiter=maxiter, ortho=ortho, M=M, inner=inner)
    assert_arnoldi(A, v, V, H, P, maxiter, ortho, M, inner, An=An)


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
@pytest.mark.parametrize("v", [np.ones(10), np.eye(10)[0]])
@pytest.mark.parametrize("maxiter", [1, 5, 9, 10])
@pytest.mark.parametrize("M", [None, _B])
@pytest.mark.parametrize(
    "inner",
    [lambda x, y: x.T.conj().dot(y), lambda x, y: x.T.conj().dot(_B.dot(y))],
)
def test_arnoldi_lanczos(A, v, maxiter, M, inner):
    An = np.linalg.norm(A, 2)
    ortho = "lanczos"

    V, H, P = krylov.arnoldi(A, v, maxiter=maxiter, ortho=ortho, M=M, inner=inner)
    assert_arnoldi(A, v, V, H, P, maxiter, ortho, M, inner, An=An)


def assert_arnoldi(
    A,
    v,
    V,
    H,
    P,
    maxiter,
    ortho,
    M,
    inner,
    lanczos=False,
    arnoldi_const=1,
    ortho_const=1,
    proj_const=10,
    An=None,
):
    # The checks in this function are based on
    #
    # [1] Drkosova, Greenbaum, Rozloznik and Strakos. Numerical Stability of GMRES.
    #     1995. BIT.
    N = v.shape[0]
    if An is None:
        An = np.linalg.norm(A, 2)
    eps = np.finfo(np.double).eps

    k = H.shape[1]

    # maxiter respected?
    assert k <= maxiter

    invariant = H.shape[0] == k
    # check shapes of V and H
    assert len(V) == len(H)

    if P is None:
        P = V

    # check that the initial vector is correct
    Mv = v if M is None else M @ v
    v1n = np.sqrt(inner(v, Mv))
    assert np.linalg.norm(P[0] - v / v1n) <= 1e-14

    # check if H is Hessenberg
    assert np.linalg.norm(np.tril(H, -2)) == 0
    if lanczos:
        # check if H is Hermitian
        assert np.linalg.norm(H - H.T.conj()) == 0
        # check if H is real
        assert np.isreal(H).all()

    # check if subdiagonal-elements are real and non-negative
    d = np.diag(H[1:, :])
    assert (np.abs(d.imag) < 1.0e-15).all()
    assert np.all(d >= 0)

    V = np.column_stack(V)
    P = np.column_stack(P)

    # check Arnoldi residual \| A*V_k - V_{k+1} H \|
    AV = A @ V if invariant else A @ V[:, :-1]
    MAV = AV if M is None else M @ AV
    arnoldi_res = MAV - V @ H
    arnoldi_resn = np.sqrt(inner(arnoldi_res, arnoldi_res)[0][0])

    # inequality (2.3) in [1]
    arnoldi_tol = arnoldi_const * k * (N ** 1.5) * eps * An
    assert arnoldi_resn <= arnoldi_tol

    # check orthogonality by measuring \| I - <V,V> \|_2
    ortho_res = np.eye(V.shape[1]) - inner(V, P)

    ortho_resn = np.linalg.norm(ortho_res, 2)
    if ortho == "house":
        ortho_tol = ortho_const * (k ** 1.5) * N * eps  # inequality (2.4) in [1]
    else:
        vAV_singvals = scipy.linalg.svd(
            np.column_stack([V[:, [0]], (MAV[:, :-1] if invariant else MAV)]),
            compute_uv=False,
        )
        if vAV_singvals[-1] == 0:
            ortho_tol = np.inf
        else:
            # inequality (2.5) in [1]
            ortho_tol = (
                ortho_const * (k ** 2) * N * eps * vAV_singvals[0] / vAV_singvals[-1]
            )
    # mgs or lanczos is not able to detect an invariant subspace reliably
    if (ortho != "mgs" or N != k) and ortho != "lanczos":
        assert ortho_resn <= ortho_tol

    # check projection residual \| <V_k, A*V_k> - H_k \|
    proj_res = inner(P, MAV) - H
    proj_tol = proj_const * (
        ortho_resn * An + arnoldi_resn * np.sqrt(np.linalg.norm(inner(V, V), 2))
    )
    assert np.linalg.norm(proj_res, 2) <= proj_tol
