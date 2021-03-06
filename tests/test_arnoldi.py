import numpy as np
import pytest
import scipy

import krylov

from .helpers import (
    get_matrix_comp_nonsymm,
    get_matrix_herm_indef,
    get_matrix_hpd,
    get_matrix_nonsymm,
    get_matrix_spd,
    get_matrix_symm_indef,
)

_B = np.diag(np.linspace(1.0, 5.0, 10))


def _unit_vec(n):
    x = np.zeros(n)
    x[0] = 1.0
    return x


@pytest.mark.parametrize(
    "A",
    [
        get_matrix_spd(),
        # get_matrix_hpd(),
        get_matrix_symm_indef(),
        # get_matrix_herm_indef(),
        get_matrix_nonsymm(),
        # get_matrix_comp_nonsymm(),
    ],
)
@pytest.mark.parametrize("v", [np.ones(10), _unit_vec(10)])
@pytest.mark.parametrize("maxiter", [1, 5, 9, 10])
def test_arnoldi_householder(A, v, maxiter):
    An = np.linalg.norm(A, 2)

    arnoldi = krylov.ArnoldiHouseholder(A, v)
    h_columns = []
    while arnoldi.iter < maxiter and not arnoldi.is_invariant:
        _, h = next(arnoldi)
        h_columns.append(h)

    V = arnoldi.V
    P = V

    # build H from h_columns
    H = np.zeros((arnoldi.iter + 1, arnoldi.iter), dtype=arnoldi.dtype)
    for k, val in enumerate(h_columns):
        H[: len(val), k] = val

    # conditionally cut off last row (which should be 0)
    if arnoldi.is_invariant:
        H = H[:-1]

    def inner(x, y):
        return x.T.conj().dot(y)

    ortho = "householder"
    assert_arnoldi(A, v, V, H, P, maxiter, ortho, M=None, inner=inner, An=An)


@pytest.mark.parametrize(
    "A",
    [
        get_matrix_spd(),
        get_matrix_hpd(),
        get_matrix_symm_indef(),
        get_matrix_herm_indef(),
        get_matrix_nonsymm(),
        get_matrix_comp_nonsymm(),
    ],
)
@pytest.mark.parametrize("v", [np.ones(10), np.eye(10)[0]])
@pytest.mark.parametrize("maxiter", [1, 5, 9, 10])
@pytest.mark.parametrize("M", [None, _B])
@pytest.mark.parametrize(
    "inner",
    [lambda x, y: x.T.conj().dot(y), lambda x, y: x.T.conj().dot(_B.dot(y))],
)
def test_arnoldi_mgs(A, v, maxiter, M, inner):
    An = np.linalg.norm(A, 2)

    arnoldi = krylov.ArnoldiMGS(A, v, M=M, inner=inner)
    h_columns = []
    while arnoldi.iter < maxiter and not arnoldi.is_invariant:
        _, h = next(arnoldi)
        h_columns.append(h)

    V = arnoldi.V
    P = arnoldi.P

    # build H from h_columns
    H = np.zeros((arnoldi.iter + 1, arnoldi.iter), dtype=arnoldi.dtype)
    for k, val in enumerate(h_columns):
        H[: k + 2, k] = val

    # conditionally cut off last row (which should be 0)
    if arnoldi.is_invariant:
        H = H[:-1]

    ortho = "mgs"
    assert_arnoldi(A, v, V, H, P, maxiter, ortho, M, inner, An=An)


@pytest.mark.parametrize(
    "A",
    [
        get_matrix_spd(),
        get_matrix_hpd(),
        get_matrix_symm_indef(),
        get_matrix_herm_indef(),
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

    v0 = v.copy()
    arnoldi = krylov.ArnoldiLanczos(A, v0, M=M, inner=inner)
    V = [arnoldi.v.copy()]
    P = [arnoldi.p.copy()]
    tridiag_H = []
    for _ in range(maxiter):
        if arnoldi.is_invariant:
            break
        v, h, p = next(arnoldi)
        if v is not None:
            V.append(v)
        if p is not None:
            P.append(p)
        print(h)
        tridiag_H.append(h.copy())

    # build the triadiagonal Hessenberg matrix
    k = len(tridiag_H)
    H = np.zeros((k + 1, k), dtype=arnoldi.dtype)
    for i, vals in enumerate(tridiag_H):
        if i == 0:
            H[:2, i] = vals[1:]
        else:
            H[i - 1 : i + 2, i] = vals
    print(H)
    print(arnoldi.is_invariant)
    print(k)
    # cut off the last (0) row
    if arnoldi.is_invariant:
        H = H[:k]

    # k = arnoldi.iter if arnoldi.is_invariant else arnoldi.iter + 1
    # H = arnoldi.H[:k, :k].T

    ortho = "lanczos"
    assert_arnoldi(A, v0, V, H, P, maxiter, ortho, M, inner, An=An)


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
    assert np.linalg.norm(P[0] - v / v1n) <= 1.0e-14

    # check if H is upper Hessenberg
    assert np.all(np.tril(H, -2) == 0.0)

    if lanczos:
        # check that H is Hermitian
        assert np.all(H == H.T.conj())
        # check that H is real
        assert np.all(np.isreal(H))

    # check that subdiagonal-elements are real and non-negative
    d = np.diag(H[1:, :])
    assert np.all(np.abs(d.imag) < 1.0e-14)
    assert np.all(d >= 0.0)

    V = np.column_stack(V)
    P = np.column_stack(P)

    # check Arnoldi residual \| A*V_k - V_{k+1} H \|
    AV = A @ V if invariant else A @ V[:, :-1]
    MAV = AV if M is None else M @ AV
    arnoldi_res = MAV - V @ H

    arnoldi_resnorm = np.linalg.norm(inner(arnoldi_res, arnoldi_res), 2)

    # inequality (2.3) in [1]
    arnoldi_tol = arnoldi_const * k * (N ** 1.5) * eps * An
    assert arnoldi_resnorm <= arnoldi_tol

    # check orthogonality by measuring \| I - <V,V> \|_2
    ortho_res = np.eye(V.shape[1]) - inner(V, P)

    ortho_resnorm = np.linalg.norm(ortho_res, 2)
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
        assert ortho_resnorm <= ortho_tol

    # check projection residual \| <V_k, A*V_k> - H_k \|
    proj_res = inner(P, MAV) - H
    proj_tol = proj_const * (
        ortho_resnorm * An + arnoldi_resnorm * np.sqrt(np.linalg.norm(inner(V, V), 2))
    )
    assert np.linalg.norm(proj_res, 2) <= np.max([proj_tol, eps])
