import numpy as np
import pytest
import scipy.linalg
from numpy.testing import assert_almost_equal, assert_array_equal, assert_equal

import krylov

from .helpers import (
    get_inners,
    get_matrix_comp_nonsymm,
    get_matrix_herm_indef,
    get_matrix_hpd,
    get_matrix_nonsymm,
    get_matrix_spd,
    get_matrix_symm_indef,
)


@pytest.mark.parametrize("X", [np.eye(10, 5), scipy.linalg.hilbert(10)[:, :5]])
@pytest.mark.parametrize("inner", get_inners())
@pytest.mark.parametrize("reorthos", [0, 1, 2])
def test_qr(X, inner, reorthos):
    (N, k) = X.shape
    s = scipy.linalg.svd(X, compute_uv=False)
    Q, R = krylov.utils.qr(X, inner=inner, reorthos=reorthos)

    # check shapes
    assert Q.shape == (N, k)
    assert R.shape == (k, k)
    # check residual
    assert np.linalg.norm(np.dot(Q, R) - X, 2) <= 1e-14 * max(s)
    # check orthogonality
    orthotol = 1e-8 if reorthos < 1 else 1e-14
    assert np.linalg.norm(inner(Q, Q) - np.eye(k), 2) <= orthotol
    # check if R is upper triangular
    assert np.linalg.norm(np.tril(R, -1)) == 0


_FGs = [
    np.eye(10, 1),
    1j * np.eye(10, 1),
    np.eye(10, 4),
    np.eye(10)[:, -4:],
    np.dot(np.eye(10, 4), np.diag([1, 1e1, 1e2, 1e3])),
    np.eye(10, 4),
]


@pytest.mark.parametrize("F", _FGs)
@pytest.mark.parametrize("G", _FGs)
@pytest.mark.parametrize("inner", get_inners())
@pytest.mark.parametrize("compute_vectors", [False, True])
def test_angles(F, G, inner, compute_vectors):
    if compute_vectors:
        theta, U, V = krylov.utils.angles(
            F, G, inner=inner, compute_vectors=compute_vectors
        )
    else:
        theta = krylov.utils.angles(F, G, inner=inner, compute_vectors=compute_vectors)

    # check shape of theta
    assert theta.shape == (max(F.shape[1], G.shape[1]),)
    # check if theta is sorted
    assert ((theta - np.sort(theta)) == 0).all()
    # check that 0 <= theta <= pi/2
    assert (theta >= 0).all()
    assert (theta <= np.pi / 2).all()
    # check pi/2 angles if dimensions differ
    n = abs(F.shape[1] - G.shape[1])
    if n > 0:
        # == 0 is safe here
        assert (np.abs(theta[-n:] - np.pi / 2) == 0).all()
    # check 0 angles if F==G
    if F is G:
        assert np.linalg.norm(theta) <= 1e-15

    if compute_vectors:
        # check shapes of U and V
        assert U.shape == F.shape
        assert V.shape == G.shape
        # check that inner_product(U,V) = diag(cos(theta))
        UV = inner(U, V)
        assert (
            np.linalg.norm(UV - np.diag(np.cos(theta))[: F.shape[1], : G.shape[1]])
            <= 1e-14
        )


def _get_m():
    m = np.array(range(1, 11))
    m[-1] = 1.0
    return m


_x = [np.ones((10, 1)), np.full((10, 1), 1.0j + 1)]


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
@pytest.mark.parametrize("x", _x)
@pytest.mark.parametrize(
    "x0", [np.zeros((10, 1)), np.linspace(1, 5, 10).reshape((10, 1))] + _x
)
@pytest.mark.parametrize("M", [None, np.diag(_get_m())])
@pytest.mark.parametrize("Ml", [None, np.diag(_get_m())])
@pytest.mark.parametrize("inner", get_inners())
def test_hegedus(A, x, x0, M, Ml, inner):
    b = A @ x

    x0new = krylov.utils.hegedus(A, b, x0, M, Ml, inner)

    r0 = b - A @ x0
    Mlr0 = r0 if Ml is None else Ml @ r0
    MMlr0 = Mlr0 if M is None else M @ Mlr0
    MMlr0_norm = np.sqrt(inner(Mlr0, MMlr0))

    r0new = b - A @ x0new
    Mlr0new = r0new if Ml is None else Ml @ r0new
    MMlr0new = Mlr0new if M is None else M @ Mlr0new
    MMlr0new_norm = np.sqrt(inner(Mlr0new, MMlr0new))

    assert MMlr0new_norm <= MMlr0_norm + 1e-13


def test_strakos():
    krylov.utils.strakos(5)


_B = np.diag(np.linspace(1, 5, 10))


def test_gap():
    assert_almost_equal(krylov.utils.gap([1, 2], [-4, 3]), 1)
    assert_almost_equal(krylov.utils.gap(5, -5), 10)
    assert_almost_equal(krylov.utils.gap([-5, 5], -5), 0)
    assert_almost_equal(krylov.utils.gap(5, -5, mode="interval"), 10)
    assert_almost_equal(krylov.utils.gap(5, [-5, 6], mode="interval"), 1)
    assert_almost_equal(krylov.utils.gap(-5, [-5, 6], mode="interval"), 0)
    assert krylov.utils.gap([-5, 5], [0], mode="interval") is None


def test_Interval():
    Interval = krylov.utils.Interval
    Intervals = krylov.utils.Intervals

    I = Interval(-2, -1)
    J = Interval(1, 2)
    K = Interval(-10, 1.5)
    L = Interval(5)
    M = Interval(-100, -50)
    N = Interval(50, 100)

    assert (I & J) is None
    assert (I | J) is None
    assert (J & K).left == 1
    assert (J & K).right == 1.5
    assert (J | K).left == -10
    assert (J | K).right == 2

    ints = Intervals([I, M])
    assert ints.max() == -1
    assert ints.min() == -100
    assert ints.max_neg() == -1
    assert ints.min_pos() is None

    ints = Intervals([I, J, M, N])
    assert ints.max() == 100
    assert ints.min_pos() == 1
    assert ints.max_neg() == -1
    assert ints.min() == -100
    assert ints.contains(0) is False

    ints = Intervals([I, J, K, L])
    assert ints.max() == 5
    assert ints.min_pos() is None
    assert ints.max_neg() is None
    assert ints.min() == -10
    assert ints.max_abs() == 10
    assert ints.contains(0) is True
    assert_array_equal(ints.get_endpoints(), [-10, 2, 5])


@pytest.mark.parametrize(
    "roots", [[1, 2], [1, 1j], [1, 2, 1e8], [1, 2, 1e8, 1e8 + 1e-3]]
)
def test_NormalizedRootsPolynomial(roots):
    p = krylov.utils.NormalizedRootsPolynomial(roots)

    # check if roots are exactly (!) zero
    assert_array_equal(p(roots), np.zeros((len(roots),)))

    # check if polynomial is normalized at origin
    assert_equal(p(0), 1)

    if np.isrealobj(roots):
        interval = np.linspace(roots[0], roots[1], 100)
        candidates = p.minmax_candidates()
        c = [roots[0], roots[1]]
        for candidate in candidates:
            if roots[0] <= candidate <= roots[1]:
                c.append(candidate)
        assert_almost_equal(
            np.max(np.abs(p(interval))), np.max(np.abs(p(c))), decimal=4
        )
