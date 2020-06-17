import numpy
import pytest
import scipy.linalg
from numpy.testing import assert_almost_equal, assert_array_equal, assert_equal

import krylov
from helpers import (
    get_ip_Bs,
    get_matrix_comp_nonsymm,
    get_matrix_herm_indef,
    get_matrix_hpd,
    get_matrix_nonsymm,
    get_matrix_spd,
    get_matrix_symm_indef,
)

_factors = [0.0, 1.0, 1.0j, 1.0 + 1.0j, 1e8, 1e-8]


@pytest.mark.parametrize("X", [numpy.eye(10, 5), scipy.linalg.hilbert(10)[:, :5]])
@pytest.mark.parametrize("ip_B", get_ip_Bs())
@pytest.mark.parametrize("reorthos", [0, 1, 2])
def test_qr(X, ip_B, reorthos):
    (N, k) = X.shape
    s = scipy.linalg.svd(X, compute_uv=False)
    Q, R = krylov.utils.qr(X, ip_B=ip_B, reorthos=reorthos)

    # check shapes
    assert Q.shape == (N, k)
    assert R.shape == (k, k)
    # check residual
    assert numpy.linalg.norm(numpy.dot(Q, R) - X, 2) <= 1e-14 * max(s)
    # check orthogonality
    orthotol = 1e-8 if reorthos < 1 else 1e-14
    assert (
        numpy.linalg.norm(krylov.utils.inner(Q, Q, ip_B=ip_B) - numpy.eye(k), 2)
        <= orthotol
    )
    # check if R is upper triangular
    assert numpy.linalg.norm(numpy.tril(R, -1)) == 0


_FGs = [
    numpy.eye(10, 1),
    1j * numpy.eye(10, 1),
    numpy.eye(10, 4),
    numpy.eye(10)[:, -4:],
    numpy.dot(numpy.eye(10, 4), numpy.diag([1, 1e1, 1e2, 1e3])),
    numpy.eye(10, 4),
]


@pytest.mark.parametrize("F", _FGs)
@pytest.mark.parametrize("G", _FGs)
@pytest.mark.parametrize("ip_B", get_ip_Bs())
@pytest.mark.parametrize("compute_vectors", [False, True])
def test_angles(F, G, ip_B, compute_vectors):
    if compute_vectors:
        theta, U, V = krylov.utils.angles(
            F, G, ip_B=ip_B, compute_vectors=compute_vectors
        )
    else:
        theta = krylov.utils.angles(F, G, ip_B=ip_B, compute_vectors=compute_vectors)

    # check shape of theta
    assert theta.shape == (max(F.shape[1], G.shape[1]),)
    # check if theta is sorted
    assert ((theta - numpy.sort(theta)) == 0).all()
    # check that 0 <= theta <= pi/2
    assert (theta >= 0).all()
    assert (theta <= numpy.pi / 2).all()
    # check pi/2 angles if dimensions differ
    n = abs(F.shape[1] - G.shape[1])
    if n > 0:
        # == 0 is safe here
        assert (numpy.abs(theta[-n:] - numpy.pi / 2) == 0).all()
    # check 0 angles if F==G
    if F is G:
        assert numpy.linalg.norm(theta) <= 1e-15

    if compute_vectors:
        # check shapes of U and V
        assert U.shape == F.shape
        assert V.shape == G.shape
        # check that inner_product(U,V) = diag(cos(theta))
        UV = krylov.utils.inner(U, V, ip_B=ip_B)
        assert (
            numpy.linalg.norm(
                UV - numpy.diag(numpy.cos(theta))[: F.shape[1], : G.shape[1]]
            )
            <= 1e-14
        )


def _get_m():
    m = numpy.array(range(1, 11))
    m[-1] = 1.0
    return m


_x = [numpy.ones((10, 1)), numpy.full((10, 1), 1.0j + 1)]


@pytest.mark.parametrize(
    "matrix",
    [
        get_matrix_spd(),
        get_matrix_hpd(),
        get_matrix_symm_indef(),
        get_matrix_herm_indef(),
        get_matrix_nonsymm(),
        get_matrix_comp_nonsymm(),
    ],
)
@pytest.mark.parametrize(
    "get_operator", [lambda A: A, lambda A: krylov.MatrixLinearOperator(A)]
)
@pytest.mark.parametrize("x", _x)
@pytest.mark.parametrize(
    "x0", [numpy.zeros((10, 1)), numpy.linspace(1, 5, 10).reshape((10, 1))] + _x
)
@pytest.mark.parametrize("M", [None, numpy.diag(_get_m())])
@pytest.mark.parametrize("Ml", [None, numpy.diag(_get_m())])
@pytest.mark.parametrize("ip_B", get_ip_Bs())
def test_hegedus(matrix, get_operator, x, x0, M, Ml, ip_B):
    b = numpy.dot(matrix, x)
    A = get_operator(matrix)

    x0new = krylov.utils.hegedus(A, b, x0, M, Ml, ip_B)

    N = len(b)
    shape = (N, N)
    A = krylov.utils.get_linearoperator(shape, A)
    M = krylov.utils.get_linearoperator(shape, M)
    Ml = krylov.utils.get_linearoperator(shape, Ml)

    Mlr0 = Ml * (b - A * x0)
    MMlr0_norm = krylov.utils.norm(Mlr0, M * Mlr0, ip_B=ip_B)

    Mlr0new = Ml * (b - A * x0new)
    MMlr0new_norm = krylov.utils.norm(Mlr0new, M * Mlr0new, ip_B=ip_B)

    assert MMlr0new_norm <= MMlr0_norm + 1e-13


_B = numpy.diag(numpy.linspace(1, 5, 10))


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
    assert_array_equal(p(roots), numpy.zeros((len(roots),)))

    # check if polynomial is normalized at origin
    assert_equal(p(0), 1)

    if numpy.isrealobj(roots):
        interval = numpy.linspace(roots[0], roots[1], 100)
        candidates = p.minmax_candidates()
        c = [roots[0], roots[1]]
        for candidate in candidates:
            if roots[0] <= candidate <= roots[1]:
                c.append(candidate)
        assert_almost_equal(
            numpy.max(numpy.abs(p(interval))), numpy.max(numpy.abs(p(c))), decimal=4
        )
