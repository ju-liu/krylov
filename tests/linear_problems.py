import numpy as np
from scipy.sparse import spdiags


def spd_dense(b_shape):
    a = np.linspace(1.0, 2.0, b_shape[0])
    a[-1] = 1e-2
    A = np.diag(a)
    b = np.ones(b_shape)
    return A, b


def spd_sparse(b_shape):
    a = np.linspace(1.0, 2.0, b_shape[0])
    a[-1] = 1e-2
    n = b_shape[0]
    A = spdiags(a, [0], n, n)
    b = np.ones(b_shape)
    return A, b


def spd_rhs_0(b_shape):
    a = np.linspace(1.0, 2.0, b_shape[0])
    a[-1] = 1e-2
    A = np.diag(a)
    b = np.zeros(b_shape)
    return A, b


def spd_rhs_0sol0():
    a = np.linspace(1.0, 2.0, 5)
    a[-1] = 1e-2
    A = np.diag(a)
    np.random.seed(0)

    b = np.column_stack(
        [
            np.zeros(5),
            np.random.rand(5),
            np.random.rand(5),
        ]
    )
    sol = np.linalg.solve(A, b[:, 1])
    b = np.column_stack(
        [
            np.zeros(5),
            sol,
            np.zeros(5),
        ]
    )
    return A, b


def hpd():
    a = np.array(np.linspace(1.0, 2.0, 5), dtype=complex)
    a[0] = 5.0
    a[-1] = 1.0e-1
    A = np.diag(a)
    A[-1, 0] = 1.0e-1j
    A[0, -1] = -1.0e-1j

    b = np.ones(5, dtype=complex)
    return A, b


def symmetric_indefinite():
    n = 5
    a = np.linspace(1.0, 2.0, n)
    a[-1] = -1.0
    A = np.diag(a)
    b = np.ones(n)
    return A, b


def hermitian_indefinite():
    n = 5
    a = np.array(np.linspace(1.0, 2.0, n), dtype=complex)
    a[-1] = 1e-3
    A = np.diag(a)
    A[-1, 0] = 10.0j
    A[0, -1] = -10.0j
    b = np.ones(n, dtype=complex)
    return A, b


def real_unsymmetric():
    n = 5
    a = np.arange(1, n + 1, dtype=float)
    a[-1] = -10.0
    A = np.diag(a)
    A[0, -1] = 10.0
    b = np.ones(n)
    return A, b


def complex_unsymmetric():
    n = 5
    a = np.arange(1, n + 1).astype(complex)
    a[-1] = -10.0
    A = np.diag(a)
    A[0, -1] = 10.0j
    b = np.full(n, 1.0 + 0.5j)
    return A, b
