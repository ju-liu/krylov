import numpy as np


def spd(b_shape):
    a = np.linspace(1.0, 2.0, b_shape[0])
    a[-1] = 1e-2
    A = np.diag(a)
    b = np.ones(b_shape)
    return A, b


def spd_funny_rhs():
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
    A[-1, 0] = 10j
    A[0, -1] = -10j
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
    a = np.array(range(1, n + 1), dtype=complex)
    a[-1] = -1e1
    A = np.diag(a)
    A[0, -1] = 1.0e1j
    b = np.ones(n, dtype=complex)
    return A, b
