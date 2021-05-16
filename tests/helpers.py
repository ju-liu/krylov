import numpy as np


def assert_consistent(A, b, info, sol, tol):
    res = b - A @ info.xk
    resnorm = np.sqrt(np.einsum("i...,i...->...", res, res.conj()))
    bnorm = np.sqrt(np.einsum("i...,i...->...", b, b.conj()))

    if info.success:
        assert sol.shape == b.shape
        assert np.all(resnorm < tol * (1.0 + bnorm))
        # <https://stackoverflow.com/a/61800084/353337>
        assert np.may_share_memory(sol, info.xk)

    print()
    print(resnorm)
    print(info.resnorms)

    assert np.issubdtype(np.asarray(info.resnorms).dtype, np.floating)
    if info.errnorms is not None:
        print(info.errnorms)
        assert np.issubdtype(np.asarray(info.errnorms).dtype, np.floating)

    assert np.all(np.abs(resnorm - info.resnorms[-1]) <= 1.0e-12 * (1 + resnorm))
    # resnorm shape
    assert np.asarray(info.resnorms).shape == (info.numsteps + 1, *b.shape[1:])

    # errnorm shape
    if info.errnorms is not None:
        assert np.asarray(info.errnorms).shape == (info.numsteps + 1, *b.shape[1:])


def get_matrix_spd():
    a = np.linspace(1, 2, 10)
    a[-1] = 1e-2
    return np.diag(a)


def get_matrix_hpd():
    a = np.array(np.linspace(1, 2, 10), dtype=complex)
    a[0] = 5
    a[-1] = 1e-1
    A = np.diag(a)
    A[-1, 0] = 1e-1j
    A[0, -1] = -1e-1j
    return A


def get_matrix_symm_indef():
    a = np.linspace(1, 2, 10)
    a[-1] = -1
    return np.diag(a)


def get_matrix_herm_indef():
    a = np.array(np.linspace(1, 2, 10), dtype=complex)
    a[-1] = 1e-3
    A = np.diag(a)
    A[-1, 0] = 10j
    A[0, -1] = -10j
    return A


def get_matrix_nonsymm():
    a = np.array(range(1, 11), dtype=float)
    a[-1] = -1e1
    A = np.diag(a)
    A[0, -1] = 1e1
    return A


def get_matrix_comp_nonsymm():
    a = np.array(range(1, 11), dtype=complex)
    a[-1] = -1e1
    A = np.diag(a)
    A[0, -1] = 1.0e1j
    return A


def get_matrices(
    spd=True,
    hpd=True,
    symm_indef=True,
    herm_indef=True,
    nonsymm=True,
    comp_nonsymm=True,
):
    matrices = []
    if spd:
        matrices.append(get_matrix_spd())
    if hpd:
        matrices.append(get_matrix_hpd())
    if symm_indef:
        matrices.append(get_matrix_symm_indef())
    if herm_indef:
        matrices.append(get_matrix_herm_indef())
    if nonsymm:
        matrices.append(get_matrix_nonsymm())
    if comp_nonsymm:
        matrices.append(get_matrix_comp_nonsymm())
    return matrices


_matrices_herm = [
    get_matrix_spd(),
    # get_matrix_hpd(),
    # get_matrix_symm_indef(),
    # get_matrix_herm_indef(),
]
_matrices_nonherm = [get_matrix_nonsymm(), get_matrix_comp_nonsymm()]


def get_inners():
    B = np.diag(np.linspace(1, 5, 10))
    return [
        lambda x, y: np.dot(x.T.conj(), y),
        lambda x, y: np.dot(x.T.conj(), np.dot(B, y)),
    ]
