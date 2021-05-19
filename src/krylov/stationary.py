from typing import Callable, Optional

import numpy as np
from scipy.linalg import solve_triangular
from scipy.sparse import tril, triu
from scipy.sparse.linalg import spsolve_triangular

from ._helpers import Info, aslinearoperator, get_default_inner


def richardson(*args, omega: float = 1.0, **kwargs):
    return _stationary(lambda r: omega * r, *args, **kwargs)


def jacobi(A, *args, omega: float = 1.0, **kwargs):
    # There's no difference in speed between division and multiplication, so keep D
    # here. <https://gist.github.com/nschloe/7e4cb61dd391b4edbeb10d23038aa98e>
    D = A.diagonal()

    def _update(r):
        return omega * (r.T / D).T

    return _stationary(_update, A, *args, **kwargs)


def gauss_seidel(A, *args, omega: float = 1.0, lower: bool = True, **kwargs):
    if isinstance(A, np.ndarray):

        def tri_solve(y):
            return omega * solve_triangular(A, y, lower=lower)

    else:
        # scipy doesn't accept non-triangular matrices into spsolve_triangular
        # https://github.com/scipy/scipy/issues/14091
        M = tril(A) if lower else triu(A)

        def tri_solve(y):
            return omega * spsolve_triangular(M, y, lower=lower)

    return _stationary(tri_solve, A, *args, **kwargs)


def sor(A, *args, omega: float = 1.0, lower: bool = True, **kwargs):
    """x_{k+1} = xk + omega * (D + omega * L)^{-1} r"""
    d_ = A.diagonal() / omega

    if isinstance(A, np.ndarray):
        A_ = A.copy()
        np.fill_diagonal(A_, d_)

        def tri_solve(y):
            return solve_triangular(A_, y, lower=lower)

    else:
        M = tril(A) if lower else triu(A)
        M.setdiag(d_)

        def tri_solve(y):
            return spsolve_triangular(M, y, lower=lower)

    return _stationary(tri_solve, A, *args, **kwargs)


def ssor(A, *args, omega: float = 1.0, **kwargs):
    """https://en.wikipedia.org/wiki/Successive_over-relaxation

    P = omega / (2 - omega) * (D/omega + L) D^{-1} (D/omega + U)
    x_{k+1} = x_k + P^{-1} r
    """
    d = A.diagonal()

    if isinstance(A, np.ndarray):
        A_ = A.copy()
        np.fill_diagonal(A_, d / omega)

        def solve(y):
            y = solve_triangular(A_, y, lower=True)
            y = (y.T * d).T
            y = solve_triangular(A_, y, lower=False)
            return (2 - omega) / omega * y

    else:
        L = tril(A)
        L.setdiag(d / omega)
        U = triu(A)
        U.setdiag(d / omega)

        def solve(y):
            y = spsolve_triangular(L, y, lower=True)
            y = (y.T * d).T
            y = spsolve_triangular(U, y, lower=False)
            return (2 - omega) / omega * y

    return _stationary(solve, A, *args, **kwargs)


def _stationary(
    update,
    A,
    b,
    x0=None,
    inner: Optional[Callable] = None,
    tol: float = 1e-5,
    atol: float = 1.0e-15,
    maxiter: Optional[int] = None,
    callback: Optional[Callable] = None,
):
    assert len(A.shape) == 2
    assert A.shape[0] == A.shape[1]
    assert A.shape[1] == b.shape[0]

    A = aslinearoperator(A)

    inner = get_default_inner(b.shape) if inner is None else inner

    def _norm(x):
        xx = inner(x, x)
        if np.any(xx.imag != 0.0):
            raise ValueError("inner product <x, x> gave nonzero imaginary part")
        return np.sqrt(xx.real)

    if x0 is None:
        x = np.zeros_like(b)
        r = b.copy()
    else:
        x = x0.copy()
        r = b - A @ x

    if callback is not None:
        callback(x, r)

    resnorms = [_norm(r)]

    k = 0
    success = False
    criterion = np.maximum(tol * resnorms[0], atol)
    while True:
        if np.all(resnorms[-1] <= criterion):
            success = True
            break

        if k == maxiter:
            break

        x += update(r)
        # TODO check which is faster
        r = b - A @ x
        # r -= A @ update

        if callback is not None:
            callback(x, r)

        resnorms.append(_norm(r))

        k += 1

    return x if success else None, Info(success, x, k, resnorms)
