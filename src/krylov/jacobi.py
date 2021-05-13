"""
https://www.netlib.org/templates/templates.pdf
"""
from typing import Callable, Optional

import numpy as np

from ._helpers import Info, get_inner


def jacobi(
    A,
    b,
    # https://en.wikipedia.org/wiki/Jacobi_method#Weighted_Jacobi_method
    omega: float = 2.0 / 3.0,
    exact_solution=None,
    x0=None,
    inner: Optional[Callable] = None,
    tol: float = 1e-5,
    atol: float = 1.0e-15,
    maxiter: Optional[int] = None,
):
    assert len(A.shape) == 2
    assert A.shape[0] == A.shape[1]
    assert A.shape[1] == b.shape[0]

    # There's no difference in speed between division and multiplication, so keep D
    # here. <https://gist.github.com/nschloe/7e4cb61dd391b4edbeb10d23038aa98e>
    if isinstance(A, np.ndarray):
        D = np.diag(A)
    else:
        # this works for scipy sparse matrices
        D = A.diagonal()

    x0 = np.zeros_like(b) if x0 is None else x0

    inner = get_inner(b.shape) if inner is None else inner

    def _norm(x):
        xx = inner(x, x)
        if np.any(xx.imag != 0.0):
            raise ValueError("inner product <x, x> gave nonzero imaginary part")
        return np.sqrt(xx.real)

    b_norm = _norm(b)

    x = x0.copy()
    r = b - A @ x
    resnorms = [_norm(r)]

    # compute error?
    if exact_solution is None:
        errnorms = None
    else:
        errnorms = [_norm(exact_solution - x)]

    k = 0
    success = False
    criterion = np.maximum(tol * b_norm, atol)
    while True:
        if np.all(resnorms[-1] <= criterion):
            success = True
            break

        if k == maxiter:
            break

        x -= omega * (r.T / D).T
        r = b - A @ x

        resnorms.append(_norm(r))

        if exact_solution is not None:
            errnorms.append(_norm(exact_solution - x))

        k += 1

    return x if success else None, Info(
        success,
        x,
        k,
        resnorms,
        errnorms,
        num_operations=None,
        arnoldi=None,
    )
