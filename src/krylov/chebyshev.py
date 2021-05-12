"""
https://www.netlib.org/templates/templates.pdf
"""
from typing import Callable, Optional

import numpy as np

from ._helpers import Identity, Info, aslinearoperator, get_inner


def chebyshev(
    A,
    b,
    eigenvalue_estimates,
    M=None,
    exact_solution=None,
    x0=None,
    inner: Optional[Callable] = None,
    tol: float = 1e-5,
    atol: float = 1.0e-15,
    maxiter: Optional[int] = None,
    use_explicit_residual: bool = False,
):
    assert len(A.shape) == 2
    assert A.shape[0] == A.shape[1]
    assert A.shape[1] == b.shape[0]

    A = aslinearoperator(A)
    M = Identity() if M is None else aslinearoperator(M)

    x0 = np.zeros_like(b) if x0 is None else x0

    inner = get_inner(b.shape) if inner is None else inner

    def _norm(x):
        xx = inner(x, M @ x)
        if np.any(xx.imag != 0.0):
            raise ValueError("inner product <x, x> gave nonzero imaginary part")
        return np.sqrt(xx.real)

    assert len(eigenvalue_estimates) == 2
    assert eigenvalue_estimates[0] <= eigenvalue_estimates[1]
    lmin, lmax = eigenvalue_estimates

    d = (lmax + lmin) / 2
    c = (lmax - lmin) / 2

    b_norm = _norm(b)

    x = x0.copy()

    r = b - A @ x

    resnorms = [_norm(r)]

    # compute error?
    if exact_solution is None:
        errnorms = None
    else:
        errnorms = [_norm(exact_solution - x)]

    alpha = None
    p = None

    k = 0
    success = False
    criterion = np.maximum(tol * b_norm, atol)
    while True:
        if np.all(resnorms[-1] <= criterion):
            # oh really?
            if not use_explicit_residual:
                resnorms[-1] = _norm(b - A @ x)

            if np.all(resnorms[-1] <= criterion):
                success = True
                break

        if k == maxiter:
            break

        z = M @ r

        if k == 0:
            p = z.copy()
            alpha = 1.0 / d
        else:
            beta = 0.5 * (c * alpha) ** 2
            if k > 1:
                beta *= 0.5

            alpha = 1.0 / (d - beta / alpha)
            p = z + beta * p

        x += alpha * p

        if use_explicit_residual:
            r = b - A @ x
        else:
            r -= alpha * (A @ p)

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
