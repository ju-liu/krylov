"""
https://www.netlib.org/templates/templates.pdf
"""
from typing import Callable, Optional

import numpy as np

from ._helpers import Identity, Info, aslinearoperator, get_default_inner


def cgs(
    A,
    b,
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

    inner = get_default_inner(b.shape) if inner is None else inner

    def _norm(x):
        xx = inner(x, M @ x)
        if np.any(xx.imag != 0.0):
            raise ValueError("inner product <x, x> gave nonzero imaginary part")
        return np.sqrt(xx.real)

    xk = x0.copy()

    r0 = b - A @ xk
    # common but arbitrary choice:
    r0_ = r0

    r = r0.copy()
    resnorms = [_norm(r0)]

    rho = 1.0
    alpha = None

    # compute error?
    if exact_solution is None:
        errnorms = None
    else:
        errnorms = [_norm(exact_solution - x0)]

    p = np.zeros_like(b)
    q = np.zeros_like(b)

    # rMr = inner(r[1], M @ r[0])

    k = 0
    success = False
    criterion = np.maximum(tol * resnorms[0], atol)
    while True:
        if np.all(resnorms[-1] <= criterion):
            # oh really?
            if not use_explicit_residual:
                resnorms[-1] = _norm(b - A @ xk)

            if np.all(resnorms[-1] <= criterion):
                success = True
                break

        if k == maxiter:
            break

        rho_old = rho
        rho = inner(r0_, r)

        # TODO break-down for rho==0?

        beta = rho / np.where(rho_old != 0.0, rho_old, 1.0)
        u = r + beta * q
        p = u + beta * (q + beta * p)

        v = A @ (M @ p)

        r0v = inner(r0_, v)
        alpha = rho / np.where(r0v != 0.0, r0v, 1.0)

        q = u - alpha * v

        u_ = M @ (u + q)

        xk += alpha * u_
        q_ = A @ u_
        r -= alpha * q_

        if use_explicit_residual:
            resnorms.append(_norm(b - A @ xk))
        else:
            resnorms.append(_norm(r))

        if exact_solution is not None:
            errnorms.append(_norm(exact_solution - xk))

        k += 1

    return xk if success else None, Info(
        success,
        xk,
        k,
        resnorms,
        errnorms,
        num_operations=None,
        arnoldi=None,
    )
