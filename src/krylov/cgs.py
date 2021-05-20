"""
Peter Sonneveld,
CGS: A fast Lanczos-Type Solver for Nonsymmetric Linear Systems,
SIAM J. Sci. Stat. Comput.,
10(1):36-52, 1989,
<https://doi.org/10.1137/0910004>.

https://www.netlib.org/templates/templates.pdf
"""
from typing import Callable, Optional

import numpy as np
from numpy.typing import ArrayLike

from ._helpers import (
    Identity,
    Info,
    LinearOperator,
    aslinearoperator,
    get_default_inner,
)


def cgs(
    A: LinearOperator,
    b: ArrayLike,
    M: Optional[LinearOperator] = None,
    x0: Optional[ArrayLike] = None,
    inner: Optional[Callable] = None,
    tol: float = 1e-5,
    atol: float = 1.0e-15,
    maxiter: Optional[int] = None,
    callback: Optional[Callable] = None,
):
    b = np.asarray(b)

    assert len(A.shape) == 2
    assert A.shape[0] == A.shape[1]
    assert A.shape[1] == b.shape[0]

    A = aslinearoperator(A)

    M = Identity() if M is None else aslinearoperator(M)

    inner = get_default_inner(b.shape) if inner is None else inner

    def _norm(x):
        xx = inner(x, M @ x)
        if np.any(xx.imag != 0.0):
            raise ValueError("inner product <x, x> gave nonzero imaginary part")
        return np.sqrt(xx.real)

    if x0 is None:
        x = np.zeros_like(b)
        r0 = b.copy()
    else:
        x = np.array(x0)
        r0 = b - A @ x

    # common but arbitrary choice:
    rp = r0

    r = r0.copy()

    if callback:
        callback(x, r)

    resnorms = [_norm(r)]

    rho = 1.0
    alpha = None

    p = np.zeros_like(b)
    q = np.zeros_like(b)

    k = 0
    success = False
    criterion = np.maximum(tol * resnorms[0], atol)
    while True:
        if np.all(resnorms[-1] <= criterion):
            resnorms[-1] = _norm(b - A @ x)
            if np.all(resnorms[-1] <= criterion):
                success = True
                break

        if k == maxiter:
            break

        rho_old = rho
        rho = inner(rp, r)

        # TODO break-down for rho==0?

        beta = rho / np.where(rho_old != 0.0, rho_old, 1.0)
        u = r + beta * q
        p = u + beta * (q + beta * p)

        v = A @ (M @ p)

        s = inner(rp, v)
        alpha = rho / np.where(s != 0.0, s, 1.0)

        q = u - alpha * v

        u_ = M @ (u + q)

        x += alpha * u_
        r -= alpha * (A @ u_)

        if callback:
            callback(x, r)

        resnorms.append(_norm(r))

        k += 1

    return x if success else None, Info(success, x, k, resnorms)
