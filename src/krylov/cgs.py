"""
https://www.netlib.org/templates/templates.pdf
"""
from typing import Callable, Optional

import numpy as np
from numpy.typing import ArrayLike

from ._helpers import Identity, Info, aslinearoperator, get_default_inner


def cgs(
    A,
    b: ArrayLike,
    M=None,
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
    r0_ = r0

    r = r0.copy()

    if callback:
        callback(x, r)

    resnorms = [_norm(r0)]

    rho = 1.0
    alpha = None

    p = np.zeros_like(b)
    q = np.zeros_like(b)

    # rMr = inner(r[1], M @ r[0])

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

        x += alpha * u_
        q_ = A @ u_
        r -= alpha * q_

        if callback:
            callback(x, r)

        resnorms.append(_norm(r))

        k += 1

    return x if success else None, Info(
        success,
        x,
        k,
        resnorms,
        num_operations=None,
        arnoldi=None,
    )
