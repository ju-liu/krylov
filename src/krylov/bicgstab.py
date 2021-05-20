"""
H.A. Van der Vorst,
Bi-CGSTAB: A Fast and Smoothly Converging Variant of Bi-CG for the Solution of
Nonsymmetric Linear Systems,
SIAM J. Sci. Stat. Comput. 13 (2): 631–644, 1992,
<https://doi.org/10.1137%2F0913035>

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


def bicgstab(
    A: LinearOperator,
    b: ArrayLike,
    Ml: Optional[LinearOperator] = None,
    Mr: Optional[LinearOperator] = None,
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

    Ml = Identity() if Ml is None else aslinearoperator(Ml)
    Mr = Identity() if Mr is None else aslinearoperator(Mr)

    inner = get_default_inner(b.shape) if inner is None else inner

    def _norm(x):
        xx = inner(x, Ml @ x)
        if np.any(xx.imag != 0.0):
            raise ValueError("inner product <x, x> gave nonzero imaginary part")
        return np.sqrt(xx.real)

    if x0 is None:
        x = np.zeros_like(b)
        r0 = b.copy()
    else:
        x = np.asarray(x0)
        r0 = b - A @ x

    # common but arbitrary choice:
    r0_ = r0

    r = r0.copy()

    if callback is not None:
        callback(x, r)

    resnorms = [_norm(r0)]

    rho = 1.0
    alpha = 1.0
    omega = 1.0

    p = np.zeros_like(b)
    v = np.zeros_like(b)

    # rMr = inner(r[1], M @ r[0])

    k = 0
    success = False
    criterion = np.maximum(tol * resnorms[0], atol)
    while True:
        if np.all(resnorms[-1] <= criterion):
            # oh really?
            resnorms[-1] = _norm(b - A @ x)

            if np.all(resnorms[-1] <= criterion):
                success = True
                break

        if k == maxiter:
            break

        rho_old = rho
        rho = inner(r0_, r)

        # TODO break-down for rho==0?
        rho_old_omega = rho_old * omega
        beta = rho * alpha / np.where(rho_old_omega != 0.0, rho_old_omega, 1.0)

        # rho_ratio = rho / rho_old
        # alpha_omega = alpha / omega
        # beta2 = rho_ratio * alpha_omega

        p = r + beta * (p - omega * v)
        y = Mr @ (Ml @ p)

        v = A @ y

        r0v = inner(r0_, v)
        alpha = rho / np.where(r0v != 0.0, r0v, 1.0)

        s = r - alpha * v

        # TODO norm(s) == resnorm?
        h = x + alpha * y
        resnorm_h = _norm(Ml @ (b - A @ x))
        if np.all(resnorm_h <= criterion):
            resnorms[-1] = resnorm_h
            success = True
            break

        Ml_s = Ml @ s

        z = Mr @ Ml_s
        t = A @ z

        Ml_t = Ml @ t
        tt = inner(Ml_t, Ml_t)
        omega = inner(Ml_t, Ml_s) / np.where(tt != 0.0, tt, 1.0)

        x = h + omega * z
        r = s - omega * t

        if callback is not None:
            callback(x, r)

        resnorms.append(_norm(r))

        k += 1

    return x if success else None, Info(success, x, k, resnorms)
