from typing import Callable, Optional

import numpy as np

from ._helpers import Identity, Info, aslinearoperator, get_inner


def bicgstab(
    A,
    b,
    Ml=None,
    Mr=None,
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

    Ml = Identity() if Ml is None else aslinearoperator(Ml)
    Mr = Identity() if Mr is None else aslinearoperator(Mr)

    x0 = np.zeros_like(b) if x0 is None else x0

    inner = get_inner(b.shape) if inner is None else inner

    def _norm(x):
        xx = inner(x, Ml @ x)
        if np.any(xx.imag != 0.0):
            raise ValueError("inner product <x, x> gave nonzero imaginary part")
        return np.sqrt(xx.real)

    b_norm = _norm(b)

    xk = x0.copy()

    r0 = b - A @ xk
    # common but arbitrary choice:
    r0_ = r0

    r = r0.copy()
    resnorms = [_norm(r0)]

    rho = 1.0
    alpha = 1.0
    omega = 1.0

    # compute error?
    if exact_solution is None:
        errnorms = None
    else:
        errnorms = [_norm(exact_solution - x0)]

    p = np.zeros_like(b)
    v = np.zeros_like(b)

    # rMr = inner(r[1], M @ r[0])

    k = 0
    success = False
    criterion = np.maximum(tol * b_norm, atol)
    while True:
        if np.all(resnorms[-1] <= criterion):
            # oh really?
            if not use_explicit_residual:
                resnorms[-1] = _norm(Ml @ (b - A @ xk))

            if np.all(resnorms[-1] <= criterion):
                success = True
                break

        if k == maxiter:
            break

        rho_old = rho
        rho = inner(r0_, r)

        rho_old_omega = rho_old * omega
        beta = rho * alpha / np.where(rho_old_omega != 0.0, rho_old_omega, 1.0)

        p = r + beta * (p - omega * v)
        y = Mr @ p

        v = A @ y

        r0v = inner(r0_, v)
        alpha = rho / np.where(r0v != 0.0, r0v, 1.0)

        h = xk + alpha * y
        resnorm_h = _norm(Ml @ (b - A @ xk))
        if np.all(resnorm_h <= criterion):
            resnorms[-1] = resnorm_h
            success = True
            break

        s = r - alpha * v

        z = Mr @ s
        t = A @ z

        Ml_t = Ml @ t
        Ml_s = Ml @ s
        tt = inner(Ml_t, Ml_t)
        omega = inner(Ml_t, Ml_s) / np.where(tt != 0.0, tt, 1.0)

        xk = h + omega * z

        r = s - omega * t

        if use_explicit_residual:
            resnorms.append(_norm(Ml @ (b - A @ xk)))
        else:
            resnorms.append(_norm(r))

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
