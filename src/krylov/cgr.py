"""
https://en.wikipedia.org/wiki/Conjugate_residual_method
"""
from typing import Callable, Optional

import numpy as np

from ._helpers import Identity, Info, aslinearoperator, get_inner


def cgr(
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

    x = x0.copy()

    r = b - A @ x
    rAr = inner(r, A @ r)

    resnorms = [_norm(r)]

    # compute error?
    if exact_solution is None:
        errnorms = None
    else:
        errnorms = [_norm(exact_solution - x)]

    p = r.copy()

    k = 0
    success = False
    criterion = np.maximum(tol * b_norm, atol)
    while True:
        if np.all(resnorms[-1] <= criterion):
            if not use_explicit_residual:
                resnorms[-1] = _norm(b - A @ x)

            if np.all(resnorms[-1] <= criterion):
                success = True
                break

        if k == maxiter:
            break

        Ap = A @ p

        ApAp = inner(Ap, Ap)

        alpha = inner(r, A @ r) / np.where(ApAp != 0.0, ApAp, 1.0)
        x += alpha * p
        r -= alpha * Ap

        rAr_old = rAr
        rAr = inner(r, A @ r)
        beta = rAr / np.where(rAr_old != 0.0, rAr_old, 1.0)

        p = r + beta * p

        if use_explicit_residual:
            resnorms.append(_norm(b - A @ x))
        else:
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
