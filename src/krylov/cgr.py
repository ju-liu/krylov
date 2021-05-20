"""
Yousef Saad,
Iterative methods for sparse linear systems (2nd ed.),
page 194, SIAM.

https://en.wikipedia.org/wiki/Conjugate_residual_method
"""
from typing import Callable, Optional

import numpy as np
from numpy.typing import ArrayLike

from ._helpers import Identity, Info, aslinearoperator, get_default_inner


def cgr(
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

    if x0 is None:
        x = np.zeros_like(b)
        r = b.copy()
    else:
        x = np.array(x0)
        r = b - A @ x0

    r = M @ r

    inner = get_default_inner(b.shape) if inner is None else inner

    def _norm(x):
        xx = inner(x, x)
        if np.any(xx.imag != 0.0):
            raise ValueError("inner product <x, x> gave nonzero imaginary part")
        return np.sqrt(xx.real)

    Ar = A @ r
    rAr = inner(r, Ar)

    resnorms = [_norm(r)]

    if callback is not None:
        callback(x, r)

    p = r.copy()
    Ap = Ar.copy()

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

        MAp = M @ Ap
        ApMAp = inner(Ap, MAp)
        alpha = rAr / np.where(ApMAp != 0.0, ApMAp, 1.0)

        x += alpha * p
        r -= alpha * MAp

        Ar = A @ r
        rAr_old = rAr
        rAr = inner(r, Ar)
        beta = rAr / np.where(rAr_old != 0.0, rAr_old, 1.0)

        p = r + beta * p
        Ap = Ar + beta * Ap

        if callback is not None:
            callback(x, r)

        resnorms.append(_norm(r))

        k += 1

    return x if success else None, Info(success, x, k, resnorms)
