"""
https://www.netlib.org/templates/templates.pdf
"""
from typing import Callable, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike

from ._helpers import Identity, Info, aslinearoperator, get_default_inner


def chebyshev(
    A,
    b: ArrayLike,
    eigenvalue_estimates: Tuple[float, float],
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

    inner = get_default_inner(b.shape) if inner is None else inner

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

    resnorms = [_norm(r)]

    if callback is not None:
        callback(x, r)

    alpha = None
    p = None

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
        r -= alpha * (A @ p)

        if callback is not None:
            callback(x, r)

        resnorms.append(_norm(r))

        k += 1

    return x if success else None, Info(success, x, k, resnorms)
