from typing import Callable, Optional

import numpy as np
from numpy.typing import ArrayLike

from ._helpers import Identity, Info, aslinearoperator, get_default_inner


def bicg(
    A,
    b: ArrayLike,
    M=None,
    x0: Optional[ArrayLike]=None,
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
        r = np.array([b, b.conj()])
    else:
        x = np.asarray(x0).copy()
        r = np.array([b - A @ x, b.conj() - A.rmatvec(x.conj())])

    resnorms = [_norm(r[0])]

    if callback is not None:
        callback(x, r)

    # make sure to copy, in case M is the Identity
    p = [(M @ r[0]).copy(), M.rmatvec(r[1]).copy()]

    rMr = inner(r[1], M @ r[0])

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

        Ap0 = A @ p[0]
        AHp1 = A.rmatvec(p[1])

        pAp = inner(p[1], Ap0)
        # same as
        # pAp2 = inner(AHp1, p[0])

        alpha = rMr / np.where(pAp != 0, pAp, 1.0)

        x += alpha * p[0]

        r[0] -= alpha * Ap0
        r[1] -= alpha.conj() * AHp1

        rMr_old = rMr
        rMr = inner(r[1], M @ r[0])
        beta = rMr / np.where(rMr_old != 0, rMr_old, 1.0)

        if callback is not None:
            callback(x, r)

        resnorms.append(_norm(r[0]))

        p[0] = M @ r[0] + beta * p[0]
        p[1] = M.rmatvec(r[1]) + beta.conj() * p[1]

        k += 1

    return x if success else None, Info(
        success,
        x,
        k,
        resnorms,
        errnorms=None,
        num_operations=None,
        arnoldi=None,
    )
