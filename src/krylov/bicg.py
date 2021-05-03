from typing import Callable, Optional

import numpy as np

from ._helpers import Identity, Info, aslinearoperator, get_inner


def bicg(
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

    if M is None:
        M = Identity()
    else:
        M = aslinearoperator(M)

    x0 = np.zeros_like(b) if x0 is None else x0

    inner = get_inner(b.shape) if inner is None else inner

    def _norm(x):
        xx = inner(x, M @ x)
        if np.any(xx.imag != 0.0):
            raise ValueError("inner product <x, x> gave nonzero imaginary part")
        return np.sqrt(xx.real)

    b_norm = _norm(b)

    xk = x0.copy()

    r = np.array([b - A @ xk, b.conj() - A.rmatvec(xk.conj())])
    resnorms = [_norm(r[0])]

    # compute error?
    if exact_solution is None:
        errnorms = None
    else:
        errnorms = [_norm(exact_solution - x0)]

    p = [M @ r[0], M.rmatvec(r[1])]

    rMr = inner(r[1], M @ r[0])

    k = 0
    success = False
    criterion = np.maximum(tol * b_norm, atol)
    while True:
        if np.all(resnorms[-1] <= criterion):
            # oh really?
            if not use_explicit_residual:
                resnorms[-1] = _norm(M @ (b - A @ xk))

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

        xk += alpha * p[0]

        r[0] -= alpha * Ap0
        r[1] -= alpha.conj() * AHp1

        rMr_old = rMr
        rMr = inner(r[1], M @ r[0])
        beta = rMr / np.where(rMr_old != 0, rMr_old, 1.0)

        if use_explicit_residual:
            resnorms.append(_norm(M @ (b - A @ xk)))
        else:
            resnorms.append(_norm(r[0]))

        p[0] = M @ r[0] + beta * p[0]
        p[1] = M.rmatvec(r[1]) + beta.conj() * p[1]

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
