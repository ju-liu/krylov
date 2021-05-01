from typing import Callable, Optional

import numpy as np

from ._helpers import Info


def bicg(
    A,
    b,
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

    x0 = np.zeros_like(b) if x0 is None else x0

    x2 = np.array([x0, x0])
    b2 = np.array([b, b])

    if inner is None:
        # np.dot is faster than einsum for flat vectors
        # <https://gist.github.com/nschloe/33b3c93b9bc0768394ba9edee1fda2bc>
        if len(b.shape) == 1:

            def inner(x, y):
                return np.dot(x.conj(), y)

        else:

            def inner(x, y):
                return np.einsum("i...,i...->...", x.conj(), y)

    r2 = np.array(
        [
            b2[0] - A @ x2[0],
            b2[1] - A.T.conj() @ x2[1],
        ]
    )
    rr = inner(r2[0], r2[1])

    resnorms = [np.sqrt(rr)]

    # compute error?
    if exact_solution is None:
        errnorms = None
    else:
        err = exact_solution - x0
        errnorms = [np.sqrt(inner(err, err))]

    p2 = r2.copy()

    b_norm = np.sqrt(inner(b, b))

    k = 0
    success = False
    criterion = np.maximum(tol * b_norm, atol)
    while True:
        if np.all(resnorms[-1] <= criterion):
            success = True
            break
            # # oh really?
            # if not use_explicit_residual:
            #     xk = _get_xk(yk) if xk is None else xk
            #     _, _, rkn2 = get_residual_and_norm2(xk)
            #     resnorms[-1] = np.sqrt(rkn2)

            # if np.all(resnorms[-1] <= criterion):
            #     success = True
            #     break

        if k == maxiter:
            break

        pAp = inner(p2[1], A @ p2[0])

        alpha = inner(r2[1], r2[0]) / np.where(pAp != 0, pAp, 1.0)
        x2 += alpha * p2
        r2 -= alpha * np.array([A @ p2[0], A.T.conj() @ p2[1]])
        rr_old = rr
        rr = inner(r2[0], r2[1])
        resnorms.append(np.sqrt(inner(r2[0], r2[0])))

        beta = rr / np.where(rr_old != 0, rr_old, 1.0)
        p2 *= beta
        p2 += r2

        k += 1

    xk = x2[0]

    return xk if success else None, Info(
        success,
        xk,
        k,
        resnorms,
        errnorms,
        num_operations=None,
        arnoldi=None
        # arnoldi=[V, H, P] if return_arnoldi else None,
    )
