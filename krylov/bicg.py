import itertools
from typing import Optional

import numpy as np

from ._helpers import Info


def bicg(
    A,
    b,
    exact_solution=None,
    x0=None,
    tol: float = 1e-5,
    atol: float = 1.0e-15,
    maxiter: Optional[int] = None,
    use_explicit_residual: bool = False,
):
    print(">>> krylov bicg")
    assert len(A.shape) == 2
    assert A.shape[0] == A.shape[1]
    assert A.shape[1] == b.shape[0]

    x0 = np.zeros_like(b) if x0 is None else x0

    x2 = np.array([x0, x0])
    b2 = np.array([b, b])

    r2 = np.array(
        [
            b2[0] - A @ x2[0],
            b2[1] - A.T @ x2[1],
        ]
    )
    rr = np.dot(r2[0], r2[1])

    resnorms = [np.sqrt(rr)]

    # compute error?
    if exact_solution is None:
        errnorms = None
    else:
        err = exact_solution - x0
        errnorms = [np.sqrt(inner(err, err))]

    p2 = r2.copy()

    b_norm = np.sqrt(np.dot(b, b))

    k = 0
    success = False
    criterion = np.maximum(tol * b_norm, atol)
    while True:
        print("XK")
        print(x2)
        if np.all(resnorms[-1] <= criterion):
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

        alpha = np.dot(r2[1], r2[0]) / np.dot(p2[1], A @ p2[0])
        x2 += alpha * p2
        r2 -= alpha * np.array([A @ p2[0], A.T @ p2[1]])
        rr_old = rr
        rr = np.dot(r2[0], r2[1])
        # if rr < tol
        resnorms.append(np.sqrt(np.dot(r2[0], r2[0])))

        beta = rr / rr_old
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
