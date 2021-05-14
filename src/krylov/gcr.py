"""
Generalized conjugate residual method

S.C. Eisenstat, H.C. Elman, and H.C. Schultz.
Variational iterative methods for nonsymmetric systems of linear equations.
SIAM J. Numer. Anal., 20, 1983,
<https://doi.org/10.1137/0720023>
"""
from typing import Callable, Optional

import numpy as np

from ._helpers import Identity, Info, aslinearoperator, get_default_inner


def gcr(
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

    A = aslinearoperator(A)

    x0 = np.zeros_like(b) if x0 is None else x0

    inner = get_default_inner(b.shape) if inner is None else inner

    def _norm(x):
        xx = inner(x, x)
        if np.any(xx.imag != 0.0):
            raise ValueError("inner product <x, x> gave nonzero imaginary part")
        return np.sqrt(xx.real)

    b_norm = _norm(b)

    x = x0.copy()

    r = b - A @ x
    resnorms = [_norm(r)]

    # compute error?
    if exact_solution is None:
        errnorms = None
    else:
        errnorms = [_norm(exact_solution - x)]

    s = []
    v = []

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

        s.append(r.copy())
        v.append(A @ s[-1])

        # modified Gram-Schmidt
        for i in range(k):
            alpha = inner(v[-1], v[i])
            v[-1] -= alpha * v[i]
            # ensure As = v
            s[-1] -= alpha * s[i]
        # normalize
        beta = _norm(v[-1])
        v[-1] /= np.where(beta != 0.0, beta, 1.0)
        s[-1] /= np.where(beta != 0.0, beta, 1.0)

        gamma = inner(b, v[-1])
        x += gamma * s[-1]

        if use_explicit_residual:
            r = b - A @ x
        else:
            r -= gamma * v[-1]

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
