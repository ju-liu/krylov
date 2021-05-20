"""
Generalized conjugate residual method

S.C. Eisenstat, H.C. Elman, and H.C. Schultz.
Variational iterative methods for nonsymmetric systems of linear equations.
SIAM J. Numer. Anal., 20, 1983,
<https://doi.org/10.1137/0720023>
"""
from typing import Callable, Optional

import numpy as np
from numpy.typing import ArrayLike

from ._helpers import Info, aslinearoperator, get_default_inner


# TODO preconditioner
def gcr(
    A,
    b: ArrayLike,
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

    if x0 is None:
        x = np.zeros_like(b)
        r = b.copy()
    else:
        x = np.array(x0)
        r = b - A @ x0

    inner = get_default_inner(b.shape) if inner is None else inner

    def _norm(x):
        xx = inner(x, x)
        if np.any(xx.imag != 0.0):
            raise ValueError("inner product <x, x> gave nonzero imaginary part")
        return np.sqrt(xx.real)

    if callback is not None:
        callback(x, r)

    resnorms = [_norm(r)]

    s = []
    v = []

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
        r -= gamma * v[-1]

        if callback is not None:
            callback(x, r)

        resnorms.append(_norm(r))

        k += 1

    return x if success else None, Info(success, x, k, resnorms)
