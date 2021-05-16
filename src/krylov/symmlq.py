"""
Other implementations:

https://github.com/PythonOptimizers/pykrylov/blob/master/pykrylov/symmlq/symmlq.py
https://www.netlib.org/linalg/symmlq
https://www.mcs.anl.gov/petsc/petsc-current/src/ksp/ksp/impls/symmlq/symmlq.c.html
"""
from typing import Callable, Optional

import numpy as np

from ._helpers import Identity, Info, aslinearoperator, get_default_inner


def symmlq(
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

    M = Identity() if M is None else aslinearoperator(M)

    inner = get_default_inner(b.shape) if inner is None else inner

    def _norm(x):
        xx = inner(x, x)
        if np.any(xx.imag != 0.0):
            raise ValueError("inner product <x, x> gave nonzero imaginary part")
        return np.sqrt(xx.real)

    # Need to store two previous values of ceta, c, and s.
    # c[0] is the current c, c[-1] the last, c[-2] the second last.
    ceta = [None, None, 0.0]
    c = [1.0, None, 1.0]
    s = [0.0, None, 0.0]

    u_old = np.zeros_like(b)
    v_old = np.zeros_like(b)
    w = np.zeros_like(b)
    w_bar = np.zeros_like(b)

    if x0 is None:
        x = np.zeros_like(b)
        r = b.copy()
    else:
        x = x0.copy()
        r = b - A @ x0

    resnorms = [_norm(r)]

    z = M @ r

    dp = inner(r, z)
    dp = np.sqrt(dp)

    beta = dp
    beta1 = beta
    s_prod = np.abs(beta1)

    v = r / beta
    u = z / beta
    w_bar = u.copy()
    # np_ = _norm(z)

    # compute error?
    if exact_solution is None:
        errnorms = None
    else:
        errnorms = [_norm(exact_solution - x)]

    k = 0
    success = False
    criterion = np.maximum(tol * resnorms[0], atol)
    while True:
        if np.all(resnorms[-1] <= criterion):
            if not use_explicit_residual:
                resnorms[-1] = _norm(b - A @ x)

            if np.all(resnorms[-1] <= criterion):
                success = True
                break

        if k == maxiter:
            break

        if k > 0:
            v_old = v.copy()
            u_old = u.copy()

            v = r.copy()
            v *= 1.0 / beta

            u = z.copy()
            u *= 1.0 / beta

            w = c[0] * w_bar + s[0] * u
            w_bar = -s[0] * w_bar + c[0] * u
            x = x + ceta[0] * w

            ceta[-2] = ceta[-1]
            ceta[-1] = ceta[0]

        # Lanczos
        r = A @ u
        alpha = inner(u, r)
        # preconditioning
        z = M @ r

        r = r - alpha * v - beta * v_old
        z = z - alpha * u - beta * u_old

        beta_old = beta
        dp = inner(r, z)
        beta = np.sqrt(dp)

        # QR factorization
        c[-2] = c[-1]
        c[-1] = c[0]
        s[-2] = s[-1]
        s[-1] = s[0]
        gamma_bar = c[-1] * alpha - c[-2] * s[-1] * beta_old
        gamma = np.sqrt(gamma_bar * gamma_bar + beta * beta)
        delta = s[-1] * alpha + c[-2] * c[-1] * beta_old
        epsilon = s[-2] * beta_old

        # Givens rotation [c, -s; s, c]
        c[0] = gamma_bar / gamma
        s[0] = beta / gamma

        if k == 0:
            ceta[0] = beta1 / gamma
        else:
            ceta[0] = -(delta * ceta[-1] + epsilon * ceta[-2]) / gamma

        s_prod *= np.abs(s[0])

        # TODO norm(r) == np_?
        # np_ = s_prod / np.where(c != 0.0, np.abs(c), 1.0e-15)
        resnorms.append(_norm(r))

        if exact_solution is not None:
            errnorms.append(_norm(exact_solution - x))

        k += 1

    # move to the CG point: xc_{k+1}
    ceta_bar = ceta[0] / np.where(c[0] != 0.0, c[0], 1.0e-15)
    x += ceta_bar * w_bar

    return x if success else None, Info(
        success,
        x,
        k,
        resnorms,
        errnorms,
        num_operations=None,
        arnoldi=None,
    )
