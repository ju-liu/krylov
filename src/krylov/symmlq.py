"""
https://www.mcs.anl.gov/petsc/petsc-current/src/ksp/ksp/impls/symmlq/symmlq.c.html
"""
from typing import Callable, Optional

import numpy as np

from ._helpers import Info, aslinearoperator, get_default_inner, Identity


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

    ceta = 0.0
    ceta_oold = 0.0
    ceta_old = 0.0
    c = 1.0
    cold = 1.0
    s = 0.0
    sold = 0.0
    dp = 0.0
    np_ = 0.0

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

    v = r.copy()
    u = r.copy()

    ibeta = 1.0 / beta
    v *= ibeta
    u *= ibeta
    w_bar = u.copy()
    np_ = _norm(z)

    ceta = 0.0

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

            w = c * w_bar + s * u
            w_bar = -s * w_bar + c * u
            x = x + ceta * w

            ceta_oold = ceta_old
            ceta_old = ceta

        # Lanczos
        r = A @ u
        alpha = inner(u, r)
        # preconditioning
        z = M @ r

        r = r - alpha * v
        z = z - alpha * u

        r = r - beta * v_old
        z = z - beta * u_old

        beta_old = beta
        dp = inner(r, z)
        beta = np.sqrt(dp)

        # QR factorization
        coold = cold
        cold = c
        soold = sold
        sold = s
        gamma_bar = cold * alpha - coold * sold * beta_old
        gamma = np.sqrt(gamma_bar * gamma_bar + beta * beta)
        delta = sold * alpha + coold * cold * beta_old
        epsilon = soold * beta_old

        # Givens rotation [c, -s; s, c]
        c = gamma_bar / gamma
        s = beta / gamma

        if k == 0:
            ceta = beta1 / gamma
        else:
            ceta = -(delta * ceta_old + epsilon * ceta_oold) / gamma

        s_prod *= np.abs(s)
        if c == 0.0:
            np_ = s_prod * 1e16
        else:
            np_ = s_prod / np.abs(c)

        # TODO norm(r) == np_
        resnorms.append(_norm(r))

        if exact_solution is not None:
            errnorms.append(_norm(exact_solution - x))

        k += 1

    # move to the CG point: xc_{k+1}
    if c == 0.0:
        ceta_bar = ceta * 1e15
    else:
        ceta_bar = ceta / c

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
