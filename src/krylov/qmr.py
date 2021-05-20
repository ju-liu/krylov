"""
Roland W. Freund, Noël M. Nachtigal,
QMR: a quasi-minimal residual method for non-Hermitian linear systems,
Numerische Mathematik volume 60, pages315–339 (1991),
<https://doi.org/10.1007/BF01385726>.

https://www.netlib.org/templates/templates.pdf
"""
from typing import Callable, Optional

import numpy as np
from numpy.typing import ArrayLike

from ._helpers import (
    Identity,
    Info,
    RLinearOperator,
    aslinearoperator,
    get_default_inner,
)


def qmr(
    A: RLinearOperator,
    b: ArrayLike,
    Ml: Optional[RLinearOperator] = None,
    Mr: Optional[RLinearOperator] = None,
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

    Ml = Identity() if Ml is None else aslinearoperator(Ml)
    Mr = Identity() if Mr is None else aslinearoperator(Mr)

    if x0 is None:
        x = np.zeros_like(b)
        r = b.copy()
    else:
        x = np.array(x0)
        r = b - A @ x0

    inner = get_default_inner(b.shape) if inner is None else inner

    def _norm(x):
        xx = inner(x, Ml @ x)
        if np.any(xx.imag != 0.0):
            raise ValueError("inner product <x, x> gave nonzero imaginary part")
        return np.sqrt(xx.real)

    if callback is not None:
        callback(x, r)

    resnorms = [_norm(r)]

    v_ = r.copy()
    y = Ml @ v_

    rho = _norm(y)

    # arbitrary choice
    w_ = r.copy()
    z = Mr.rmatvec(w_)

    xi = _norm(z)
    gamma = 1.0
    eta = -1.0
    theta = 1.0
    epsilon = 1.0

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

        v = v_ / np.where(rho != 0.0, rho, 1.0)
        y /= np.where(rho != 0.0, rho, 1.0)

        w = w_ / np.where(xi != 0.0, xi, 1.0)
        z /= np.where(xi != 0.0, xi, 1.0)

        delta = inner(z, y)

        y_ = Mr @ y
        z_ = Ml.rmatvec(z)

        if k == 0:
            p = y_.copy()
            q = z_.copy()
        else:
            delta_epilon = delta / np.where(epsilon != 0.0, epsilon, 1.0)
            p = y_ - (xi * delta_epilon) * p
            q = z_ - (rho * delta_epilon) * q

        p_ = A @ p
        epsilon = inner(q, p_)
        beta = epsilon / np.where(delta != 0.0, delta, 1.0)

        v_ = p_ - beta * v

        y = Ml @ v_
        rho_old = rho
        rho = _norm(y)

        w_ = A.rmatvec(q) - beta * w

        z = Mr.rmatvec(w_)

        xi = _norm(z)
        gamma_old = gamma
        theta_old = theta

        gamma_old_abs_beta = gamma_old * np.abs(beta)

        theta = rho / np.where(gamma_old_abs_beta != 0.0, gamma_old_abs_beta, 1.0)
        gamma = 1 / np.sqrt(1 + theta ** 2)
        beta_gamma_old2 = beta * gamma_old ** 2
        eta = (
            -eta
            * rho_old
            * gamma ** 2
            / np.where(beta_gamma_old2 != 0.0, beta_gamma_old2, 1.0)
        )

        if k == 0:
            d = eta * p
            s = eta * p_
        else:
            d = eta * p + (theta_old * gamma) ** 2 * d
            s = eta * p_ + (theta_old * gamma) ** 2 * s

        x += d
        r -= s

        if callback is not None:
            callback(x, r)

        resnorms.append(_norm(r))

        k += 1

    return x if success else None, Info(success, x, k, resnorms)
