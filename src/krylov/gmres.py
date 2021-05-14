import warnings
from typing import Optional

import numpy as np
import scipy.linalg

from ._helpers import Identity, Info, Product, get_default_inner
from .arnoldi import Arnoldi
from .errors import ArgumentError
from .givens import givens


def multi_matmul(A, b):
    """A @ b for many A, b (i.e., A.shape == (m,n,...), y.shape == (n,...))"""
    return np.einsum("ij...,j...->i...", A, b)


def multi_solve_triangular(A, B):
    """This function calls scipy.linalg.solve_triangular for every single A. A
    vectorized version would be much better here.
    """
    A_shape = A.shape
    a = A.reshape(A.shape[0], A.shape[1], -1)
    b = B.reshape(B.shape[0], -1)
    y = []
    for k in range(a.shape[2]):
        if np.all(b[:, k] == 0.0):
            y.append(np.zeros(b[:, k].shape))
        else:
            y.append(scipy.linalg.solve_triangular(a[:, :, k], b[:, k]))
    y = np.array(y).T.reshape([A_shape[0]] + list(A_shape[2:]))
    return y


def gmres(
    A,
    b,
    M=Identity(),
    Ml=Identity(),
    Mr=Identity(),
    inner=None,
    exact_solution=None,
    ortho: str = "mgs",
    x0=None,
    tol: float = 1e-5,
    atol: float = 1.0e-15,
    maxiter: Optional[int] = None,
    use_explicit_residual: bool = False,
    return_arnoldi: bool = False,
):
    r"""Preconditioned GMRES method.

    The *preconditioned generalized minimal residual method* can be used to
    solve a system of linear algebraic equations. Let the following linear
    algebraic system be given:

    .. math::

      M M_l A M_r y = M M_l b,

    where :math:`x=M_r y`.
    The preconditioned GMRES method then computes (in exact arithmetics!)
    iterates :math:`x_k \in x_0 + M_r K_k` with
    :math:`K_k:= K_k(M M_l A M_r, r_0)` such that

    .. math::

      \|M M_l(b - A x_k)\|_{M^{-1}} =
      \min_{z \in x_0 + M_r K_k} \|M M_l (b - A z)\|_{M^{-1}}.

    The Arnoldi alorithm is used with the operator
    :math:`M M_l A M_r` and the inner product defined by
    :math:`\langle x,y \rangle_{M^{-1}} = \langle M^{-1}x,y \rangle`.
    The initial vector for Arnoldi is
    :math:`r_0 = M M_l (b - Ax_0)` - note that :math:`M_r` is not used for
    the initial vector.

    Memory consumption is about maxiter+1 vectors for the Arnoldi basis.
    If :math:`M` is used the memory consumption is 2*(maxiter+1).

    If the operator :math:`M_l A M_r` is self-adjoint then consider using
    the MINRES method :py:class:`Minres`.
    """

    def _get_xk(y):
        if y is None:
            return x0
        k = arnoldi.iter
        if k > 0:
            yy = multi_solve_triangular(R[:k, :k], y)
            # The last is always 0, so we could skip it, too
            # yk = sum(c * v for c, v in zip(yy, V[:-1]))
            yk = sum(c * v for c, v in zip(yy, V))
            return x0 + Mr @ yk
        return x0

    def get_residual_norm(z):
        # \\| M M_l (b-Az)\\|_{M^{-1}}
        return get_residual_and_norm(z)[2]

    def get_residual_and_norm(z):
        # r = M M_l ( b - A z )
        Ml_r = Ml @ (b - A @ z)
        M_Ml_r = M @ Ml_r
        norm2 = inner(Ml_r, M_Ml_r)

        if np.any(norm2.imag != 0.0):
            raise ValueError("inner product <x, M x> gave nonzero imaginary part")
        norm2 = norm2.real

        return M_Ml_r, Ml_r, np.sqrt(norm2)

    inner_is_euclidean = inner is None
    inner = get_default_inner(b.shape) if inner is None else inner

    def _norm(x):
        xx = inner(x, x)
        if np.any(xx.imag != 0.0):
            raise ValueError("inner product <x, x> gave nonzero imaginary part")
        return np.sqrt(xx.real)

    assert len(A.shape) == 2
    assert A.shape[0] == A.shape[1]
    assert A.shape[1] == b.shape[0]

    # sanitize arguments
    maxiter = A.shape[0] if maxiter is None else maxiter

    # sanitize initial guess
    if x0 is None:
        x0 = np.zeros_like(b)

    # get initial residual
    M_Ml_r0, Ml_r0, M_Ml_r0_norm = get_residual_and_norm(x0)

    Ml_A_Mr = Product(Ml, A, Mr)

    # TODO: reortho
    resnorms = [M_Ml_r0_norm]

    Ml_b = Ml @ b
    M_Ml_b = M @ Ml_b
    M_Ml_b_norm = np.sqrt(inner(Ml_b, M_Ml_b))

    # compute error?
    if exact_solution is None:
        errnorms = None
    else:
        errnorms = [_norm(exact_solution - x0)]

    # initialize Arnoldi
    arnoldi = Arnoldi(
        Ml_A_Mr,
        Ml_r0,
        maxiter=maxiter,
        ortho=ortho,
        M=M,
        Mv=M_Ml_r0,
        Mv_norm=M_Ml_r0_norm,
        inner=inner,
        inner_is_euclidean=inner_is_euclidean,
    )

    # Givens rotations:
    G = []
    # QR decomposition of Hessenberg matrix via Givens and R
    dtype = M_Ml_r0.dtype
    R = np.zeros([maxiter + 1, maxiter] + list(b.shape[1:]), dtype=dtype)
    y = np.zeros([maxiter + 1] + list(b.shape[1:]), dtype=dtype)
    # Right-hand side of projected system:
    y[0] = M_Ml_r0_norm
    yk = None
    xk = None

    # iterate Arnoldi
    k = 0
    success = False
    criterion = np.maximum(tol * M_Ml_b_norm, atol)
    while True:
        if np.all(resnorms[-1] <= criterion):
            # oh really?
            if not use_explicit_residual:
                xk = _get_xk(yk) if xk is None else xk
                rkn = get_residual_norm(xk)
                resnorms[-1] = rkn

            if np.all(resnorms[-1] <= criterion):
                success = True
                break

            # # updated residual was below but explicit is not: warn
            # warnings.warn(
            #     "updated residual is below tolerance, explicit residual is NOT!"
            #     f" (upd={resnorm} <= tol={tol} < exp={resnorms[-1]})"
            # )

        if k == maxiter:
            break

        # V is only used in _get_xk()
        try:
            V, H = next(arnoldi)
        except ArgumentError as e:
            exit(1)
            warnings.warn(e.msg)
            break

        # Copy new column from Arnoldi
        R[: k + 2, k] = H[: k + 2, k]

        # Apply previous Givens rotations.
        for i in range(k):
            R[i : i + 2, k] = multi_matmul(G[i], R[i : i + 2, k])

        # Compute and apply new Givens rotation.
        G.append(givens(R[k : k + 2, k]))
        R[k : k + 2, k] = multi_matmul(G[k], R[k : k + 2, k])
        y[k : k + 2] = multi_matmul(G[k], y[k : k + 2])

        yk = y[: k + 1]
        resnorm = np.abs(y[k + 1])
        xk = None
        # compute error norm if asked for
        if exact_solution is not None:
            xk = _get_xk(yk) if xk is None else xk
            errnorms.append(_norm(exact_solution - xk))

        rkn = None
        if use_explicit_residual:
            xk = _get_xk(yk) if xk is None else xk
            rkn = get_residual_norm(xk)
            resnorm = rkn

        resnorms.append(resnorm)
        k += 1

    # compute solution if not yet done
    if xk is None:
        xk = _get_xk(y[: arnoldi.iter])

    # store arnoldi?
    if return_arnoldi:
        V, H, P = arnoldi.get()

    num_operations = {
        "A": 1 + k,
        "M": 2 + k,
        "Ml": 2 + k,
        "Mr": 1 + k,
        "inner": 2 + k + k * (k + 1) / 2,
        "axpy": 4 + 2 * k + k * (k + 1) / 2,
    }

    return xk if success else None, Info(
        success,
        xk,
        k,
        resnorms,
        errnorms,
        num_operations,
        arnoldi=[V, H, P] if return_arnoldi else None,
    )
