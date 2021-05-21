from typing import Callable, Optional

import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike

from ._helpers import (
    Identity,
    Info,
    LinearOperator,
    Product,
    aslinearoperator,
    get_default_inner,
)
from .arnoldi import ArnoldiHouseholder, ArnoldiMGS
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
    A: LinearOperator,
    b: ArrayLike,
    M: Optional[LinearOperator] = None,
    Ml: Optional[LinearOperator] = None,
    Mr: Optional[LinearOperator] = None,
    inner: Optional[Callable] = None,
    ortho: str = "mgs",
    x0: Optional[ArrayLike] = None,
    tol: float = 1e-5,
    atol: float = 1.0e-15,
    maxiter: Optional[int] = None,
    callback: Optional[Callable] = None,
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
            yk = sum(c * v for c, v in zip(yy, arnoldi.V))
            return x0 + Mr @ yk
        return x0

    def get_residual_norm(z):
        # \\| M M_l (b-Az)\\|_{M^{-1}}
        return get_residual_and_norm(z)[2]

    def get_residual_and_norm(z):
        Ml_r = Ml @ (b - A @ z)
        M_Ml_r = M @ Ml_r
        norm2 = inner(Ml_r, M_Ml_r)

        if np.any(norm2.imag != 0.0):
            raise ValueError("inner product <x, M x> gave nonzero imaginary part")
        norm2 = norm2.real

        return M_Ml_r, Ml_r, np.sqrt(norm2)

    b = np.asarray(b)

    assert len(A.shape) == 2
    assert A.shape[0] == A.shape[1]
    assert A.shape[1] == b.shape[0]

    M = Identity() if M is None else aslinearoperator(M)
    Ml = Identity() if Ml is None else aslinearoperator(Ml)
    Mr = Identity() if Mr is None else aslinearoperator(Mr)

    inner_is_none = inner is None
    inner = get_default_inner(b.shape) if inner is None else inner

    maxiter = A.shape[0] if maxiter is None else maxiter

    if x0 is None:
        x0 = np.zeros_like(b)

    x0 = np.asarray(x0)

    # get initial residual
    M_Ml_r0, Ml_r0, M_Ml_r0_norm = get_residual_and_norm(x0)

    Ml_A_Mr = Product(Ml, A, Mr)

    resnorms = [M_Ml_r0_norm]

    if callback is not None:
        callback(x0, Ml_r0)

    # initialize Arnoldi
    if ortho.startswith("mgs"):
        num_reorthos = 1 if len(ortho) == 3 else int(ortho[3:])
        arnoldi = ArnoldiMGS(
            Ml_A_Mr,
            Ml_r0,
            num_reorthos=num_reorthos,
            M=M,
            Mv=M_Ml_r0,
            Mv_norm=M_Ml_r0_norm,
            inner=inner,
        )
    else:
        assert ortho == "householder"
        assert inner_is_none
        assert isinstance(M, Identity)
        arnoldi = ArnoldiHouseholder(Ml_A_Mr, Ml_r0)

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
    criterion = np.maximum(tol * resnorms[0], atol)
    while True:
        if np.all(resnorms[-1] <= criterion):
            # oh really?
            xk = _get_xk(yk) if xk is None else xk
            resnorms[-1] = get_residual_norm(xk)

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

        # V is used in _get_xk()
        _, h = next(arnoldi)

        print(len(h), k + 2)
        print(len(R))

        # Copy new column from Arnoldi
        print(h.shape, R.shape)
        R[: k + 2, k] = h[: k + 2]

        # Apply previous Givens rotations.
        for i in range(k):
            R[i : i + 2, k] = multi_matmul(G[i], R[i : i + 2, k])

        # Compute and apply new Givens rotation.
        g, r = givens(R[k : k + 2, k])
        G.append(g)
        R[k, k] = r
        R[k + 1, k] = 0.0
        y[k : k + 2] = multi_matmul(G[k], y[k : k + 2])

        yk = y[: k + 1]
        resnorm = np.abs(y[k + 1])
        xk = None

        # make this a numpy array to give the callback the change to override it
        resnorm = np.array(resnorm)

        if callback is not None:
            xk = _get_xk(yk) if xk is None else xk
            callback(xk, resnorm)

        # convert back to scalar
        resnorm = resnorm[()]

        resnorms.append(resnorm)
        k += 1

    # compute solution if not yet done
    if xk is None:
        xk = _get_xk(y[: arnoldi.iter])

    num_operations = {
        "A": 1 + k,
        "M": 2 + k,
        "Ml": 2 + k,
        "Mr": 1 + k,
        "inner": 2 + k + k * (k + 1) / 2,
        "axpy": 4 + 2 * k + k * (k + 1) / 2,
    }

    return xk if success else None, Info(
        success, xk, k, resnorms, num_operations=num_operations
    )
