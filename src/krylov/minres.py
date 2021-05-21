from typing import Callable, Optional

import numpy as np
from numpy.typing import ArrayLike

from ._helpers import (
    Identity,
    Info,
    LinearOperator,
    Product,
    aslinearoperator,
    get_default_inner,
)
from .arnoldi import ArnoldiLanczos
from .givens import givens


def multi_dot(a, b):
    """a.dot.b for many a, b (i.e., a.shape == (n,...), y.shape == (n,...))"""
    return np.einsum("i...,i...->...", a, b)


def multi_matmul(A, b):
    """A @ b for many A, b (i.e., A.shape == (m,n,...), y.shape == (n,...))"""
    return np.einsum("ij...,j...->i...", A, b)


def minres(
    A: LinearOperator,
    b: ArrayLike,
    M: Optional[LinearOperator] = None,
    Ml: Optional[LinearOperator] = None,
    Mr: Optional[LinearOperator] = None,
    inner: Optional[Callable] = None,
    x0: Optional[ArrayLike] = None,
    tol: float = 1e-5,
    atol: float = 1.0e-15,
    maxiter: Optional[int] = None,
    callback: Optional[Callable] = None,
):
    r"""Preconditioned MINRES method.

    The *preconditioned minimal residual method* can be used to solve a
    system of linear algebraic equations where the linear operator is
    self-adjoint. Let the following linear algebraic
    system be given:

    .. math::

      M M_l A M_r y = M M_l b,

    where :math:`x=M_r y` and :math:`M_l A M_r` is self-adjoint with respect
    to the inner product
    :math:`\langle \cdot,\cdot \rangle` defined by ``inner_product``.
    The preconditioned MINRES method then computes (in exact arithmetics!)
    iterates :math:`x_k \in x_0 + M_r K_k` with
    :math:`K_k:= K_k(M M_l A M_r, r_0)` such that

    .. math::

      \|M M_l(b - A x_k)\|_{M^{-1}} =
      \min_{z \in x_0 + M_r K_k} \|M M_l (b - A z)\|_{M^{-1}}.

    The Lanczos alorithm is used with the operator
    :math:`M M_l A M_r` and the inner product defined by
    :math:`\langle x,y \rangle_{M^{-1}} = \langle M^{-1}x,y \rangle`.
    The initial vector for Lanczos is
    :math:`r_0 = M M_l (b - Ax_0)` - note that :math:`M_r` is not used for
    the initial vector.

    Memory consumption is 3 vectors or 6 vectors if :math:`M` is used.

    **Caution:** MINRES' convergence may be delayed significantly or even
    stagnate due to round-off errors, cf. chapter 5.9 in [LieS13]_.

    In addition to the attributes described in :py:class:`_KrylovSolver`, the
    following attributes are available in an instance of this solver:

    * ``lanczos``: the Lanczos relation (an instance of :py:class:`Arnoldi`).
    """
    b = np.asarray(b)

    assert len(A.shape) == 2
    assert A.shape[0] == A.shape[1]
    assert A.shape[1] == b.shape[0]

    M = Identity() if M is None else aslinearoperator(M)
    Ml = Identity() if Ml is None else aslinearoperator(Ml)
    Mr = Identity() if Mr is None else aslinearoperator(Mr)

    inner = get_default_inner(b.shape) if inner is None else inner

    N = A.shape[0]

    def _get_x(y):
        """Compute approximate solution from initial guess and approximate solution
        of the preconditioned linear system."""
        return x0 + Mr @ y

    def _norm(x):
        xx = inner(x, M @ x)
        if np.any(xx.imag != 0.0):
            raise ValueError("inner product <x, x> gave nonzero imaginary part")
        return np.sqrt(xx.real)

    def get_residual_norm(z):
        # r = Ml (b - Az)
        # ||r|| = <r, M r>
        #
        # \\| M M_l (b-Az)\\|_{M^{-1}}
        Ml_r = Ml @ (b - A @ z)
        return _norm(Ml_r)

    maxiter = N if maxiter is None else maxiter

    # sanitize initial guess
    if x0 is None:
        x0 = np.zeros_like(b)

    # get initial residual
    r = b - A @ x0
    Ml_r = Ml @ r
    M_Ml_r = M @ Ml_r
    alpha = inner(Ml_r, M @ Ml_r)
    if np.any(alpha.imag != 0.0):
        raise ValueError("inner product <x, x> gave nonzero imaginary part")
    M_Ml_r_norm = np.sqrt(alpha.real)

    dtype = M_Ml_r.dtype
    # dtype = np.find_common_type(
    #     [A.dtype, x.dtype, b.dtype, M.dtype, Ml.dtype, Mr.dtype], []
    # )

    k = 0

    Ml_A_Mr = Product(Ml, A, Mr)

    # initialize Lanczos
    arnoldi = ArnoldiLanczos(
        Ml_A_Mr, Ml_r, M=M, Mv=M_Ml_r, Mv_norm=M_Ml_r_norm, inner=inner
    )

    # Necessary for efficient update of yk:
    W = [
        np.zeros(b.shape, dtype=dtype),
        np.zeros(b.shape, dtype=dtype),
    ]
    # some small helpers
    y = np.array([M_Ml_r_norm, np.zeros_like(M_Ml_r_norm)])
    # old Givens rotations
    G = [None, None]

    # resulting approximation is xk = x0 + Mr*yk
    yk = np.zeros(b.shape, dtype=dtype)
    xk = None

    # make this a numpy array to give the callback the change to override it
    resnorm = np.array(M_Ml_r_norm)

    if callback is not None:
        callback(x0, resnorm)

    resnorms = [resnorm[()]]

    k = 0
    success = False
    criterion = np.maximum(tol * resnorms[0], atol)
    while True:
        if np.all(resnorms[-1] <= criterion):
            # oh really?
            xk = _get_x(yk) if xk is None else xk
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

        # TODO V[k] vs v
        v = arnoldi.v
        _, h, _ = next(arnoldi)

        assert np.all(np.abs(h.imag)) < 1.0e-14
        h = h.real

        # needed for QR-update:
        # R is real because Lanczos matrix is real
        R = np.zeros([4] + list(b.shape[1:]), dtype=float)
        R[1] = h[0]
        if G[1] is not None:
            # apply givens rotation
            # R0 = G[1][0][1] * R[1]
            # R1 = G[1][1][1] * R[1]
            # R[0], R[1] = R0, R1
            R[:2] = multi_matmul(G[1], R[:2])

        # (implicit) update of QR-factorization of Lanczos matrix
        R[2] = h[1]
        R[3] = h[2]
        if G[0] is not None:
            R[1:3] = multi_matmul(G[0], R[1:3])
        G[1] = G[0]
        # compute new Givens rotation
        G[0], r = givens(R[2:4])
        R[2] = r
        R[3] = 0.0
        # TODO second component of y is always 0
        y = multi_matmul(G[0], y)

        # update solution
        # The following two vector additions take the longest in this function
        z = (v - R[0] * W[0] - R[1] * W[1]) / np.where(R[2] != 0.0, R[2], 1.0)
        W[0], W[1] = W[1], z
        yk += y[0] * z
        xk = None

        y = np.array([y[1], np.zeros_like(y[1])])

        # finalize iteration
        # make this a numpy array to give the callback the change to override it
        resnorm = np.array(np.abs(y[0]))

        if callback is not None:
            xk = _get_x(yk) if xk is None else xk
            callback(xk, resnorm)

        resnorms.append(resnorm[()])

        k += 1

    # compute solution if not yet done
    if xk is None:
        xk = _get_x(yk)

    num_operations = {
        "A": 1 + k,
        "M": 2 + k,
        "Ml": 2 + k,
        "Mr": 1 + k,
        "inner": 2 + 2 * k,
        "axpy": 4 + 8 * k,
    }

    return xk if success else None, Info(
        success, xk, k, resnorms, num_operations=num_operations
    )
