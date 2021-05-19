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


def cg(
    A: LinearOperator,
    b: ArrayLike,
    M: Optional[LinearOperator] = None,
    Ml: Optional[LinearOperator] = None,
    inner: Optional[Callable] = None,
    x0: Optional[ArrayLike] = None,
    tol: float = 1e-5,
    atol: float = 1.0e-15,
    maxiter: Optional[int] = None,
    return_arnoldi: bool = False,
    callback: Optional[Callable] = None,
):
    r"""Preconditioned CG method.

    The *preconditioned conjugate gradient method* can be used to solve a system of
    linear algebraic equations where the linear operator is self-adjoint and positive
    definite. Let the following linear algebraic system be given:

    .. math::

      M M_l A M_r y = M M_l b,

    where :math:`x=M_r y` and :math:`M_l A M_r` is self-adjoint and positive definite
    with respect to the inner product :math:`\langle \cdot,\cdot \rangle` defined by
    ``inner``.  The preconditioned CG method then computes (in exact arithmetics!)
    iterates :math:`x_k \in x_0 + M_r K_k` with :math:`K_k:= K_k(M M_l A M_r, r_0)` such
    that

    .. math::

      \|x - x_k\|_A = \min_{z \in x_0 + M_r K_k} \|x - z\|_A.

    The Lanczos algorithm is used with the operator :math:`M M_l A M_r` and the inner
    product defined by :math:`\langle x,y \rangle_{M^{-1}} = \langle M^{-1}x,y \rangle`.
    The initial vector for Lanczos is :math:`r_0 = M M_l (b - Ax_0)` - note that
    :math:`M_r` is not used for the initial vector.

    Memory consumption is:

    * if ``return_arnoldi==False``: 3 vectors or 6 vectors if :math:`M` is used.
    * if ``return_arnoldi==True``: about iter+1 vectors for the Lanczos
      basis. If :math:`M` is used the memory consumption is 2*(iter+1).

    **Caution:** CG's convergence may be delayed significantly due to round-off errors,
    cf. chapter 5.9 in [LieS13]_.
    """

    def _get_xk(yk):
        """Compute approximate solution from initial guess and approximate solution of
        the preconditioned linear system."""
        # Mr_yk = yk if Mr is None else Mr @ yk
        Mr_yk = yk
        return x0 + Mr_yk

    def get_residual_and_norm2(z):
        r"""Compute residual.

        For a given :math:`z\in\mathbb{C}^N`, the residual

        .. math::

          r = M M_l ( b - A z )

        :param z: approximate solution and  the absolute residual norm

        .. math::

          \\|M M_l (b-Az)\\|_{M^{-1}}
        """
        r = b - A @ z
        Ml_r = Ml @ r
        M_Ml_r = M @ Ml_r

        norm2 = inner(Ml_r, M_Ml_r)
        if np.any(norm2.imag != 0.0):
            raise ValueError("inner product <x, M x> gave nonzero imaginary part")
        norm2 = norm2.real

        return M_Ml_r, Ml_r, norm2

    b = np.asarray(b)

    assert len(A.shape) == 2
    assert A.shape[0] == A.shape[1]
    assert A.shape[1] == b.shape[0]
    N = A.shape[0]

    inner = get_default_inner(b.shape) if inner is None else inner

    M = Identity() if M is None else aslinearoperator(M)
    Ml = Identity() if Ml is None else aslinearoperator(Ml)

    Ml_A_Mr = Product(Ml, A)

    maxiter = N if maxiter is None else maxiter

    x0 = np.zeros_like(b) if x0 is None else x0

    # get initial residual
    M_Ml_r0, Ml_r0, M_Ml_r0_norm2 = get_residual_and_norm2(x0)
    M_Ml_r0_norm = np.sqrt(M_Ml_r0_norm2)

    if callback is not None:
        callback(x0, Ml_r0)

    # TODO: reortho

    resnorms = [M_Ml_r0_norm]

    # resulting approximation is xk = x0 + Mr*yk
    yk = np.zeros(x0.shape, dtype=M_Ml_r0.dtype)
    xk = None

    # square of the old residual norm
    rhos = [None, M_Ml_r0_norm2]

    # will be updated by _compute_rkn if explicit_residual is True
    Ml_rk = Ml_r0.copy()
    M_Ml_rk = M_Ml_r0.copy()

    # search direction
    p = M_Ml_rk.copy()

    # store Lanczos vectors + matrix?
    if return_arnoldi:
        V = []
        V.append(M_Ml_r0 / np.where(M_Ml_r0_norm > 0.0, M_Ml_r0_norm, 1.0))
        if M is not None:
            P = []
            P.append(Ml_r0 / np.where(M_Ml_r0_norm > 0.0, M_Ml_r0_norm, 1.0))
        # H is always real-valued
        H = np.zeros([maxiter + 1, maxiter] + list(b.shape[1:]), dtype=float)
        alpha_old = 0  # will be set at end of iteration

    # iterate
    k = 0
    success = False
    criterion = np.maximum(tol * resnorms[0], atol)
    while True:
        if np.all(resnorms[-1] <= criterion):
            # oh really?
            xk = _get_xk(yk) if xk is None else xk
            _, _, rkn2 = get_residual_and_norm2(xk)
            resnorms[-1] = np.sqrt(rkn2)

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

        if k > 0:
            # update the search direction
            omega = rhos[-1] / np.where(rhos[-2] != 0, rhos[-2], 1.0)
            p = M_Ml_rk + omega * p
        # apply operators
        Ap = Ml_A_Mr @ p

        # compute inner product
        pAp = inner(p, Ap)
        # rho / <p, Ap>
        alpha = rhos[-1] / np.where(pAp != 0, pAp, 1.0)

        # check if alpha is real
        # if np.any(np.abs(alpha.imag) > 1e-12):
        #     warnings.warn(
        #         f"Iter {k}: abs(alpha.imag) = {abs(alpha.imag)} > 1e-12. "
        #         "Is your operator adjoint in the provided inner product?"
        #     )
        # alpha = alpha.real

        # update solution
        yk += alpha * p
        xk = None

        # update residual
        Ml_rk -= alpha * Ap

        if callback is not None:
            xk = _get_xk(yk) if xk is None else xk
            callback(xk, Ml_rk)

        # apply preconditioner
        M_Ml_rk = M @ Ml_rk
        # compute norm and rho_new
        M_Ml_rk_norm2 = inner(Ml_rk, M_Ml_rk)
        if np.any(M_Ml_rk_norm2.imag != 0.0):
            raise ValueError("inner product <r, M r> gave nonzero imaginary part")
        M_Ml_rk_norm2 = M_Ml_rk_norm2.real

        rhos = [rhos[-1], M_Ml_rk_norm2]

        M_Ml_rk_norm = np.sqrt(M_Ml_rk_norm2)
        resnorms.append(M_Ml_rk_norm)

        # compute Lanczos vector + new subdiagonal element
        if return_arnoldi:
            V.append((-1) ** (k + 1) * M_Ml_rk / M_Ml_rk_norm)
            if M is not None:
                P.append((-1) ** (k + 1) * Ml_rk / M_Ml_rk_norm)
            #
            # compute new diagonal element
            H[k, k] = 1.0 / alpha
            if k > 0:
                # copy superdiagonal from last iteration
                H[k - 1, k] = H[k, k - 1]
                H[k, k] += omega / alpha_old
            H[k + 1, k] = np.sqrt(rhos[-1] / rhos[-2]) / alpha
            alpha_old = alpha

        k += 1

    # compute solution if not yet done
    xk = _get_xk(yk) if xk is None else xk

    # trim Lanczos relation
    if return_arnoldi:
        H = H[: k + 1, :k]

    num_operations = {
        "A": 1 + k,
        "M": 2 + k,
        "Ml": 2 + k,
        "Mr": 1 + k,
        "inner": 2 + 2 * k,
        "axpy": 2 + 2 * k,
    }

    return xk if success else None, Info(
        success,
        xk,
        k,
        resnorms,
        num_operations=num_operations,
        arnoldi=[V, H, P] if return_arnoldi else None,
    )
