from collections import namedtuple

import numpy

from .arnoldi import Arnoldi
from .cg import BoundCG
from .errors import ArgumentError, AssumptionError
from .givens import givens
from .linear_system import ConvergenceError, LinearSystem
from .utils import Intervals


def minres(
    A,
    b,
    M=None,
    Ml=None,
    Mr=None,
    inner_product=lambda x, y: numpy.dot(x.T.conj(), y),
    exact_solution=None,
    ortho="mgs",
    x0=None,
    tol=1e-5,
    maxiter=None,
    use_explicit_residual=False,
    store_arnoldi=False,
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

    Memory consumption is:

    * if ``store_arnoldi==False``: 3 vectors or 6 vectors if :math:`M` is used.
    * if ``store_arnoldi==True``: about maxiter+1 vectors for the Lanczos
      basis.  If :math:`M` is used the memory consumption is 2*(maxiter+1).

    **Caution:** MINRES' convergence may be delayed significantly or even
    stagnate due to round-off errors, cf. chapter 5.9 in [LieS13]_.

    In addition to the attributes described in :py:class:`_KrylovSolver`, the
    following attributes are available in an instance of this solver:

    * ``lanczos``: the Lanczos relation (an instance of :py:class:`Arnoldi`).
    """
    assert len(A.shape) == 2
    assert A.shape[0] == A.shape[1]
    assert A.shape[1] == b.shape[0]

    linear_system = LinearSystem(
        A=A, b=b, M=M, Ml=Ml, inner=inner_product, exact_solution=exact_solution,
    )

    def _get_xk(yk):
        """Compute approximate solution from initial guess and approximate solution
        of the preconditioned linear system."""
        Mr = linear_system.Mr
        Mr_yk = yk if Mr is None else Mr @ yk
        return x0 + Mr_yk

    # sanitize arguments
    if not isinstance(linear_system, LinearSystem):
        raise ArgumentError("linear_system is not an instance of LinearSystem")
    linear_system = linear_system
    N = linear_system.N
    maxiter = N if maxiter is None else maxiter

    # sanitize initial guess
    if x0 is None:
        x0 = numpy.zeros_like(linear_system.b)

    # get initial residual
    (MMlr0, Mlr0, MMlr0_norm,) = linear_system.get_residual_and_norm(x0)

    xk = None
    """Approximate solution."""

    # find common dtype
    dtype = numpy.find_common_type([linear_system.dtype, x0.dtype], [])

    # store operator (can be modified in derived classes)
    MlAMr = linear_system.MlAMr

    # TODO: reortho
    iter = 0
    """Iteration number."""

    resnorms = []
    """Relative residual norms as described for parameter ``tol``."""

    # if rhs is exactly(!) zero, return zero solution.
    if linear_system.MMlb_norm == 0:
        xk = x0 = numpy.zeros_like(b)
        resnorms.append(0.0)
    else:
        # initial relative residual norm
        resnorms.append(MMlr0_norm / linear_system.MMlb_norm)

    # compute error?
    if linear_system.exact_solution is not None:
        errnorms = []
        """Error norms."""

        err = linear_system.exact_solution - x0
        errnorms.append(numpy.sqrt(linear_system.inner(err, err)))

    # initialize Lanczos
    lanczos = Arnoldi(
        MlAMr,
        Mlr0,
        maxiter=maxiter,
        ortho=ortho,
        M=linear_system.M,
        Mv=MMlr0,
        Mv_norm=MMlr0_norm,
        inner=linear_system.inner,
    )

    # Necessary for efficient update of yk:
    W = [
        numpy.zeros(x0.shape, dtype=dtype),
        numpy.zeros(x0.shape, dtype=dtype),
    ]
    # some small helpers
    y = [MMlr0_norm, 0]  # first entry is (updated) residual
    # old Givens rotations
    G = [None, None]

    # resulting approximation is xk = x0 + Mr*yk
    yk = numpy.zeros(x0.shape, dtype=dtype)

    # iterate Lanczos
    while (
        resnorms[-1] > tol and lanczos.iter < lanczos.maxiter and not lanczos.invariant
    ):
        k = iter = lanczos.iter
        lanczos.advance()
        V, H = lanczos.V, lanczos.H

        # needed for QR-update:
        R = numpy.zeros(4)  # real because Lanczos matrix is real
        R[1] = H[k - 1, k].real
        if G[1] is not None:
            R[:2] = G[1] @ R[:2]

        # (implicit) update of QR-factorization of Lanczos matrix
        R[2:4] = [H[k, k].real, H[k + 1, k].real]
        if G[0] is not None:
            R[1:3] = G[0] @ R[1:3]
        G[1] = G[0]
        # compute new Givens rotation
        G[0] = givens(R[2:4])
        R[2] = G[0][0] @ R[2:4]  # r
        R[3] = 0.0
        y = G[0] @ y

        # update solution
        z = (V[k] - R[0] * W[0] - R[1] * W[1]) / R[2]
        W[0], W[1] = W[1], z
        yk += y[0] * z
        y = [y[1], 0]

        # finalize iteration
        resnorm = numpy.abs(y[0])

        xk = None
        # compute error norm if asked for
        if linear_system.exact_solution is not None:
            xk = _get_xk(yk) if xk is None else xk
            err = linear_system.exact_solution - xk
            errnorms.append(numpy.sqrt(linear_system.inner(err, err)))

        rkn = None
        if use_explicit_residual:
            xk = _get_xk(yk) if xk is None else xk
            rkn = linear_system.get_residual_norm(xk)
            resnorm = rkn

        resnorms.append(resnorm / linear_system.MMlb_norm)

        # compute explicit residual if asked for or if the updated residual is below the
        # tolerance or if this is the last iteration
        if resnorm / linear_system.MMlb_norm <= tol:
            # oh really?
            if not use_explicit_residual:
                xk = _get_xk(yk) if xk is None else xk
                rkn = linear_system.get_residual_norm(xk)
                resnorms[-1] = rkn / linear_system.MMlb_norm

            # # no convergence?
            # if resnorms[-1] > tol:
            #     # updated residual was below but explicit is not: warn
            #     if (
            #         not explicit_residual
            #         and resnorm / linear_system.MMlb_norm <= tol
            #     ):
            #         warnings.warn(
            #             "updated residual is below tolerance, explicit residual is NOT!"
            #             f" (upd={resnorm} <= tol={tol} < exp={resnorms[-1]})"
            #         )

        elif iter + 1 == maxiter:
            # no convergence in last iteration -> raise exception
            # (approximate solution can be obtained from exception)
            if store_arnoldi:
                if linear_system.M is not None:
                    V, H, P = lanczos.get()
                else:
                    V, H = lanczos.get()
            raise ConvergenceError(
                (
                    "No convergence in last iteration "
                    f"(maxiter: {maxiter}, residual: {resnorms[-1]})."
                ),
            )

    # compute solution if not yet done
    if xk is None:
        xk = _get_xk(yk)
    if store_arnoldi:
        if linear_system.M is not None:
            V, H, P = lanczos.get()
        else:
            V, H = lanczos.get()

    Info = namedtuple("Point", ["resnorms", "operations"])

    operations = {
        "A": 1 + iter,
        "M": 2 + iter,
        "Ml": 2 + iter,
        "Mr": 1 + iter,
        "inner": 2 + 2 * iter,
        "axpy": 4 + 8 * iter,
    }

    return xk if resnorms[-1] < tol else None, Info(resnorms, operations)


class BoundMinres:
    r"""MINRES residual norm bound.

    Computes a bound for the MINRES residual norm when the eigenvalues of the operator
    are given, see [Gre97]_.

    :param evals: an array of eigenvalues
      :math:`\lambda_1,\ldots,\lambda_N\in\mathbb{R}`. The eigenvalues will be
      sorted internally such that
      :math:`\lambda_1\leq\ldots\lambda_s<0=\lambda_{s+1}=\ldots=\lambda_{s+t-1}<\lambda_t\leq\ldots\lambda_N`
      for :math:`s,t\in\mathbb{N}` and :math:`s<t`.
    :param steps: (optional) the number of steps :math:`k` to compute the bound
      for. If steps is ``None`` (default), then :math:`k=N` is used.

    :return: array :math:`[\eta_0,\ldots,\eta_k]` with

      .. math::

         \eta_n = 2 \left(
         \frac{ \sqrt{|\lambda_1\lambda_N|} - \sqrt{|\lambda_s\lambda_t|}}
         { \sqrt{|\lambda_1\lambda_N|} + \sqrt{|\lambda_s\lambda_t|}}
         \right)^{\left[\frac{n}{2}\right]}
         \quad\text{for}\quad
         n\in\{0,\ldots,k\}

      if :math:`s>0`. If :math:`s=0`, i.e., if the eigenvalues are
      non-negative, then the result of :py:meth:`bound_cg` is returned.
    """

    def __new__(cls, evals):
        """Use BoundCG if all eigenvalues are non-negative."""
        pos = False
        if isinstance(evals, Intervals):
            if evals.min() > 0:
                pos = True
        elif (numpy.array(evals) > -1e-15).all():
            pos = True
        if pos:
            return BoundCG(evals)
        return super().__new__(cls)

    def __init__(self, evals):
        """Initialize with array/list of eigenvalues or Intervals object."""
        if isinstance(evals, Intervals):
            if evals.contains(0):
                raise AssumptionError("zero eigenvalues not allowed with intervals")
            evals = [
                val
                for val in [evals.min(), evals.max_neg(), evals.min_pos(), evals.max()]
                if val is not None
            ]

        # empty spectrum?
        if len(evals) == 0:
            raise AssumptionError("empty spectrum not allowed")

        # all evals real?
        if not numpy.isreal(evals).all():
            raise AssumptionError("non-real eigenvalues not allowed")

        # sort
        evals = numpy.sort(numpy.array(evals, dtype=numpy.float))

        # normalize and categorize evals
        evals /= numpy.max(numpy.abs(evals))
        negative = evals < -1e-15
        positive = evals > 1e-15

        lambda_1 = numpy.min(evals[negative])
        lambda_s = numpy.max(evals[negative])
        lambda_t = numpy.min(evals[positive])
        lambda_N = numpy.max(evals[positive])

        a = numpy.sqrt(numpy.abs(lambda_1 * lambda_N))
        b = numpy.sqrt(numpy.abs(lambda_s * lambda_t))

        self.base = (a - b) / (a + b)

    def eval_step(self, step):
        """Evaluate bound for given step."""
        return 2 * self.base ** numpy.floor(step / 2.0)

    def get_step(self, tol):
        """Return step at which bound falls below tolerance. """
        return 2 * numpy.log(tol / 2.0) / numpy.log(self.base)
