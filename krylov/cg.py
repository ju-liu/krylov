import numpy

from ._helpers import Identity, Info, Product
from .errors import AssumptionError
from .utils import Intervals


def cg(
    A,
    b,
    M=Identity(),
    Ml=Identity(),
    inner=None,
    exact_solution=None,
    x0=None,
    tol=1e-5,
    atol=1.0e-15,
    maxiter=None,
    use_explicit_residual=False,
    return_arnoldi=False,
):
    r"""Preconditioned CG method.

    The *preconditioned conjugate gradient method* can be used to solve a
    system of linear algebraic equations where the linear operator is
    self-adjoint and positive definite. Let the following linear algebraic
    system be given:

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
        return M_Ml_r, Ml_r, inner(Ml_r, M_Ml_r)

    # numpy.dot is faster than einsum for flat vectors
    if inner is None:
        if len(b.shape) == 1:

            def inner(x, y):
                return numpy.dot(x.conj(), y)

        else:

            def inner(x, y):
                return numpy.einsum("i...,i...->...", x.conj(), y)

    assert len(A.shape) == 2
    assert A.shape[0] == A.shape[1]
    assert A.shape[1] == b.shape[0]
    N = A.shape[0]

    Ml_b = Ml @ b
    M_Ml_b = M @ Ml_b
    M_Ml_b_norm = numpy.sqrt(inner(Ml_b, M_Ml_b))
    # assert M_Ml_b_norm.shape == Ml_b.shape[1:], f"{M_Ml_b_norm.shape} != {Ml_b.shape}"

    Ml_A_Mr = Product(Ml, A)

    maxiter = N if maxiter is None else maxiter

    x0 = numpy.zeros_like(b) if x0 is None else x0

    # get initial residual
    M_Ml_r0, Ml_r0, M_Ml_r0_norm2 = get_residual_and_norm2(x0)
    M_Ml_r0_norm = numpy.sqrt(M_Ml_r0_norm2)

    dtype = M_Ml_r0.dtype

    # store operator (can be modified in derived classes)
    # TODO: reortho

    resnorms = [M_Ml_r0_norm]

    # compute error?
    if exact_solution is None:
        errnorms = None
    else:
        err = exact_solution - x0
        errnorms = [numpy.sqrt(inner(err, err))]

    # resulting approximation is xk = x0 + Mr*yk
    yk = numpy.zeros(x0.shape, dtype=dtype)
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
        V.append(M_Ml_r0 / numpy.where(M_Ml_r0_norm > 0.0, M_Ml_r0_norm, 1.0))
        if M is not None:
            P = []
            P.append(Ml_r0 / numpy.where(M_Ml_r0_norm > 0.0, M_Ml_r0_norm, 1.0))
        # H is always real-valued
        H = numpy.zeros([maxiter + 1, maxiter] + list(b.shape[1:]), dtype=float)
        alpha_old = 0  # will be set at end of iteration

    # iterate
    k = 0
    success = False
    criterion = numpy.maximum(tol * M_Ml_b_norm, atol)
    while True:
        if numpy.all(resnorms[-1] <= criterion):
            # oh really?
            if not use_explicit_residual:
                xk = _get_xk(yk) if xk is None else xk
                _, _, rkn2 = get_residual_and_norm2(xk)
                resnorms[-1] = numpy.sqrt(rkn2)

            if numpy.all(resnorms[-1] <= criterion):
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
            omega = rhos[-1] / numpy.where(rhos[-2] != 0, rhos[-2], 1.0)
            p = M_Ml_rk + omega * p
        # apply operators
        Ap = Ml_A_Mr @ p

        # compute inner product
        pAp = inner(p, Ap)
        # rho / <p, Ap>
        alpha = rhos[-1] / numpy.where(pAp != 0, pAp, 1.0)

        # check if alpha is real
        # if numpy.any(numpy.abs(alpha.imag) > 1e-12):
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

        # apply preconditioner
        M_Ml_rk = M @ Ml_rk

        # compute norm and rho_new
        M_Ml_rk_norm2 = inner(Ml_rk, M_Ml_rk)
        rhos = [rhos[-1], M_Ml_rk_norm2]

        M_Ml_rk_norm = numpy.sqrt(M_Ml_rk_norm2)
        resnorm = M_Ml_rk_norm

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
            H[k + 1, k] = numpy.sqrt(rhos[-1] / rhos[-2]) / alpha
            alpha_old = alpha

        # compute error norm if asked for
        if exact_solution is not None:
            xk = _get_xk(yk) if xk is None else xk
            err = exact_solution - xk
            errnorms.append(numpy.sqrt(inner(err, err)))

        if use_explicit_residual:
            xk = _get_xk(yk) if xk is None else xk
            _, _, resnorm2 = get_residual_and_norm2(xk)
            resnorm = numpy.sqrt(resnorm2)
            # update rho while we're at it
            rhos[-1] = resnorm2

        resnorms.append(resnorm)
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
        resnorms,
        errnorms,
        num_operations,
        arnoldi=[V, H, P] if return_arnoldi else None,
    )


class BoundCG:
    r"""CG residual norm bound.

    Computes the :math:`\kappa`-bound for the CG error :math:`A`-norm when the
    eigenvalues of the operator are given, see [LieS13]_.

    :param evals: an array of eigenvalues
      :math:`\lambda_1,\ldots,\lambda_N\in\mathbb{R}`. The eigenvalues will be
      sorted internally such that
      :math:`0=\lambda_1=\ldots=\lambda_{t-1}<\lambda_t\leq\ldots\lambda_N`
      for :math:`t\in\mathbb{N}`.
    :param steps: (optional) the number of steps :math:`k` to compute the bound
      for. If steps is ``None`` (default), then :math:`k=N` is used.

    :return: array :math:`[\eta_0,\ldots,\eta_k]` with

      .. math::

         \eta_n = 2 \left(
           \frac{\sqrt{\kappa_{\text{eff}}} - 1}
           {\sqrt{\kappa_{\text{eff}}} + 1}
         \right)^n
         \quad\text{for}\quad
         n\in\{0,\ldots,k\}

      where :math:`\kappa_{\text{eff}}=\frac{\lambda_N}{\lambda_t}`.
    """

    def __init__(self, evals, exclude_zeros=False):
        """Initialize with array/list of eigenvalues or Intervals object."""
        if isinstance(evals, Intervals):
            evals = [evals.min(), evals.max()]
            if evals[0] <= 0:
                raise AssumptionError(
                    "non-positive eigenvalues not allowed with intervals"
                )

        # empty spectrum?
        if len(evals) == 0:
            raise AssumptionError("empty spectrum not allowed")

        # all evals real?
        if not numpy.isreal(evals).all():
            raise AssumptionError("non-real eigenvalues not allowed")

        # sort
        evals = numpy.sort(numpy.array(evals, dtype=float))

        # normalize
        evals /= evals[-1]

        if exclude_zeros is False and not (evals > 1e-15).all():
            raise AssumptionError(
                "non-positive eigenvalues not allowed (use exclude_zeros?)"
            )

        # check that all are non-negative
        assert evals[0] > -1e-15

        # compute effective condition number
        kappa = 1 / numpy.min(evals[evals > 1e-15])
        self.base = (numpy.sqrt(kappa) - 1) / (numpy.sqrt(kappa) + 1)

    def eval_step(self, step):
        """Evaluate bound for given step."""
        return 2 * self.base ** step

    def get_step(self, tol):
        """Return step at which bound falls below tolerance."""
        return numpy.log(tol / 2.0) / numpy.log(self.base)
