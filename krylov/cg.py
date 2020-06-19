import warnings

import numpy

from .errors import AssumptionError, ConvergenceError
from .linear_system import LinearSystem, _KrylovSolver
from .utils import Intervals


class _Cg(_KrylovSolver):
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

    * if ``store_arnoldi==False``: 3 vectors or 6 vectors if :math:`M` is used.
    * if ``store_arnoldi==True``: about maxiter+1 vectors for the Lanczos
      basis. If :math:`M` is used the memory consumption is 2*(maxiter+1).

    **Caution:** CG's convergence may be delayed significantly due to round-off errors,
    cf. chapter 5.9 in [LieS13]_.
    """

    def __init__(self, linear_system, **kwargs):
        """
        All parameters of :py:class:`_KrylovSolver` are valid in this solver.
        Note the restrictions on ``M``, ``Ml``, ``A``, ``Mr`` and ``inner``
        above.
        """
        super().__init__(linear_system, **kwargs)

    def _get_xk(self, yk):
        """Compute approximate solution from initial guess and approximate solution of
        the preconditioned linear system."""
        Mr = self.linear_system.Mr
        Mr_yk = yk if Mr is None else Mr @ yk
        return self.x0 + Mr_yk

    def _solve(self):
        N = self.linear_system.N
        M = self.linear_system.M

        # resulting approximation is xk = x0 + Mr*yk
        yk = numpy.zeros(self.x0.shape, dtype=self.dtype)

        # square of the old residual norm
        self.rhos = rhos = [self.MMlr0_norm ** 2]

        # will be updated by _compute_rkn if explicit_residual is True
        self.Mlrk = self.Mlr0.copy()
        self.MMlrk = self.MMlr0.copy()

        # search direction
        p = self.MMlrk.copy()
        self.iter = 0

        # store Lanczos vectors + matrix?
        if self.store_arnoldi:
            self.V = numpy.zeros((N, self.maxiter + 1), dtype=self.dtype)
            if self.MMlr0_norm > 0:
                self.V[:, [0]] = self.MMlr0 / self.MMlr0_norm
            if M is not None:
                self.P = numpy.zeros((N, self.maxiter + 1), dtype=self.dtype)
                if self.MMlr0_norm > 0:
                    self.P[:, [0]] = self.Mlr0 / self.MMlr0_norm
            self.H = numpy.zeros((self.maxiter + 1, self.maxiter))  # real
            alpha_old = 0  # will be set at end of iteration

        # iterate
        while self.resnorms[-1] > self.tol and self.iter < self.maxiter:
            k = self.iter
            if k > 0:
                # update the search direction
                p = self.MMlrk + rhos[-1] / rhos[-2] * p
                if self.store_arnoldi:
                    omega = rhos[-1] / rhos[-2]
            # apply operators
            Ap = self.MlAMr @ p

            # compute inner product
            alpha = rhos[-1] / self.linear_system.inner(p, Ap)

            # check if alpha is real
            if abs(alpha.imag) > 1e-12:
                warnings.warn(
                    f"Iter {k}: abs(alpha.imag) = {abs(alpha.imag)} > 1e-12. "
                    "Is your operator self-adjoint in the provided inner product?"
                )
            alpha = alpha.real

            # compute new diagonal element
            if self.store_arnoldi:
                if k > 0:
                    # copy superdiagonal from last iteration
                    self.H[k - 1, k] = self.H[k, k - 1]
                    self.H[k, k] = (1.0 + alpha * omega / alpha_old) / alpha
                else:
                    self.H[k, k] = 1.0 / alpha

            # update solution
            yk += alpha * p

            # update residual
            self.Mlrk -= alpha * Ap

            # apply preconditioner
            self.MMlrk = self.Mlrk if M is None else M @ self.Mlrk

            # compute norm and rho_new
            MMlrk_norm = numpy.sqrt(self.linear_system.inner(self.Mlrk, self.MMlrk))
            rhos.append(MMlrk_norm ** 2)

            resnorm = MMlrk_norm

            # compute Lanczos vector + new subdiagonal element
            if self.store_arnoldi:
                self.V[:, [k + 1]] = (-1) ** (k + 1) * self.MMlrk / MMlrk_norm
                if M is not None:
                    self.P[:, [k + 1]] = (-1) ** (k + 1) * self.Mlrk / MMlrk_norm
                self.H[k + 1, k] = numpy.sqrt(rhos[-1] / rhos[-2]) / alpha
                alpha_old = alpha

            self.xk = None
            # compute error norm if asked for
            if self.linear_system.exact_solution is not None:
                self.xk = self._get_xk(yk) if self.xk is None else self.xk
                err = self.linear_system.exact_solution - self.xk
                self.errnorms.append(numpy.sqrt(self.linear_system.inner(err, err)))

            if self.explicit_residual:
                self.xk = self._get_xk(yk) if self.xk is None else self.xk
                resnorm = self.linear_system.get_residual_norm(self.xk)
                # update rho while we're at it
                rhos[-1] = resnorm ** 2

            self.resnorms.append(resnorm / self.linear_system.MMlb_norm)

            # compute explicit residual if asked for or if the updated residual is below the
            # tolerance or if this is the last iteration
            if resnorm / self.linear_system.MMlb_norm <= self.tol:
                # oh really?
                if not self.explicit_residual:
                    self.xk = self._get_xk(yk) if self.xk is None else self.xk
                    rkn = self.linear_system.get_residual_norm(self.xk)
                    self.resnorms[-1] = rkn / self.linear_system.MMlb_norm

                    if self.resnorms[-1] / self.linear_system.MMlb_norm <= self.tol:
                        break

                # # no convergence?
                # if self.resnorms[-1] > self.tol:
                #     # updated residual was below but explicit is not: warn
                #     if (
                #         not self.explicit_residual
                #         and resnorm / self.linear_system.MMlb_norm <= self.tol
                #     ):
                #         warnings.warn(
                #             "updated residual is below tolerance, explicit residual is NOT!"
                #             f" (upd={resnorm} <= tol={self.tol} < exp={self.resnorms[-1]})"
                #         )

            if self.iter + 1 == self.maxiter:
                # no convergence in last iteration -> raise exception
                # (approximate solution can be obtained from exception)
                self._finalize()
                raise ConvergenceError(
                    (
                        "No convergence in last iteration "
                        f"(maxiter: {self.maxiter}, residual: {self.resnorms[-1]})."
                    ),
                    self,
                )

            self.iter += 1

        # compute solution if not yet done
        self.xk = self._get_xk(yk) if self.xk is None else self.xk

    def _finalize(self):
        super()._finalize()
        # trim Lanczos relation
        if self.store_arnoldi:
            self.V = self.V[:, : self.iter + 1]
            self.H = self.H[: self.iter + 1, : self.iter]

    @staticmethod
    def operations(nsteps):
        """Returns the number of operations needed for n steps of CG"""
        return {
            "A": 1 + nsteps,
            "M": 2 + nsteps,
            "Ml": 2 + nsteps,
            "Mr": 1 + nsteps,
            "inner": 2 + 2 * nsteps,
            "axpy": 2 + 2 * nsteps,
        }


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
        evals = numpy.sort(numpy.array(evals, dtype=numpy.float))

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


def cg(
    A,
    b,
    M=None,
    Ml=None,
    inner_product=lambda x, y: numpy.dot(x.T.conj(), y),
    exact_solution=None,
    x0=None,
    tol=1e-5,
    maxiter=None,
    use_explicit_residual=False,
    store_arnoldi=False,
):
    assert len(A.shape) == 2
    assert A.shape[0] == A.shape[1]
    assert A.shape[1] == b.shape[0]

    linear_system = LinearSystem(
        A=A, b=b, M=M, Ml=Ml, inner=inner_product, exact_solution=exact_solution,
    )
    out = _Cg(
        linear_system,
        x0=x0,
        tol=tol,
        maxiter=maxiter,
        explicit_residual=use_explicit_residual,
        store_arnoldi=store_arnoldi,
    )
    return out.xk if out.resnorms[-1] < out.tol else None, out
