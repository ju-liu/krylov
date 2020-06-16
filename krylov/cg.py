import warnings

import numpy

from . import utils
from .linsys import _KrylovSolver


class Cg(_KrylovSolver):
    r"""Preconditioned CG method.

    The *preconditioned conjugate gradient method* can be used to solve a
    system of linear algebraic equations where the linear operator is
    self-adjoint and positive definite. Let the following linear algebraic
    system be given:

    .. math::

      M M_l A M_r y = M M_l b,

    where :math:`x=M_r y` and :math:`M_l A M_r` is self-adjoint and
    positive definite with respect to the inner product
    :math:`\langle \cdot,\cdot \rangle` defined by ``ip_B``.
    The preconditioned CG method then computes (in exact arithmetics!)
    iterates :math:`x_k \in x_0 + M_r K_k` with
    :math:`K_k:= K_k(M M_l A M_r, r_0)` such that

    .. math::

      \|x - x_k\|_A = \min_{z \in x_0 + M_r K_k} \|x - z\|_A.

    The Lanczos alorithm is used with the operator
    :math:`M M_l A M_r` and the inner product defined by
    :math:`\langle x,y \rangle_{M^{-1}} = \langle M^{-1}x,y \rangle`.
    The initial vector for Lanczos is
    :math:`r_0 = M M_l (b - Ax_0)` - note that :math:`M_r` is not used for
    the initial vector.

    Memory consumption is:

    * if ``store_arnoldi==False``: 3 vectors or 6 vectors if :math:`M` is used.
    * if ``store_arnoldi==True``: about maxiter+1 vectors for the Lanczos
      basis. If :math:`M` is used the memory consumption is 2*(maxiter+1).

    **Caution:** CG's convergence may be delayed significantly due to round-off
    errors, cf. chapter 5.9 in [LieS13]_.

    """

    def __init__(self, linear_system, **kwargs):
        """
        All parameters of :py:class:`_KrylovSolver` are valid in this solver.
        Note the restrictions on ``M``, ``Ml``, ``A``, ``Mr`` and ``ip_B``
        above.
        """
        if not linear_system.self_adjoint or not linear_system.positive_definite:
            warnings.warn(
                "Cg applied to a non-self-adjoint or non-definite "
                "linear system. Consider using Minres or Gmres."
            )
        super(Cg, self).__init__(linear_system, **kwargs)

    def __repr__(self):
        string = "krylov CG object\n"
        string += "    MMlr0 = [{}, ..., {}]\n".format(self.MMlr0[0], self.MMlr0[-1])
        string += "    MMlr0_norm = {}\n".format(self.MMlr0_norm)
        string += "    MlAMr: {} x {} matrix\n".format(*self.MlAMr.shape)
        string += "    Mlr0: [{}, ..., {}]\n".format(self.Mlr0[0], self.Mlr0[-1])
        string += "    flat_vecs: {}\n".format(self.flat_vecs)
        string += "    store_arnoldi: {}\n".format(self.store_arnoldi)
        string += "    tol: {}\n".format(self.tol)
        string += "    maxiter: {}\n".format(self.maxiter)
        string += "    iter: {}\n".format(self.iter)
        string += "    explicit residual: {}\n".format(self.explicit_residual)
        string += "    resnorms: [{}, ..., {}]\n".format(
            self.resnorms[0], self.resnorms[-1]
        )
        string += "    x0: [{}, ..., {}]\n".format(self.x0[0], self.x0[-1])
        string += "    xk: [{}, ..., {}]".format(self.xk[0], self.xk[-1])
        return string

    def _solve(self):
        N = self.linear_system.N

        # resulting approximation is xk = x0 + Mr*yk
        yk = numpy.zeros((N, 1), dtype=self.dtype)

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
            if not isinstance(self.linear_system.M, utils.IdentityLinearOperator):
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
            Ap = self.MlAMr * p

            # compute inner product
            alpha = rhos[-1] / utils.inner(p, Ap, ip_B=self.linear_system.ip_B)[0, 0]

            # check if alpha is real
            if abs(alpha.imag) > 1e-12:
                warnings.warn(
                    f"Iter {k}: abs(alpha.imag) = {abs(alpha.imag)} > 1e-12. "
                    "Is your operator self-adjoint in the provided inner "
                    "product?"
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
            self.MMlrk = self.linear_system.M * self.Mlrk

            # compute norm and rho_new
            MMlrk_norm = utils.norm(self.Mlrk, self.MMlrk, ip_B=self.linear_system.ip_B)
            rhos.append(MMlrk_norm ** 2)

            # compute Lanczos vector + new subdiagonal element
            if self.store_arnoldi:
                self.V[:, [k + 1]] = (-1) ** (k + 1) * self.MMlrk / MMlrk_norm
                if not isinstance(self.linear_system.M, utils.IdentityLinearOperator):
                    self.P[:, [k + 1]] = (-1) ** (k + 1) * self.Mlrk / MMlrk_norm
                self.H[k + 1, k] = numpy.sqrt(rhos[-1] / rhos[-2]) / alpha
                alpha_old = alpha

            # compute norms
            # if explicit_residual: compute Mlrk and MMlrk here
            # (with preconditioner application)
            rkn = self._finalize_iteration(yk, MMlrk_norm)

            # update rho_new if it was updated in _compute_norms
            if rkn is not None:
                # new rho
                rhos[-1] = rkn ** 2

            self.iter += 1

        # compute solution if not yet done
        if self.xk is None:
            self.xk = self._get_xk(yk)

    def _finalize(self):
        super(Cg, self)._finalize()
        # trim Lanczos relation
        if self.store_arnoldi:
            self.V = self.V[:, : self.iter + 1]
            self.H = self.H[: self.iter + 1, : self.iter]

    @staticmethod
    def operations(nsteps):
        """Returns the number of operations needed for nsteps of CG"""
        return {
            "A": 1 + nsteps,
            "M": 2 + nsteps,
            "Ml": 2 + nsteps,
            "Mr": 1 + nsteps,
            "ip_B": 2 + 2 * nsteps,
            "axpy": 2 + 2 * nsteps,
        }
