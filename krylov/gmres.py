import numpy
import scipy.linalg

from . import utils
from .arnoldi import Arnoldi
from .linsys import _KrylovSolver


class Gmres(_KrylovSolver):
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

    def __init__(self, linear_system, ortho="mgs", **kwargs):
        """
        All parameters of :py:class:`_KrylovSolver` are valid in this solver.
        """
        self.ortho = ortho
        super(Gmres, self).__init__(linear_system, **kwargs)

    def __repr__(self):
        string = "krylov GMRES object\n"
        string += "    MMlr0 = [{}, ..., {}]\n".format(self.MMlr0[0], self.MMlr0[-1])
        string += "    MMlr0_norm = {}\n".format(self.MMlr0_norm)
        string += "    MlAMr: {} x {} matrix\n".format(*self.MlAMr.shape)
        string += "    Mlr0: [{}, ..., {}]\n".format(self.Mlr0[0], self.Mlr0[-1])
        string += "    R: {} x {} matrix\n".format(*self.R.shape)
        string += "    V: {} x {} matrix\n".format(*self.V.shape)
        string += "    flat_vecs: {}\n".format(self.flat_vecs)
        string += "    store_arnoldi: {}\n".format(self.store_arnoldi)
        string += "    ortho: {}\n".format(self.ortho)
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

    def _get_xk(self, y):
        if y is None:
            return self.x0
        k = self.arnoldi.iter
        if k > 0:
            yy = scipy.linalg.solve_triangular(self.R[:k, :k], y)
            yk = self.V[:, :k].dot(yy)
            return self.x0 + self.linear_system.Mr * yk
        return self.x0

    def _solve(self):
        # initialize Arnoldi
        self.arnoldi = Arnoldi(
            self.MlAMr,
            self.Mlr0,
            maxiter=self.maxiter,
            ortho=self.ortho,
            M=self.linear_system.M,
            Mv=self.MMlr0,
            Mv_norm=self.MMlr0_norm,
            ip_B=self.linear_system.ip_B,
        )
        # Givens rotations:
        G = []
        # QR decomposition of Hessenberg matrix via Givens and R
        self.R = numpy.zeros([self.maxiter + 1, self.maxiter], dtype=self.dtype)
        y = numpy.zeros((self.maxiter + 1, 1), dtype=self.dtype)
        # Right hand side of projected system:
        y[0] = self.MMlr0_norm

        # iterate Arnoldi
        while (
            self.resnorms[-1] > self.tol
            and self.arnoldi.iter < self.arnoldi.maxiter
            and not self.arnoldi.invariant
        ):
            k = self.iter = self.arnoldi.iter
            self.arnoldi.advance()

            # Copy new column from Arnoldi
            self.V = self.arnoldi.V
            self.R[: k + 2, k] = self.arnoldi.H[: k + 2, k]

            # Apply previous Givens rotations.
            for i in range(k):
                self.R[i : i + 2, k] = G[i].apply(self.R[i : i + 2, k])

            # Compute and apply new Givens rotation.
            G.append(utils.Givens(self.R[k : k + 2, [k]]))
            self.R[k : k + 2, k] = G[k].apply(self.R[k : k + 2, k])
            y[k : k + 2] = G[k].apply(y[k : k + 2])

            self._finalize_iteration(y[: k + 1], abs(y[k + 1, 0]))

        # compute solution if not yet done
        if self.xk is None:
            self.xk = self._get_xk(y[: self.arnoldi.iter])

    def _finalize(self):
        super(Gmres, self)._finalize()
        # store arnoldi?
        if self.store_arnoldi:
            if not isinstance(self.linear_system.M, utils.IdentityLinearOperator):
                self.V, self.H, self.P = self.arnoldi.get()
            else:
                self.V, self.H = self.arnoldi.get()

    @staticmethod
    def operations(nsteps):
        """Returns the number of operations needed for nsteps of GMRES"""
        return {
            "A": 1 + nsteps,
            "M": 2 + nsteps,
            "Ml": 2 + nsteps,
            "Mr": 1 + nsteps,
            "ip_B": 2 + nsteps + nsteps * (nsteps + 1) / 2,
            "axpy": 4 + 2 * nsteps + nsteps * (nsteps + 1) / 2,
        }


class _RestartedSolver(object):
    """Base class for restarted solvers."""

    def __init__(self, Solver, linear_system, max_restarts=0, **kwargs):
        """
        :param max_restarts: the maximum number of restarts. The maximum
          number of iterations is ``(max_restarts+1)*maxiter``.
        """
        # initial approximation will be set by first run of Solver
        self.xk = None

        # work on own copy of args in order to include proper initial guesses
        kwargs = dict(kwargs)

        # append dummy values for first run
        self.resnorms = [numpy.Inf]
        if linear_system.exact_solution is not None:
            self.errnorms = [numpy.Inf]

        # dummy value, gets reset in the first iteration
        tol = None

        restart = 0
        while restart == 0 or (self.resnorms[-1] > tol and restart <= max_restarts):
            try:
                if self.xk is not None:
                    # use last approximate solution as initial guess
                    kwargs.update({"x0": self.xk})

                # try to solve
                sol = Solver(linear_system, **kwargs)
            except utils.ConvergenceError as e:
                # use solver of exception
                sol = e.solver

            # set last approximate solution
            self.xk = sol.xk
            tol = sol.tol

            # concat resnorms / errnorms
            del self.resnorms[-1]
            self.resnorms += sol.resnorms
            if linear_system.exact_solution is not None:
                del self.errnorms[-1]
                self.errnorms += sol.errnorms

            restart += 1

        if self.resnorms[-1] > tol:
            raise utils.ConvergenceError(
                f"No convergence after {max_restarts} restarts.", self
            )


class RestartedGmres(_RestartedSolver):
    """Restarted GMRES method.

    See :py:class:`_RestartedSolver`."""

    def __init__(self, *args, **kwargs):
        super(RestartedGmres, self).__init__(Gmres, *args, **kwargs)
