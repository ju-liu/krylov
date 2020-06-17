import warnings

import numpy

from . import utils
from .arnoldi import Arnoldi
from .cg import BoundCG
from .errors import AssumptionError
from .linsys import _KrylovSolver
from .utils import Intervals


class Minres(_KrylovSolver):
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

    def __init__(self, linear_system, ortho="lanczos", **kwargs):
        """
        All parameters of :py:class:`_KrylovSolver` are valid in this solver.
        Note the restrictions on ``M``, ``Ml``, ``A``, ``Mr`` and ``ip_B``
        above.
        """
        if not linear_system.self_adjoint:
            warnings.warn(
                "Minres applied to a non-self-adjoint "
                "linear system. Consider using Gmres."
            )
        self.ortho = ortho
        super(Minres, self).__init__(linear_system, **kwargs)

    def __repr__(self):
        string = "krylov MINRES object\n"
        string += "    MMlr0 = [{}, ..., {}]\n".format(self.MMlr0[0], self.MMlr0[-1])
        string += "    MMlr0_norm = {}\n".format(self.MMlr0_norm)
        string += "    MlAMr: {} x {} matrix\n".format(*self.MlAMr.shape)
        string += "    Mlr0: [{}, ..., {}]\n".format(self.Mlr0[0], self.Mlr0[-1])
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

    def _solve(self):
        N = self.linear_system.N

        # initialize Lanczos
        self.lanczos = Arnoldi(
            self.MlAMr,
            self.Mlr0,
            maxiter=self.maxiter,
            ortho=self.ortho,
            M=self.linear_system.M,
            Mv=self.MMlr0,
            Mv_norm=self.MMlr0_norm,
            ip_B=self.linear_system.ip_B,
        )

        # Necessary for efficient update of yk:
        W = numpy.c_[numpy.zeros(N, dtype=self.dtype), numpy.zeros(N)]
        # some small helpers
        y = [self.MMlr0_norm, 0]  # first entry is (updated) residual
        G2 = None  # old givens rotation
        G1 = None  # even older givens rotation ;)

        # resulting approximation is xk = x0 + Mr*yk
        yk = numpy.zeros((N, 1), dtype=self.dtype)

        # iterate Lanczos
        while (
            self.resnorms[-1] > self.tol
            and self.lanczos.iter < self.lanczos.maxiter
            and not self.lanczos.invariant
        ):
            k = self.iter = self.lanczos.iter
            self.lanczos.advance()
            V, H = self.lanczos.V, self.lanczos.H

            # needed for QR-update:
            R = numpy.zeros((4, 1))  # real because Lanczos matrix is real
            R[1] = H[k - 1, k].real
            if G1 is not None:
                R[:2] = G1.apply(R[:2])

            # (implicit) update of QR-factorization of Lanczos matrix
            R[2:4, 0] = [H[k, k].real, H[k + 1, k].real]
            if G2 is not None:
                R[1:3] = G2.apply(R[1:3])
            G1 = G2
            # compute new givens rotation.
            G2 = utils.Givens(R[2:4])
            R[2] = G2.r
            R[3] = 0.0
            y = G2.apply(y)

            # update solution
            z = (V[:, [k]] - R[0, 0] * W[:, [0]] - R[1, 0] * W[:, [1]]) / R[2, 0]
            W = numpy.c_[W[:, [1]], z]
            yk = yk + y[0] * z
            y = [y[1], 0]

            self._finalize_iteration(yk, numpy.abs(y[0]))

        # compute solution if not yet done
        if self.xk is None:
            self.xk = self._get_xk(yk)

    def _finalize(self):
        super(Minres, self)._finalize()
        # store arnoldi?
        if self.store_arnoldi:
            if not isinstance(self.linear_system.M, utils.IdentityLinearOperator):
                self.V, self.H, self.P = self.lanczos.get()
            else:
                self.V, self.H = self.lanczos.get()

    @staticmethod
    def operations(nsteps):
        """Returns the number of operations needed for nsteps of MINRES"""
        return {
            "A": 1 + nsteps,
            "M": 2 + nsteps,
            "Ml": 2 + nsteps,
            "Mr": 1 + nsteps,
            "ip_B": 2 + 2 * nsteps,
            "axpy": 4 + 8 * nsteps,
        }


class BoundMinres(object):
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
        return super(BoundMinres, cls).__new__(cls)

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
