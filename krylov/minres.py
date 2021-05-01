import numpy as np

from ._helpers import Identity, Info, Product
from .arnoldi import Arnoldi
from .cg import BoundCG
from .errors import AssumptionError
from .givens import givens
from .utils import Intervals


def multi_dot(a, b):
    """a.dot.b for many a, b (i.e., a.shape == (n,...), y.shape == (n,...))"""
    return np.einsum("i...,i...->...", a, b)


def multi_matmul(A, b):
    """A @ b for many A, b (i.e., A.shape == (m,n,...), y.shape == (n,...))"""
    return np.einsum("ij...,j...->i...", A, b)


def minres(
    A,
    b,
    M=Identity(),
    Ml=Identity(),
    Mr=Identity(),
    inner=None,
    exact_solution=None,
    ortho="lanczos",
    x0=None,
    tol=1e-5,
    atol=1.0e-15,
    maxiter=None,
    use_explicit_residual=False,
    return_arnoldi=False,
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

    * if ``return_arnoldi==False``: 3 vectors or 6 vectors if :math:`M` is used.
    * if ``return_arnoldi==True``: about maxiter+1 vectors for the Lanczos
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

    if inner is None:
        inner_is_euclidean = True
        if len(b.shape) == 1:
            # np.dot is faster than einsum for flat vectors
            def inner(x, y):
                return np.dot(x.conj(), y)

        else:

            def inner(x, y):
                return np.einsum("i...,i...->...", x.conj(), y)

    else:
        inner_is_euclidean = False

    N = A.shape[0]

    if exact_solution is not None:
        assert exact_solution.shape == b.shape

    # Compute M^{-1}-norm of M*Ml*b.
    Ml_b = Ml @ b
    M_Ml_b = M @ Ml_b
    M_Ml_b_norm = np.sqrt(inner(Ml_b, M_Ml_b))

    def _get_xk(yk):
        """Compute approximate solution from initial guess and approximate solution
        of the preconditioned linear system."""
        Mr_yk = yk if Mr is None else Mr @ yk
        return x0 + Mr_yk

    def get_residual_norm(z):
        # \\| M M_l (b-Az)\\|_{M^{-1}}
        return get_residual_and_norm(z)[2]

    def get_residual_and_norm(z):
        # r = M M_l ( b - A z )
        Ml_r = Ml @ (b - A @ z)
        M_Ml_r = M @ Ml_r
        alpha = inner(Ml_r, M_Ml_r)
        nrm = np.sqrt(alpha.real ** 2 + alpha.imag ** 2)
        assert np.all(alpha.imag <= 1.0e-12 * nrm)
        alpha = alpha.real
        return M_Ml_r, Ml_r, np.sqrt(alpha)

    maxiter = N if maxiter is None else maxiter

    # sanitize initial guess
    if x0 is None:
        x0 = np.zeros_like(b)

    # get initial residual
    M_Ml_r0, Ml_r0, M_Ml_r0_norm = get_residual_and_norm(x0)

    dtype = M_Ml_r0.dtype

    # TODO: reortho
    k = 0

    resnorms = [M_Ml_r0_norm]
    """Residual norms as described for parameter ``tol``."""

    # compute error?
    if exact_solution is None:
        errnorms = None
    else:
        err = exact_solution - x0
        errnorms = [np.sqrt(inner(err, err))]

    Ml_A_Mr = Product(Ml, A, Mr)

    # initialize Lanczos
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

    # Necessary for efficient update of yk:
    W = [
        np.zeros(x0.shape, dtype=dtype),
        np.zeros(x0.shape, dtype=dtype),
    ]
    # some small helpers
    y = np.array([M_Ml_r0_norm, np.zeros_like(M_Ml_r0_norm)])
    # old Givens rotations
    G = [None, None]

    # resulting approximation is xk = x0 + Mr*yk
    yk = np.zeros(x0.shape, dtype=dtype)
    xk = None

    # iterate Lanczos
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

        V, H = arnoldi.__next__()
        assert np.all(np.abs(H.imag)) < 1.0e-14
        H = H.real

        # needed for QR-update:
        # R is real because Lanczos matrix is real
        R = np.zeros([4] + list(b.shape[1:]), dtype=float)
        # print(R.shape)
        # exit(1)

        R[1] = H[k - 1, k]
        if G[1] is not None:
            # apply givens rotation
            # R0 = G[1][0][1] * R[1]
            # R1 = G[1][1][1] * R[1]
            # R[0], R[1] = R0, R1
            R[:2] = multi_matmul(G[1], R[:2])

        # (implicit) update of QR-factorization of Lanczos matrix
        R[2:4] = [H[k, k], H[k + 1, k]]
        if G[0] is not None:
            R[1:3] = multi_matmul(G[0], R[1:3])
        G[1] = G[0]
        # compute new Givens rotation
        G[0] = givens(R[2:4])
        R[2] = multi_dot(G[0][0], R[2:4])  # r
        R[3] = 0.0
        # TODO second component of y is always 0
        y = multi_matmul(G[0], y)

        # update solution
        # The following two vector additions take the longest in this function
        z = (V[k] - R[0] * W[0] - R[1] * W[1]) / np.where(R[2] != 0.0, R[2], 1.0)
        W[0], W[1] = W[1], z
        yk += y[0] * z
        xk = None

        y = np.array([y[1], np.zeros_like(y[1])])

        # finalize iteration
        resnorm = np.abs(y[0])

        # compute error norm if asked for
        if exact_solution is not None:
            xk = _get_xk(yk) if xk is None else xk
            err = exact_solution - xk
            errnorms.append(np.sqrt(inner(err, err)))

        rkn = None
        if use_explicit_residual:
            xk = _get_xk(yk) if xk is None else xk
            rkn = get_residual_norm(xk)
            resnorm = rkn

        resnorms.append(resnorm)
        k += 1

    # compute solution if not yet done
    if xk is None:
        xk = _get_xk(yk)
    if return_arnoldi:
        V, H, P = arnoldi.get()

    num_operations = {
        "A": 1 + k,
        "M": 2 + k,
        "Ml": 2 + k,
        "Mr": 1 + k,
        "inner": 2 + 2 * k,
        "axpy": 4 + 8 * k,
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
        elif (np.array(evals) > -1e-15).all():
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
        if not np.isreal(evals).all():
            raise AssumptionError("non-real eigenvalues not allowed")

        # sort
        evals = np.sort(np.array(evals, dtype=float))

        # normalize and categorize evals
        evals /= np.max(np.abs(evals))
        negative = evals < -1e-15
        positive = evals > 1e-15

        lambda_1 = np.min(evals[negative])
        lambda_s = np.max(evals[negative])
        lambda_t = np.min(evals[positive])
        lambda_N = np.max(evals[positive])

        a = np.sqrt(np.abs(lambda_1 * lambda_N))
        b = np.sqrt(np.abs(lambda_s * lambda_t))

        self.base = (a - b) / (a + b)

    def eval_step(self, step):
        """Evaluate bound for given step."""
        return 2 * self.base ** np.floor(step / 2.0)

    def get_step(self, tol):
        """Return step at which bound falls below tolerance. """
        return 2 * np.log(tol / 2.0) / np.log(self.base)
