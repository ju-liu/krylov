import warnings

import numpy

from .errors import ArgumentError
from .householder import Householder
from .utils import (
    IdentityLinearOperator,
    LinearOperator,
    find_common_dtype,
    get_linearoperator,
    inner,
    norm,
)


def arnoldi_res(A, V, H, ip_B=None):
    """Measure Arnoldi residual.

    :param A: a linear operator that can be used with scipy's aslinearoperator with
      ``shape==(N,N)``.
    :param V: Arnoldi basis matrix with ``shape==(N,n)``.
    :param H: Hessenberg matrix: either :math:`\\underline{H}_{n-1}` with
      ``shape==(n,n-1)`` or :math:`H_n` with ``shape==(n,n)`` (if the Arnoldi basis
      spans an A-invariant subspace).
    :param ip_B: (optional) the inner product to use, see :py:meth:`inner`.

    :returns: either :math:`\\|AV_{n-1} - V_n \\underline{H}_{n-1}\\|` or
      :math:`\\|A V_n - V_n H_n\\|` (in the invariant case).
    """
    N = V.shape[0]
    invariant = H.shape[0] == H.shape[1]
    A = get_linearoperator((N, N), A)
    if invariant:
        res = A * V - numpy.dot(V, H)
    else:
        res = A * V[:, :-1] - numpy.dot(V, H)
    return norm(res, ip_B=ip_B)


class Arnoldi(object):
    def __init__(
        self, A, v, maxiter=None, ortho="mgs", M=None, Mv=None, Mv_norm=None, ip_B=None
    ):
        """Arnoldi algorithm.

        Computes V and H such that :math:`AV_n=V_{n+1}\\underline{H}_n`.  If
        the Krylov subspace becomes A-invariant then V and H are truncated such
        that :math:`AV_n = V_n H_n`.

        :param A: a linear operator that can be used with scipy's
          aslinearoperator with ``shape==(N,N)``.
        :param v: the initial vector with ``shape==(N,1)``.
        :param maxiter: (optional) maximal number of iterations. Default: N.
        :param ortho: (optional) orthogonalization algorithm: may be one of

            * ``'mgs'``: modified Gram-Schmidt (default).
            * ``'dmgs'``: double Modified Gram-Schmidt.
            * ``'lanczos'``: Lanczos short recurrence.
            * ``'house'``: Householder.
        :param M: (optional) a self-adjoint and positive definite
          preconditioner. If ``M`` is provided, then also a second basis
          :math:`P_n` is constructed such that :math:`V_n=MP_n`. This is of
          importance in preconditioned methods. ``M`` has to be ``None`` if
          ``ortho=='house'`` (see ``B``).
        :param ip_B: (optional) defines the inner product to use. See
          :py:meth:`inner`.

          ``ip_B`` has to be ``None`` if ``ortho=='house'``. It's unclear to me
          (andrenarchy), how a variant of the Householder QR algorithm can be
          used with a non-Euclidean inner product. Compare
          http://math.stackexchange.com/questions/433644/is-householder-orthogonalization-qr-practicable-for-non-euclidean-inner-products
        """
        N = v.shape[0]

        # save parameters
        self.A = get_linearoperator((N, N), A)
        self.maxiter = N if maxiter is None else maxiter
        self.ortho = ortho
        self.M = get_linearoperator((N, N), M)
        if isinstance(self.M, IdentityLinearOperator):
            self.M = None
        self.ip_B = ip_B

        self.dtype = find_common_dtype(A, v, M)
        # number of iterations
        self.iter = 0
        # Arnoldi basis
        self.V = numpy.zeros((N, self.maxiter + 1), dtype=self.dtype)
        if self.M is not None:
            self.P = numpy.zeros((N, self.maxiter + 1), dtype=self.dtype)
        # Hessenberg matrix
        self.H = numpy.zeros((self.maxiter + 1, self.maxiter), dtype=self.dtype)
        # flag indicating if Krylov subspace is invariant
        self.invariant = False

        if ortho == "house":
            if (
                self.M is not None and not isinstance(self.M, IdentityLinearOperator)
            ) or (
                not isinstance(self.ip_B, IdentityLinearOperator)
                and self.ip_B is not None
            ):
                raise ArgumentError(
                    "Only euclidean inner product allowed "
                    "with Householder orthogonalization"
                )
            self.houses = [Householder(v)]
            self.vnorm = numpy.linalg.norm(v, 2)
        elif ortho in ["mgs", "dmgs", "lanczos"]:
            self.reorthos = 0
            if ortho == "dmgs":
                self.reorthos = 1
            if self.M is not None:
                p = v
                if Mv is None:
                    v = self.M * p
                else:
                    v = Mv
                if Mv_norm is None:
                    self.vnorm = norm(p, v, ip_B=ip_B)
                else:
                    self.vnorm = Mv_norm
                if self.vnorm > 0:
                    if len(p.shape) == 1:
                        # TODO remove
                        p = p.reshape(len(p), 1)
                    self.P[:, [0]] = p / self.vnorm
            else:
                if Mv_norm is None:
                    self.vnorm = norm(v, ip_B=ip_B)
                else:
                    self.vnorm = Mv_norm
        else:
            raise ArgumentError(
                f"Invalid value '{ortho}' for argument 'ortho'. "
                + "Valid are house, mgs, dmgs and lanczos."
            )
        if self.vnorm > 0:
            if len(v.shape) == 1:
                # TODO remove
                v = v.reshape(len(v), 1)
            self.V[:, [0]] = v / self.vnorm
        else:
            self.invariant = True

    def advance(self):
        """Carry out one iteration of Arnoldi."""
        if self.iter >= self.maxiter:
            raise ArgumentError("Maximum number of iterations reached.")
        if self.invariant:
            raise ArgumentError(
                "Krylov subspace was found to be invariant in the previous iteration."
            )

        N = self.V.shape[0]
        k = self.iter

        # the matrix-vector multiplication
        Av = self.A * self.V[:, [k]]

        if self.ortho == "house":
            # Householder
            for j in range(k + 1):
                Av[j:] = self.houses[j].apply(Av[j:])
                Av[j] *= numpy.conj(self.houses[j].alpha)
            if k + 1 < N:
                house = Householder(Av[k + 1 :])
                self.houses.append(house)
                Av[k + 1 :] = house.apply(Av[k + 1 :]) * numpy.conj(house.alpha)
                self.H[: k + 2, [k]] = Av[: k + 2]
            else:
                self.H[: k + 1, [k]] = Av[: k + 1]
            # next line is safe due to the multiplications with alpha
            self.H[k + 1, k] = numpy.abs(self.H[k + 1, k])
            if (
                self.H[k + 1, k] / numpy.linalg.norm(self.H[: k + 2, : k + 1], 2)
                <= 1e-14
            ):
                self.invariant = True
            else:
                vnew = numpy.zeros((N, 1), dtype=self.dtype)
                vnew[k + 1] = 1
                for j in range(k + 1, -1, -1):
                    vnew[j:] = self.houses[j].apply(vnew[j:])
                self.V[:, [k + 1]] = vnew * self.houses[-1].alpha
        else:
            # determine vectors for orthogonalization
            start = 0

            # Lanczos?
            if self.ortho == "lanczos":
                start = k
                if k > 0:
                    self.H[k - 1, k] = self.H[k, k - 1]
                    if self.M is not None and not isinstance(
                        self.M, IdentityLinearOperator
                    ):
                        Av -= self.H[k, k - 1] * self.P[:, [k - 1]]
                    else:
                        Av -= self.H[k, k - 1] * self.V[:, [k - 1]]

            # (double) modified Gram-Schmidt
            for reortho in range(self.reorthos + 1):
                # orthogonalize
                for j in range(start, k + 1):
                    alpha = inner(self.V[:, [j]], Av, ip_B=self.ip_B)[0, 0]
                    if self.ortho == "lanczos":
                        # check if alpha is real
                        if abs(alpha.imag) > 1e-10:
                            warnings.warn(
                                f"Iter {self.iter}: abs(alpha.imag) = {abs(alpha.imag)} > 1e-10. "
                                "Is your operator self-adjoint in the "
                                "provided inner product?"
                            )
                        alpha = alpha.real
                    self.H[j, k] += alpha
                    if self.M is not None:
                        Av -= alpha * self.P[:, [j]]
                    else:
                        Av -= alpha * self.V[:, [j]]
            if self.M is not None:
                MAv = self.M * Av
                self.H[k + 1, k] = norm(Av, MAv, ip_B=self.ip_B)
            else:
                self.H[k + 1, k] = norm(Av, ip_B=self.ip_B)
            if (
                self.H[k + 1, k] / numpy.linalg.norm(self.H[: k + 2, : k + 1], 2)
                <= 1e-14
            ):
                self.invariant = True
            else:
                if self.M is not None:
                    self.P[:, [k + 1]] = Av / self.H[k + 1, k]
                    self.V[:, [k + 1]] = MAv / self.H[k + 1, k]
                else:
                    self.V[:, [k + 1]] = Av / self.H[k + 1, k]

        # increase iteration counter
        self.iter += 1

    def get(self):
        k = self.iter
        if self.invariant:
            V, H = self.V[:, :k], self.H[:k, :k]
            if self.M:
                return V, H, self.P[:, :k]
            return V, H
        else:
            V, H = self.V[:, : k + 1], self.H[: k + 1, :k]
            if self.M:
                return V, H, self.P[:, : k + 1]
            return V, H

    def get_last(self):
        k = self.iter
        if self.invariant:
            V, H = None, self.H[:k, [k - 1]]
            if self.M:
                return V, H, None
            return V, H
        else:
            V, H = self.V[:, [k]], self.H[: k + 1, [k - 1]]
            if self.M:
                return V, H, self.P[:, [k]]
            return V, H


def arnoldi(*args, **kwargs):
    _arnoldi = Arnoldi(*args, **kwargs)
    while _arnoldi.iter < _arnoldi.maxiter and not _arnoldi.invariant:
        _arnoldi.advance()
    return _arnoldi.get()


def arnoldi_projected(H, P, k, ortho="mgs"):
    """Compute (perturbed) Arnoldi relation for projected operator.

    Assume that you have computed an Arnoldi relation

    .. math ::

        A V_n = V_{n+1} \\underline{H}_n

    where :math:`V_{n+1}\\in\\mathbb{C}^{N,n+1}` has orthogonal columns
    (with respect to an inner product :math:`\\langle\\cdot,\\cdot\\rangle`)
    and :math:`\\underline{H}_n\\in\\mathbb{C}^{n+1,n}` is an extended
    upper Hessenberg matrix.

    For :math:`k<n` you choose full rank matrices
    :math:`X\\in\\mathbb{C}^{n-1,k}` and :math:`Y\\in\\mathbb{C}^{n,k}` and
    define :math:`\\tilde{X}:=A V_{n_1}X = V_n \\underline{H}_{n-1} X` and
    :math:`\\tilde{Y}:=V_n Y` such that
    :math:`\\langle \\tilde{Y}, \\tilde{X} \\rangle = Y^*\\underline{H}_{n-1} X`
    is invertible. Then the projections :math:`P` and :math:`\\tilde{P}`
    characterized by

    * :math:`\\tilde{P}x = x -
      \\tilde{X} \\langle \\tilde{Y},\\tilde{X} \\rangle^{-1}
      \\langle\\tilde{Y},x\\rangle`
    * :math:`P = I - \\underline{H}_{n-1}X (Y^*\\underline{H}_{n-1}X)^{-1}Y^*`

    are well defined and :math:`\\tilde{P}V_{n+1} = [V_n P, v_{n+1}]` holds.

    This method computes for :math:`i<n-k` the Arnoldi relation

    .. math ::

        (\\tilde{P}A + E_i) W_i
        = W_{i+1} \\underline{G}_i

    where :math:`W_{i+1}=V_n U_{i+1}` has orthogonal columns with respect
    to :math:`\\langle\\cdot,\\cdot\\rangle`,
    :math:`\\underline{G}_i` is an extended upper Hessenberg matrix
    and :math:`E_i x = v_{n+1} F_i \\langle W_i,x\\rangle` with
    :math:`F_i=[f_1,\\ldots,f_i]\\in\\mathbb{C}^{1,i}`.

    The perturbed Arnoldi relation can also be generated with the operator
    :math:`P_{V_n} \\tilde{P} A`:

    .. math ::

        P_{V_n} \\tilde{P} A W_i
        = W_{i+1} \\underline{G}_i.

    In a sense the perturbed Arnoldi relation is the best prediction for the
    behavior of the Krylov subspace :math:`K_i(\\tilde{P}A,\\tilde{P}v_1)`
    that can be generated only with the data from :math:`K_{n+1}(A,v_1)` and
    without carrying out further matrix-vector multiplications with A.

    :param H: the extended upper Hessenberg matrix
      :math:`\\underline{H}_n` with ``shape==(n+1,n)``.
    :param P: the projection
      :math:`P:\\mathbb{C}^n\\longrightarrow\\mathbb{C}^n` (has to be
      compatible with :py:meth:`get_linearoperator`).
    :param k: the dimension of the null space of P.
    :returns: U, G, F where

      * U is the coefficient matrix :math:`U_{i+1}` with ``shape==(n,i+1)``,
      * G is the extended upper Hessenberg matrix :math:`\\underline{G}_i`
        with ``shape==(i+1,i)``,
      * F is the error matrix :math:`F_i` with ``shape==(1,i)``.
    """
    n = H.shape[1]
    dtype = find_common_dtype(H, P)
    invariant = H.shape[0] == n
    hlast = 0 if invariant else H[-1, -1]
    H = get_linearoperator((n, n), H if invariant else H[:-1, :])
    P = get_linearoperator((n, n), P)
    v = P * numpy.eye(n, 1)
    maxiter = n - k + 1
    F = numpy.zeros((1, maxiter), dtype=dtype)
    PH = LinearOperator((n, n), dtype, lambda x: P * (H * x))
    _arnoldi = Arnoldi(PH, v, maxiter=maxiter, ortho=ortho)
    while _arnoldi.iter < _arnoldi.maxiter and not _arnoldi.invariant:
        u, _ = _arnoldi.get_last()
        F[0, _arnoldi.iter] = hlast * u[-1, 0]
        _arnoldi.advance()
    U, G = _arnoldi.get()
    return U, G, F[[0], : _arnoldi.iter]
