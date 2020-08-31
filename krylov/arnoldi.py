import warnings

import numpy

from .errors import ArgumentError
from .householder import Householder


def arnoldi_res(A, V, H, inner=None):
    """Measure Arnoldi residual.

    :param A: a linear operator that can be used with scipy's aslinearoperator with
      ``shape==(N,N)``.
    :param V: Arnoldi basis matrix with ``shape==(N,n)``.
    :param H: Hessenberg matrix: either :math:`\\underline{H}_{n-1}` with
      ``shape==(n,n-1)`` or :math:`H_n` with ``shape==(n,n)`` (if the Arnoldi basis
      spans an A-invariant subspace).
    :param inner: (optional) the inner product to use, see :py:meth:`inner`.

    :returns: either :math:`\\|AV_{n-1} - V_n \\underline{H}_{n-1}\\|` or
      :math:`\\|A V_n - V_n H_n\\|` (in the invariant case).
    """
    invariant = H.shape[0] == H.shape[1]
    V1 = V if invariant else V[:, :-1]
    res = A * V1 - numpy.dot(V, H)
    return numpy.sqrt(inner(res, res))


def matrix_2_norm(A):
    """Computes the max singular value of all matrices of shape (n, n, ...). The result
    has shape (...).
    """
    return numpy.max(numpy.linalg.svd(A.T, compute_uv=False).T, axis=0)


class Arnoldi:
    def __init__(
        self, A, v, maxiter=None, ortho="mgs", M=None, Mv=None, Mv_norm=None, inner=None
    ):
        """Arnoldi algorithm.

        Computes V and H such that :math:`AV_n = V_{n+1}\\underline{H}_n`. If the Krylov
        subspace becomes A-invariant then V and H are truncated such that :math:`AV_n =
        V_n H_n`.

        :param A: a linear operator that can be used with scipy's aslinearoperator with
        ``shape==(N,N)``.
        :param v: the initial vector.
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
        :param inner: (optional) defines the inner product to use. See
          :py:meth:`inner`.

          ``inner`` has to be ``None`` if ``ortho=='house'``. It's unclear to me
          (andrenarchy), how a variant of the Householder QR algorithm can be
          used with a non-Euclidean inner product. Compare
          http://math.stackexchange.com/questions/433644/is-householder-orthogonalization-qr-practicable-for-non-euclidean-inner-products
        """
        N = v.shape[0]

        inner_is_euclidean = inner is None
        if inner is None:
            self.inner = lambda x, y: numpy.dot(x.T.conj(), y)
        else:
            self.inner = inner

        # save parameters
        self.A = A
        self.maxiter = N if maxiter is None else maxiter
        self.ortho = ortho
        self.M = M

        # we're computing the products only to find out the dtype; perhaps there's a
        # better way
        Av = A @ v
        MAv = Av if self.M is None else self.M @ Av
        self.dtype = MAv.dtype

        # number of iterations
        self.iter = 0
        # Arnoldi basis
        self.V = numpy.zeros([self.maxiter + 1] + list(v.shape), dtype=self.dtype)
        if self.M is not None:
            self.P = numpy.zeros([self.maxiter + 1] + list(v.shape), dtype=self.dtype)
        # Hessenberg matrix
        self.H = numpy.zeros(
            [self.maxiter + 1, self.maxiter] + list(v.shape[1:]), dtype=self.dtype
        )
        # flag indicating if Krylov subspace is invariant
        self.invariant = False

        if ortho == "house":
            if self.M is not None or not inner_is_euclidean:
                raise ArgumentError(
                    "Only Euclidean inner product allowed "
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
                    v = self.M @ p
                else:
                    v = Mv
                if Mv_norm is None:
                    self.vnorm = numpy.sqrt(inner(p, v))
                else:
                    self.vnorm = Mv_norm

                mask = self.vnorm > 0.0
                self.P[0][:, mask] = p[:, mask] / self.vnorm[mask]
            else:
                if Mv_norm is None:
                    self.vnorm = numpy.sqrt(inner(v, v))
                else:
                    self.vnorm = Mv_norm
        else:
            raise ArgumentError(
                f"Invalid value '{ortho}' for argument 'ortho'. "
                + "Valid are house, mgs, dmgs and lanczos."
            )

        # TODO set self.invariant = True for self.vnorm == 0
        mask = self.vnorm > 0.0
        self.V[0][:, mask] = v[:, mask] / self.vnorm[mask]

        # if self.vnorm > 0:
        #     self.V[0] = v / self.vnorm
        # else:
        #     self.invariant = True

    def __next__(self):
        """Carry out one iteration of Arnoldi."""
        if self.iter >= self.maxiter:
            raise ArgumentError("Maximum number of iterations reached.")
        if self.invariant:
            raise ArgumentError(
                "Krylov subspace was found to be invariant in the previous iteration."
            )

        N = self.V.shape[1]
        k = self.iter

        # the matrix-vector multiplication
        Av = self.A @ self.V[k]

        if self.ortho == "house":
            # Householder
            for j in range(k + 1):
                Av[j:] = self.houses[j].apply(Av[j:])
                Av[j] *= numpy.conj(self.houses[j].alpha)
            if k + 1 < N:
                house = Householder(Av[k + 1 :])
                self.houses.append(house)
                Av[k + 1 :] = house.apply(Av[k + 1 :]) * numpy.conj(house.alpha)
                self.H[: k + 2, k] = Av[: k + 2]
            else:
                self.H[: k + 1, k] = Av[: k + 1]
            # next line is safe due to the multiplications with alpha
            self.H[k + 1, k] = numpy.abs(self.H[k + 1, k])
            nrm = numpy.linalg.norm(self.H[: k + 2, : k + 1], 2)
            if self.H[k + 1, k] <= 1e-14 * nrm:
                self.invariant = True
            else:
                vnew = numpy.zeros((N, 1), dtype=self.dtype)
                vnew[k + 1] = 1
                for j in range(k + 1, -1, -1):
                    vnew[j:] = self.houses[j].apply(vnew[j:])
                self.V[k + 1] = vnew * self.houses[-1].alpha
        else:
            # determine vectors for orthogonalization
            start = 0

            # Lanczos?
            if self.ortho == "lanczos":
                start = k
                if k > 0:
                    self.H[k - 1, k] = self.H[k, k - 1]
                    P = self.V if self.M is None else self.P
                    Av -= self.H[k, k - 1] * P[k - 1]

            # (double) modified Gram-Schmidt
            for reortho in range(self.reorthos + 1):
                # orthogonalize
                for j in range(start, k + 1):
                    alpha = self.inner(self.V[j], Av)
                    if self.ortho == "lanczos":
                        # check if alpha is real
                        if abs(alpha.imag) > 1e-10:
                            warnings.warn(
                                f"Iter {self.iter}: "
                                f"abs(alpha.imag) = {abs(alpha.imag)} > 1e-10. "
                                "Is your operator self-adjoint "
                                "in the provided inner product?"
                            )
                        alpha = alpha.real
                    self.H[j, k] += alpha

                    P = self.V if self.M is None else self.P
                    Av -= alpha * P[j]

            MAv = Av if self.M is None else self.M @ Av
            self.H[k + 1, k] = numpy.sqrt(self.inner(Av, MAv))

            Hk_nrm = matrix_2_norm(self.H[: k + 2, : k + 1])
            if numpy.all(self.H[k + 1, k] <= 1e-14 * Hk_nrm + 1.0e-14):
                self.invariant = True
            else:
                Hk1k = numpy.where(self.H[k + 1, k] != 0.0, self.H[k + 1, k], 1.0)
                if self.M is not None:
                    self.P[k + 1] = Av / Hk1k
                    self.V[k + 1] = MAv / Hk1k
                else:
                    self.V[k + 1] = Av / Hk1k

        # increase iteration counter
        self.iter += 1
        return self.V, self.H

    def get(self):
        k = self.iter if self.invariant else self.iter + 1
        V, H = self.V[:k], self.H[:k, :k]
        P = None if self.M is None else self.P[:k]
        return V, H, P


def arnoldi(*args, **kwargs):
    _arnoldi = Arnoldi(*args, **kwargs)
    while _arnoldi.iter < _arnoldi.maxiter and not _arnoldi.invariant:
        next(_arnoldi)
    return _arnoldi.get()
