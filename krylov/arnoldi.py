import numpy as np

from ._helpers import Identity
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
    res = A * V1 - np.dot(V, H)
    return np.sqrt(inner(res, res))


def matrix_2_norm(A):
    """Computes the max singular value of all matrices of shape (n, n, ...). The result
    has shape (...).
    """
    return np.max(np.linalg.svd(A.T, compute_uv=False).T, axis=0)


class Arnoldi:
    def __init__(
        self,
        A,
        v,
        maxiter=None,
        ortho="mgs",
        M=Identity(),
        Mv=None,
        Mv_norm=None,
        inner=None,
        inner_is_euclidean=False,
    ):
        """Arnoldi algorithm.

        Computes V and H such that :math:`AV_n = V_{n+1}\\underline{H}_n`. If the Krylov
        subspace becomes A-invariant then V and H are truncated such that :math:`AV_n =
        V_n H_n`.

        :param A: a linear operator that works with the @-operator
        :param v: the initial vector.
        :param maxiter: (optional) maximal number of iterations. Default: N.
        :param ortho: (optional) orthogonalization algorithm: may be one of

            * ``'mgs'``: modified Gram-Schmidt (default).
            * ``'dmgs'``: double Modified Gram-Schmidt.
            * ``'lanczos'``: Lanczos short recurrence.
            * ``'householder'``: Householder.
        :param M: (optional) a self-adjoint and positive-definite preconditioner. If
        ``M`` is provided, then also a second basis :math:`P_n` is constructed such that
        :math:`V_n=MP_n`. This is of importance in preconditioned methods. ``M`` has to
        be ``None`` if ``ortho=='householder'`` (see ``B``).
        :param inner: (optional) defines the inner product to use. See
          :py:meth:`inner`.

          ``inner`` has to be ``None`` if ``ortho=='householder'``. It's unclear how a
          variant of the Householder QR algorithm can be used with a non-Euclidean inner
          product. Compare <https://math.stackexchange.com/q/433644/36678>.
        """
        N = v.shape[0]

        self.inner = inner if inner is not None else lambda x, y: np.dot(x.T.conj(), y)

        # save parameters
        self.A = A
        self.v = v
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
        self.V = []
        if self.M is not None:
            self.P = []
        # Hessenberg matrix
        self.H = np.zeros(
            [self.maxiter + 1, self.maxiter] + list(v.shape[1:]), dtype=self.dtype
        )
        # flag indicating if Krylov subspace is invariant
        self.invariant = False

        if ortho == "householder":
            if not isinstance(self.M, Identity) or not inner_is_euclidean:
                raise ArgumentError(
                    "Only Euclidean inner product allowed "
                    "with Householder orthogonalization"
                )
            self.houses = [Householder(v)]
            self.vnorm = np.linalg.norm(v, 2)
        elif ortho in ["mgs", "dmgs", "lanczos"]:
            self.num_reorthos = 1 if ortho == "dmgs" else 0
            if self.M is None:
                if Mv_norm is None:
                    self.vnorm = np.sqrt(inner(v, v))
                else:
                    self.vnorm = Mv_norm
            else:
                p = v
                if Mv is None:
                    v = self.M @ p
                else:
                    v = Mv
                if Mv_norm is None:
                    self.vnorm = np.sqrt(inner(p, v))
                else:
                    self.vnorm = Mv_norm

                self.P.append(p / np.where(self.vnorm != 0.0, self.vnorm, 1.0))
        else:
            raise ArgumentError(
                f"Invalid value '{ortho}' for argument 'ortho'. "
                + "Valid are householder, mgs, dmgs and lanczos."
            )

        # TODO set self.invariant = True for self.vnorm == 0
        self.V.append(v / np.where(self.vnorm != 0.0, self.vnorm, 1.0))

        # if self.vnorm > 0:
        #     self.V[0] = v / self.vnorm
        # else:
        #     self.invariant = True

    def next_householder(self, k, Av):
        # Householder
        for j in range(k + 1):
            Av[j:] = self.houses[j].apply(Av[j:])
            Av[j] *= np.conj(self.houses[j].alpha)
        N = self.v.shape[0]
        if k + 1 < N:
            house = Householder(Av[k + 1 :])
            self.houses.append(house)
            Av[k + 1 :] = house.apply(Av[k + 1 :]) * np.conj(house.alpha)
            self.H[: k + 2, k] = Av[: k + 2]
        else:
            self.H[: k + 1, k] = Av[: k + 1]
        # next line is safe due to the multiplications with alpha
        self.H[k + 1, k] = np.abs(self.H[k + 1, k])
        nrm = matrix_2_norm(self.H[: k + 2, : k + 1])
        if self.H[k + 1, k] <= 1e-14 * nrm:
            self.invariant = True
        else:
            vnew = np.zeros_like(self.v)
            vnew[k + 1] = 1
            for j in range(k + 1, -1, -1):
                vnew[j:] = self.houses[j].apply(vnew[j:])
            self.V.append(vnew * self.houses[-1].alpha)

    def next_lanczos(self, k, Av):
        if k > 0:
            self.H[k - 1, k] = self.H[k, k - 1]
            P = self.V if self.M is None else self.P
            Av -= self.H[k, k - 1] * P[k - 1]
        # (double) modified Gram-Schmidt
        P = self.V if self.M is None else self.P
        # orthogonalize
        alpha = self.inner(self.V[k], Av)
        # if self.ortho == "lanczos":
        #     # check if alpha is real
        #     if abs(alpha.imag) > 1e-10:
        #         warnings.warn(
        #             f"Iter {self.iter}: "
        #             f"abs(alpha.imag) = {abs(alpha.imag)} > 1e-10. "
        #             "Is your operator self-adjoint "
        #             "in the provided inner product?"
        #         )
        #     alpha = alpha.real
        self.H[k, k] += alpha
        Av -= alpha * P[k]

    def next_mgs(self, k, Av):
        # modified Gram-Schmidt
        P = self.V if self.M is None else self.P
        # orthogonalize
        for j in range(k + 1):
            alpha = self.inner(self.V[j], Av)
            # if self.ortho == "lanczos":
            #     # check if alpha is real
            #     if abs(alpha.imag) > 1e-10:
            #         warnings.warn(
            #             f"Iter {self.iter}: "
            #             f"abs(alpha.imag) = {abs(alpha.imag)} > 1e-10. "
            #             "Is your operator self-adjoint "
            #             "in the provided inner product?"
            #         )
            #     alpha = alpha.real
            self.H[j, k] += alpha
            Av -= alpha * P[j]

    def __next__(self):
        """Carry out one iteration of Arnoldi."""
        if self.iter >= self.maxiter:
            raise ArgumentError("Maximum number of iterations reached.")
        if self.invariant:
            raise ArgumentError(
                "Krylov subspace was found to be invariant in the previous iteration."
            )

        k = self.iter

        # the matrix-vector multiplication
        Av = self.A @ self.V[k]

        if self.ortho == "householder":
            self.next_householder(k, Av)
        else:
            # determine vectors for orthogonalization
            if self.ortho == "lanczos":
                self.next_lanczos(k, Av)
            elif self.ortho == "mgs":
                self.next_mgs(k, Av)
            else:
                assert self.ortho == "dmgs"
                # double modified Gram-Schmidt
                self.next_mgs(k, Av)
                self.next_mgs(k, Av)

            MAv = Av if self.M is None else self.M @ Av
            self.H[k + 1, k] = np.sqrt(self.inner(Av, MAv))

            Hk_nrm = matrix_2_norm(self.H[: k + 2, : k + 1])
            if np.all(self.H[k + 1, k] <= 1e-14 * Hk_nrm + 1.0e-14):
                self.invariant = True
            else:
                Hk1k = np.where(self.H[k + 1, k] != 0.0, self.H[k + 1, k], 1.0)
                if self.M is not None:
                    self.P.append(Av / Hk1k)
                    self.V.append(MAv / Hk1k)
                else:
                    self.V.append(Av / Hk1k)

        # increase iteration counter
        self.iter += 1
        return self.V, self.H

    def get(self):
        k = self.iter if self.invariant else self.iter + 1
        H = self.H[:k, :k]
        P = None if self.M is None else self.P
        return self.V, H, P


def arnoldi(*args, **kwargs):
    _arnoldi = Arnoldi(*args, **kwargs)
    while _arnoldi.iter < _arnoldi.maxiter and not _arnoldi.invariant:
        next(_arnoldi)
    return _arnoldi.get()
