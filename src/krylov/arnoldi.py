"""Arnoldi algorithm.

Computes V and H such that :math:`AV_n = V_{n+1}\\underline{H}_n`. If the Krylov
subspace becomes A-invariant then V and H are truncated such that :math:`AV_n = V_n
H_n`.

:param A: a linear operator that works with the @-operator
:param v: the initial vector.
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
import numpy as np

from ._helpers import Identity, aslinearoperator, get_default_inner
from .errors import ArgumentError
from .householder import Householder


class ArnoldiHouseholder:
    def __init__(self, A, v):
        self.inner = get_default_inner(v.shape)

        # save parameters
        self.A = A
        self.v = v

        self.dtype = np.find_common_type([A.dtype, v.dtype], [])

        # number of iterations
        self.iter = 0
        # Arnoldi basis
        self.V = []

        # flag indicating if Krylov subspace is invariant
        self.is_invariant = False

        self.houses = [Householder(v)]
        self.vnorm = np.linalg.norm(v, 2)

        # TODO set self.is_invariant = True for self.vnorm == 0
        self.V.append(v / np.where(self.vnorm != 0.0, self.vnorm, 1.0))

        # if self.vnorm > 0:
        #     self.V[0] = v / self.vnorm
        # else:
        #     self.is_invariant = True

    def __iter__(self):
        return self

    def __next__(self):
        """Carry out one iteration of Arnoldi."""
        if self.is_invariant:
            raise ArgumentError(
                "Krylov subspace was found to be invariant in the previous iteration."
            )

        k = self.iter
        Av = self.A @ self.V[k]

        for j in range(k + 1):
            Av[j:] = self.houses[j] @ Av[j:]
            Av[j] *= np.conj(self.houses[j].alpha)

        N = self.v.shape[0]
        if k < N - 1:
            house = Householder(Av[k + 1 :])
            self.houses.append(house)
            Av[k + 1 :] = (house @ Av[k + 1 :]) * np.conj(house.alpha)
            h = Av[: k + 2]
            h[-1] = np.abs(h[-1])

            if h[-1] <= 1.0e-14:
                self.is_invariant = True
                v = None
            else:
                vnew = np.zeros_like(self.v)
                vnew[k + 1] = 1
                for j in range(k + 1, -1, -1):
                    vnew[j:] = self.houses[j] @ vnew[j:]
                v = vnew * self.houses[-1].alpha
                self.V.append(v)
        else:
            h = np.zeros([len(Av) + 1] + list(self.v.shape[1:]), Av.dtype)
            h[:-1] = Av
            self.is_invariant = True
            v = None

        self.iter += 1
        return v, h


class ArnoldiMGS:
    def __init__(
        self,
        A,
        v,
        num_reorthos: int = 1,
        M=None,
        Mv=None,
        Mv_norm=None,
        inner=None,
    ):
        self.inner = get_default_inner(v.shape) if inner is None else inner

        # save parameters
        self.A = A
        self.v = v
        self.num_reorthos = num_reorthos
        self.M = Identity() if M is None else aslinearoperator(M)

        self.dtype = np.find_common_type([A.dtype, self.M.dtype, v.dtype], [])

        # number of iterations
        self.iter = 0
        # Arnoldi basis
        self.V = []
        self.P = []

        # flag indicating if Krylov subspace is invariant
        self.is_invariant = False

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

        # TODO set self.is_invariant = True for self.vnorm == 0
        self.V.append(v / np.where(self.vnorm != 0.0, self.vnorm, 1.0))

        # if self.vnorm > 0:
        #     self.V[0] = v / self.vnorm
        # else:
        #     self.is_invariant = True

    def next_mgs(self, k, Av):
        # modified Gram-Schmidt orthogonalization
        for j in range(k + 1):
            alpha = self.inner(self.V[j], Av)
            self.h[j] += alpha
            Av -= alpha * self.P[j]

    def __iter__(self):
        return self

    def __next__(self):
        if self.is_invariant:
            raise ArgumentError(
                "Krylov subspace was found to be invariant in the previous iteration."
            )

        k = self.iter

        # the matrix-vector multiplication
        Av = self.A @ self.V[k]

        self.h = np.zeros([k + 2] + list(self.v.shape[1:]), dtype=self.dtype)

        # determine vectors for orthogonalization
        for _ in range(self.num_reorthos):
            self.next_mgs(k, Av)

        MAv = self.M @ Av
        self.h[k + 1] = np.sqrt(self.inner(Av, MAv))

        if np.all(self.h[k + 1] <= 1.0e-14):
            self.is_invariant = True
            v = None
        else:
            Hk1k = np.where(self.h[k + 1] != 0.0, self.h[k + 1], 1.0)
            self.P.append(Av / Hk1k)
            v = MAv / Hk1k

        if v is not None:
            self.V.append(v)

        # increase iteration counter
        self.iter += 1
        return v, self.h


class ArnoldiLanczos:
    def __init__(self, A, v, M=None, Mv=None, Mv_norm=None, inner=None):
        self.A = A
        self.M = Identity() if M is None else aslinearoperator(M)
        self.inner = get_default_inner(v.shape) if inner is None else inner

        self.dtype = np.find_common_type([A.dtype, self.M.dtype, v.dtype], [])

        # number of iterations
        self.num_iter = 0

        # stores the three tridiagonal entries of the Hessenberg matrix
        self.h = np.zeros([3] + list(v.shape[1:]), dtype=self.dtype)

        # flag indicating if Krylov subspace is invariant
        self.is_invariant = False

        p = v
        v = self.M @ p if Mv is None else Mv
        self.vnorm = np.sqrt(inner(p, v)) if Mv_norm is None else Mv_norm

        # self.P.append(p / np.where(self.vnorm != 0.0, self.vnorm, 1.0))
        # # TODO set self.is_invariant = True for self.vnorm == 0
        # self.V.append(v / np.where(self.vnorm != 0.0, self.vnorm, 1.0))

        self.p_old = None
        self.p = p / np.where(self.vnorm != 0.0, self.vnorm, 1.0)
        self.v = v / np.where(self.vnorm != 0.0, self.vnorm, 1.0)

        # if self.vnorm > 0:
        #     self.V[0] = v / self.vnorm
        # else:
        #     self.is_invariant = True

    def __next__(self):
        """Carry out one iteration of Arnoldi."""
        if self.is_invariant:
            raise ArgumentError(
                "Krylov subspace was found to be invariant in the previous iteration."
            )

        Av = self.A @ self.v

        if self.num_iter > 0:
            # copy the old lower-diagonal entry to the upper diagonal
            self.h[0] = self.h[2]
            Av -= self.h[0] * self.p_old

        # orthogonalize
        alpha = self.inner(self.v, Av)
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
        self.h[1] = alpha
        Av -= alpha * self.p

        MAv = self.M @ Av
        self.h[2] = np.sqrt(self.inner(Av, MAv))

        if np.all(self.h[2] <= 1.0e-14):
            self.is_invariant = True
            self.v = None
            self.p = None
        else:
            Hk1k = np.where(self.h[2] != 0.0, self.h[2], 1.0)
            self.p_old = self.p
            self.p = Av / Hk1k
            self.v = MAv / Hk1k

        # increase iteration counter
        self.num_iter += 1
        return self.v, self.h, self.p


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
