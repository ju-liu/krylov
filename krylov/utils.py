"""
Collection of standard functions.

This method provides functions like inner products, norms, ...
"""
import warnings

import numpy
import scipy.linalg
from scipy.sparse import isspmatrix

from .errors import ArgumentError
from .givens import givens

__all__ = [
    "NormalizedRootsPolynomial",
    "angles",
    "gap",
    "hegedus",
    "qr",
    "strakos",
]


def find_common_dtype(*args):
    """Returns common dtype of numpy and scipy objects.

    Recognizes ndarray, spmatrix. All other objects are ignored (most notably None)."""
    dtypes = []
    for arg in args:
        if type(arg) is numpy.ndarray or isspmatrix(arg):
            if hasattr(arg, "dtype"):
                dtypes.append(arg.dtype)
            else:
                warnings.warn(f"object {arg.__repr__} does not have a dtype.")
    return numpy.find_common_type(dtypes, [])


def qr(X, inner=None, reorthos=1):
    """QR factorization with customizable inner product.

    :param X: array with ``shape==(N,k)``
    :param inner: (optional) inner product, see :py:meth:`inner`.
    :param reorthos: (optional) number of reorthogonalizations. Defaults to 1 (i.e. 2
      runs of modified Gram-Schmidt) which should be enough in most cases (TODO: add
      reference).

    :return: Q, R where :math:`X=QR` with :math:`\\langle Q,Q \\rangle=I_k` and
      R upper triangular.
    """
    if inner is None and X.shape[1] > 0:
        return scipy.linalg.qr(X, mode="economic")

    (N, k) = X.shape
    Q = X.copy()
    R = numpy.zeros((k, k), dtype=X.dtype)
    for i in range(k):
        for reortho in range(reorthos + 1):
            for j in range(i):
                alpha = inner(Q[:, [j]], Q[:, [i]])
                R[j, i] += alpha
                Q[:, [i]] -= alpha * Q[:, [j]]

        R[i, i] = numpy.sqrt(numpy.linalg.norm(inner(Q[:, [i]], Q[:, [i]]), 2))
        if R[i, i] >= 1e-15:
            Q[:, [i]] /= R[i, i]
    return Q, R


def angles(F, G, inner=None, compute_vectors=False):
    """Principal angles between two subspaces.

    This algorithm is based on algorithm 6.2 in `Knyazev, Argentati. Principal angles
    between subspaces in an A-based scalar product: algorithms and perturbation
    estimates. 2002.` This algorithm can also handle small angles (in contrast to the
    naive cosine-based svd algorithm).

    :param F: array with ``shape==(N,k)``.
    :param G: array with ``shape==(N,l)``.
    :param inner: (optional) angles are computed with respect to this
      inner product. See :py:meth:`inner`.
    :param compute_vectors: (optional) if set to ``False`` then only the angles are
      returned (default). If set to ``True`` then also the principal vectors are
      returned.

    :return:

      * ``theta`` if ``compute_vectors==False``
      * ``theta, U, V`` if ``compute_vectors==True``

      where

      * ``theta`` is the array with ``shape==(max(k,l),)`` containing the principal
      angles
        :math:`0\\leq\\theta_1\\leq\\ldots\\leq\\theta_{\\max\\{k,l\\}}\\leq
        \\frac{\\pi}{2}`.
      * ``U`` are the principal vectors from F with
        :math:`\\langle U,U \\rangle=I_k`.
      * ``V`` are the principal vectors from G with
        :math:`\\langle V,V \\rangle=I_l`.

    The principal angles and vectors fulfill the relation
    :math:`\\langle U,V \\rangle = \
    \\begin{bmatrix} \
    \\cos(\\Theta) & 0_{m,l-m} \\\\ \
    0_{k-m,m} & 0_{k-m,l-m} \
    \\end{bmatrix}`
    where :math:`m=\\min\\{k,l\\}` and
    :math:`\\cos(\\Theta)=\\operatorname{diag}(\\cos(\\theta_1),\\ldots,\\cos(\\theta_m))`.
    Furthermore,
    :math:`\\theta_{m+1}=\\ldots=\\theta_{\\max\\{k,l\\}}=\\frac{\\pi}{2}`.
    """
    # make sure that F.shape[1]>=G.shape[1]
    reverse = False
    if F.shape[1] < G.shape[1]:
        reverse = True
        F, G = G, F

    QF, _ = qr(F, inner=inner)
    QG, _ = qr(G, inner=inner)

    # one or both matrices empty? (enough to check G here)
    if G.shape[1] == 0:
        theta = numpy.ones(F.shape[1]) * numpy.pi / 2
        U = QF
        V = QG
    else:
        Y, s, Z = scipy.linalg.svd(inner(QF, QG))
        Vcos = numpy.dot(QG, Z.T.conj())
        n_large = numpy.flatnonzero((s ** 2) < 0.5).shape[0]
        n_small = s.shape[0] - n_large
        theta = numpy.hstack(
            [
                numpy.arccos(s[n_small:]),  # [-i:] does not work if i==0
                numpy.ones(F.shape[1] - G.shape[1]) * numpy.pi / 2,
            ]
        )
        if compute_vectors:
            Ucos = numpy.dot(QF, Y)
            U = Ucos[:, n_small:]
            V = Vcos[:, n_small:]

        if n_small > 0:
            RG = Vcos[:, :n_small]
            S = RG - numpy.dot(QF, inner(QF, RG))
            _, R = qr(S, inner=inner)
            Y, u, Z = scipy.linalg.svd(R)
            theta = numpy.hstack([numpy.arcsin(u[::-1][:n_small]), theta])
            if compute_vectors:
                RF = Ucos[:, :n_small]
                Vsin = numpy.dot(RG, Z.T.conj())
                # next line is hand-crafted since the line from the paper does
                # not seem to work.
                Usin = numpy.dot(
                    RF,
                    numpy.dot(
                        numpy.diag(1 / s[:n_small]),
                        numpy.dot(Z.T.conj(), numpy.diag(s[:n_small])),
                    ),
                )
                U = numpy.column_stack([Usin, U])
                V = numpy.column_stack([Vsin, V])

    if compute_vectors:
        if reverse:
            U, V = V, U
        return theta, U, V
    else:
        return theta


def hegedus(A, b, x0, M=None, Ml=None, inner=None):
    """Rescale initial guess appropriately (Hegedüs trick).

    The Hegedüs trick rescales the initial guess to :math:`\\gamma_{\\min} x_0` such
    that

    .. math ::

      \\|r_0\\|_{M^{-1}}
      = \\| M M_l (b - A \\gamma_{\\min} x_0) \\|_{M^{-1}}
      = \\min_{\\gamma\\in\\mathbb{C}}
        \\| M M_l (b - A \\gamma x_0) \\|_{M^{-1}}
      \\leq \\| M M_l b \\|_{M^{-1}}.

    This is achieved by
    :math:`\\gamma_{\\min} = \\frac{\\langle z, M M_l b \\rangle_{M^{-1}}}{\\|z\\|_{M^{-1}}^2}` for
    :math:`z=M M_l A x_0` because then :math:`r_0=P_{z^\\perp}b`. (Note that
    the right hand side of formula (5.8.16) in [LieS13]_ has to be complex
    conjugated.)

    The parameters are the parameters you want to pass to
    :py:meth:`~krylov.linear_system.gmres`,
    :py:meth:`~krylov.linear_system.minres` or
    :py:meth:`~krylov.linear_system.cg`.

    :return: the adapted initial guess with the above property.
    """
    Ax0 = A @ x0

    MlAx0 = Ax0 if Ml is None else Ml @ Ax0
    z = MlAx0 if M is None else M @ MlAx0
    znorm2 = inner(z, MlAx0)
    if znorm2 <= 1e-15:
        return numpy.zeros_like(b)
    Mlb = b if Ml is None else Ml @ b
    gamma = inner(z, Mlb) / znorm2
    return gamma * x0


def strakos(n, l_min=0.1, l_max=100, rho=0.9):
    """Return the Strakoš matrix.

    See [Str92]_.
    """
    d = [
        l_min + (i - 1) * 1.0 / (n - 1) * (l_max - l_min) * (rho ** (n - i))
        for i in range(1, n + 1)
    ]
    return numpy.diag(d)


def gap(lamda, sigma, mode="individual"):
    """Compute spectral gap.

    Useful for eigenvalue/eigenvector bounds. Computes the gap
    :math:`\\delta\\geq 0` between two sets of real numbers ``lamda`` and
    ``sigma``. The gap can be computed in several ways and may not exist, see
    the ``mode`` parameter.

    :param lamda: a non-empty set
      :math:`\\Lambda=\\{\\lambda_1,\\ldots,\\lambda_n\\}` given as a single real
      number or a list or ``numpy.array`` with real numbers.
    :param sigma: a non-empty set :math:`\\Sigma=\\{\\sigma_1,\\ldots,\\sigma_m\\}`.
      See ``lamda``.
    :param mode: (optional). Defines how the gap should be computed. May be one
      of

      * ``'individual'`` (default):
        :math:`\\delta=\\min_{\\substack{i\\in\\{1,\\ldots,n\\}\\\\j\\in\\{1,\\ldots,m\\}}} |\\lambda_i - \\sigma_j|`.
        With this mode, the gap is always be defined.
      * ``'interval'``: determine the maximal :math:`\\delta` such that
        :math:`\\Sigma\\subset\\mathbb{R}\\setminus[\\min_{\\lambda\\in\\Lambda}\\lambda-\\delta,\\max_{\\lambda\\in\\Lambda}\\lambda+\\delta]`.
        If the gap does not exists, ``None`` is returned.

    :return: :math:`\\delta` or ``None``.
    """
    # sanitize input
    if numpy.isscalar(lamda):
        lamda = [lamda]
    lamda = numpy.array(lamda)
    if numpy.isscalar(sigma):
        sigma = [sigma]
    sigma = numpy.array(sigma)

    if not numpy.isreal(lamda).all() or not numpy.isreal(sigma).all():
        raise ArgumentError("complex spectra not yet implemented")

    if mode == "individual":
        return numpy.min(
            numpy.abs(
                numpy.reshape(lamda, (len(lamda), 1))
                - numpy.reshape(sigma, (1, len(sigma)))
            )
        )
    elif mode == "interval":
        lamda_min, lamda_max = numpy.min(lamda), numpy.max(lamda)
        # determine all values in sigma<lamda_min or >lamda_max
        sigma_lo = sigma <= lamda_min
        sigma_hi = sigma >= lamda_max
        # is a sigma value in lamda interval?
        if not numpy.all(sigma_lo + sigma_hi):
            return None
        delta = numpy.Infinity
        if numpy.any(sigma_lo):
            delta = lamda_min - numpy.max(sigma[sigma_lo])
        if numpy.any(sigma_hi):
            delta = numpy.min([delta, numpy.min(sigma[sigma_hi]) - lamda_max])

        return delta


class Interval:
    """A closed interval on the real line (may also be a single point)."""

    def __init__(self, left, right=None):
        if right is None:
            right = left
        if left > right:
            raise ArgumentError("left > right not allowed.")
        self.left = left
        self.right = right

    def __and__(self, other):
        """Return intersection interval or None"""
        left = numpy.max([self.left, other.left])
        right = numpy.min([self.right, other.right])
        if left <= right:
            return Interval(left, right)
        return None

    def __or__(self, other):
        """Return union of intervals if they intersect or None."""
        if self & other:
            left = numpy.min([self.left, other.left])
            right = numpy.max([self.right, other.right])
            return Interval(left, right)
        return None

    def __repr__(self):
        return f"[{self.left},{self.right}]"

    def contains(self, alpha):
        """Returns True if alpha is an element of the interval."""
        return self.left <= alpha and alpha <= self.right

    def distance(self, other):
        """Returns the distance to other (0 if intersection is nonempty)."""
        if self & other:
            return 0
        return numpy.max([other.left - self.right, self.left - other.right])


class Intervals:
    """A set of non-intersecting intervals."""

    def __init__(self, intervals=None):
        self.intervals = set()
        if intervals is not None:
            for interval in intervals:
                self.add(interval)

    def add(self, new):
        intersecting = {interval for interval in self.intervals if interval & new}
        # compute union
        for interval in intersecting:
            new = new | interval
        self.intervals = self.intervals.difference(intersecting)
        self.intervals.add(new)

    def contains(self, alpha):
        for interval in self.intervals:
            if interval.contains(alpha):
                return True
        return False

    def get_endpoints(self):
        endpoints = []
        for interval in self.intervals:
            if interval.left == interval.right:
                endpoints += [interval.left]
            else:
                endpoints += [interval.left, interval.right]
        endpoints.sort()
        return endpoints

    def __len__(self):
        return len(self.intervals)

    def __iter__(self):
        return self.intervals.__iter__()

    def __repr__(self):
        return ", ".join([interval.__repr__() for interval in self.intervals])

    def min(self):
        if self.__len__() == 0:
            return ArgumentError("empty set has no minimum.")
        return numpy.min(list(map(lambda i: i.left, self.intervals)))

    def max(self):
        if self.__len__() == 0:
            return ArgumentError("empty set has no maximum.")
        return numpy.max(list(map(lambda i: i.right, self.intervals)))

    def min_pos(self):
        """Returns minimal positive value or None."""
        if self.__len__() == 0:
            return ArgumentError("empty set has no minimum positive value.")
        if self.contains(0):
            return None
        positive = [interval for interval in self.intervals if interval.left > 0]
        if len(positive) == 0:
            return None
        return numpy.min(list(map(lambda i: i.left, positive)))

    def max_neg(self):
        """Returns maximum negative value or None."""
        if self.__len__() == 0:
            return ArgumentError("empty set has no maximum negative value.")
        if self.contains(0):
            return None
        negative = [interval for interval in self.intervals if interval.right < 0]
        if len(negative) == 0:
            return None
        return numpy.max(list(map(lambda i: i.right, negative)))

    def max_abs(self):
        """Returns maximum absolute value."""
        if self.__len__() == 0:
            return ArgumentError("empty set has no maximum absolute value.")
        return numpy.max(numpy.abs([self.max(), self.min()]))


class NormalizedRootsPolynomial:
    def __init__(self, roots):
        r"""A polynomial with specified roots and p(0)=1.

        Represents the polynomial

        .. math::

            p(\lambda) = \prod_{i=1}^n \left(1-\frac{\lambda}{\theta_i}\right).

        :param roots: array with roots :math:`\theta_1,\dots,\theta_n` of the
          polynomial and ``roots.shape==(n,)``.
        """
        # check input
        roots = numpy.asarray(roots)
        if len(roots.shape) != 1:
            raise ArgumentError("one-dimensional array of roots expected.")
        self.roots = roots

    def minmax_candidates(self):
        """Get points where derivative is zero.

        Useful for computing the extrema of the polynomial over an interval if
        the polynomial has real roots. In this case, the maximum is attained
        for one of the interval endpoints or a point from the result of this
        function that is contained in the interval.
        """
        from numpy.polynomial import Polynomial as P

        p = P.fromroots(self.roots)
        return p.deriv(1).roots()

    def __call__(self, points):
        """Evaluate polyonmial at given points.

        :param points: a point :math:`x` or array of points
          :math:`x_1,\\dots,x_m` with ``points.shape==(m,)``.
        :returns: :math:`p(x)` or array of shape ``(m,)`` with
          :math:`p(x_1),\\dots,p(x_m)`.
        """
        # check input
        p = numpy.asarray(points)
        if len(p.shape) > 1:
            raise ArgumentError("scalar or one-dimensional array of points expected.")
        n = self.roots.shape[0]
        vals = 1 - p / self.roots.reshape(n, 1)

        # prevent under/overflow by multiplying interlaced large and small
        # values
        for j in range(vals.shape[1]):
            sort_tmp = numpy.argsort(numpy.abs(vals[:, j]))
            sort = numpy.zeros((n,), dtype=numpy.int)
            mid = int(numpy.ceil(float(n) / 2))
            sort[::2] = sort_tmp[:mid]
            sort[1::2] = sort_tmp[mid:][::-1]
            vals[:, j] = vals[sort, j]

        # form product of each column
        vals = numpy.prod(vals, axis=0)

        if numpy.isscalar(points):
            return vals.item()
        return vals


def get_residual_norms(H, self_adjoint=False):
    """Compute relative residual norms from Hessenberg matrix.

    It is assumed that the initial guess is chosen as zero."""
    H = H.copy()
    n_, n = H.shape
    y = numpy.eye(n_, 1, dtype=H.dtype)
    resnorms = [1.0]
    for i in range(n_ - 1):
        G = givens(H[i : i + 2, [i]])
        if self_adjoint:
            H[i : i + 2, i : i + 3] = G.apply(H[i : i + 2, i : i + 3])
        else:
            H[i : i + 2, i:] = G.apply(H[i : i + 2, i:])
        y[i : i + 2] = G @ y[i : i + 2]
        resnorms.append(numpy.abs(y[i + 1, 0]))
    if n_ == n:
        resnorms.append(0.0)
    return numpy.array(resnorms)
