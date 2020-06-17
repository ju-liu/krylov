"""
Collection of standard functions.

This method provides functions like inner products, norms, ...
"""
import time
import warnings
from collections import defaultdict

import numpy
import scipy.linalg

# for Givens rotations
from scipy.sparse import isspmatrix

# from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.sputils import isintlike

from .errors import ArgumentError, InnerProductError, LinearOperatorError
from .givens import Givens

__all__ = [
    "IdentityLinearOperator",
    "LinearOperator",
    "MatrixLinearOperator",
    "NormalizedRootsPolynomial",
    "Projection",
    "Timer",
    "angles",
    "bound_perturbed_gmres",
    "gap",
    "get_linearoperator",
    "hegedus",
    "inner",
    "ip_euclid",
    "norm",
    "norm_MMlr",
    "norm_squared",
    "orthonormality",
    "qr",
    "shape_vec",
    "shape_vecs",
    "strakos",
]


def find_common_dtype(*args):
    """Returns common dtype of numpy and scipy objects.

    Recognizes ndarray, spmatrix and LinearOperator. All other objects are
    ignored (most notably None)."""
    dtypes = []
    for arg in args:
        if (
            type(arg) is numpy.ndarray
            or isspmatrix(arg)
            or isinstance(arg, LinearOperator)
        ):
            if hasattr(arg, "dtype"):
                dtypes.append(arg.dtype)
            else:
                warnings.warn("object %s does not have a dtype." % arg.__repr__)
    return numpy.find_common_type(dtypes, [])


def shape_vec(x):
    """Take a (n,) ndarray and return it as (n,1) ndarray."""
    return numpy.reshape(x, (x.shape[0], 1))


def shape_vecs(*args):
    """Reshape all ndarrays with ``shape==(n,)`` to ``shape==(n,1)``.

    Recognizes ndarrays and ignores all others."""
    ret_args = []
    flat_vecs = True
    for arg in args:
        if type(arg) is numpy.ndarray:
            if len(arg.shape) == 1:
                arg = shape_vec(arg)
            else:
                flat_vecs = False
        ret_args.append(arg)
    return flat_vecs, ret_args


def ip_euclid(X, Y):
    """Euclidean inner product.

    numpy.vdot only works for vectors and numpy.dot does not use the conjugate
    transpose.

    :param X: numpy array with ``shape==(N,m)``
    :param Y: numpy array with ``shape==(N,n)``

    :return: numpy array :math:`X^* Y` with ``shape==(m,n)``.
    """
    return numpy.dot(X.T.conj(), Y)


def inner(X, Y, ip_B=None):
    """Euclidean and non-Euclidean inner product.

    numpy.vdot only works for vectors and numpy.dot does not use the conjugate
    transpose.

    :param X: numpy array with ``shape==(N,m)``
    :param Y: numpy array with ``shape==(N,n)``
    :param ip_B: (optional) May be one of the following

        * ``None``: Euclidean inner product.
        * a self-adjoint and positive definite operator :math:`B` (as
          ``numpy.array`` or ``LinearOperator``). Then :math:`X^*B Y` is
          returned.
        * a callable which takes 2 arguments X and Y and returns
          :math:`\\langle X,Y\\rangle`.

    **Caution:** a callable should only be used if necessary. The choice
    potentially has an impact on the round-off behavior, e.g. of projections.

    :return: numpy array :math:`\\langle X,Y\\rangle` with ``shape==(m,n)``.
    """
    if ip_B is None or isinstance(ip_B, IdentityLinearOperator):
        return numpy.dot(X.T.conj(), Y)
    (N, m) = X.shape
    (_, n) = Y.shape
    try:
        B = get_linearoperator((N, N), ip_B)
    except TypeError:
        return ip_B(X, Y)
    if m > n:
        return numpy.dot((B * X).T.conj(), Y)
    else:
        return numpy.dot(X.T.conj(), B * Y)


def norm_squared(x, Mx=None, inner_product=ip_euclid):
    """Compute the norm^2 w.r.t. to a given scalar product."""
    assert len(x.shape) == 2
    if Mx is None:
        rho = inner_product(x, x)
    else:
        assert len(Mx.shape) == 2
        rho = inner_product(x, Mx)

    if rho.shape == (1, 1):
        if abs(rho[0, 0].imag) > abs(rho[0, 0]) * 1e-10 or rho[0, 0].real < 0.0:
            raise InnerProductError(
                ("<x,Mx> = %g. Is the inner product " "indefinite?") % rho[0, 0]
            )

    return numpy.linalg.norm(rho, 2)


def norm(x, y=None, ip_B=None):
    r"""Compute norm (Euclidean and non-Euclidean).

    :param x: a 2-dimensional ``numpy.array``.
    :param y: a 2-dimensional ``numpy.array``.
    :param ip_B: see :py:meth:`inner`.

    Compute :math:`\sqrt{\langle x,y\rangle}` where the inner product is
    defined via ``ip_B``.
    """
    # Euclidean inner product?
    if y is None and (ip_B is None or isinstance(ip_B, IdentityLinearOperator)):
        return numpy.linalg.norm(x, 2)
    if y is None:
        y = x
    ip = inner(x, y, ip_B=ip_B)
    nrm_diag = numpy.linalg.norm(numpy.diag(ip), 2)
    nrm_diag_imag = numpy.linalg.norm(numpy.imag(numpy.diag(ip)), 2)
    if nrm_diag_imag > nrm_diag * 1e-10:
        raise InnerProductError(
            "inner product defined by ip_B not positive "
            "definite? ||diag(ip).imag||/||diag(ip)||="
            f"{nrm_diag_imag/nrm_diag}"
        )
    return numpy.sqrt(numpy.linalg.norm(ip, 2))


def get_linearoperator(shape, A, timer=None):
    """Enhances aslinearoperator if A is None."""
    ret = None
    import scipy.sparse.linalg as scipylinalg

    if isinstance(A, LinearOperator):
        ret = A
    elif A is None:
        ret = IdentityLinearOperator(shape)
    elif isinstance(A, numpy.ndarray) or isspmatrix(A):
        ret = MatrixLinearOperator(A)
    elif isinstance(A, numpy.matrix):
        ret = MatrixLinearOperator(numpy.atleast_2d(numpy.asarray(A)))
    elif isinstance(A, scipylinalg.LinearOperator):
        if not hasattr(A, "dtype"):
            raise ArgumentError("scipy LinearOperator has no dtype.")
        ret = LinearOperator(A.shape, dot=A.matvec, dot_adj=A.rmatvec, dtype=A.dtype)
    else:
        raise TypeError("type not understood")

    # set up timer if requested
    if (
        A is not None
        and not isinstance(A, IdentityLinearOperator)
        and timer is not None
    ):
        ret = TimedLinearOperator(ret, timer)

    # check shape
    if shape != ret.shape:
        raise LinearOperatorError("shape mismatch")

    return ret


def norm_MMlr(M, Ml, A, Mr, b, x0, yk, inner_product=ip_euclid):
    xk = x0 + Mr * yk
    r = b - A * xk
    Mlr = Ml * r
    # normalize residual before applying the preconditioner here.
    # otherwise MMlr can become 0 exactly (pyamg doesnt respect relative
    # residual)
    # TODO for testing: 2-norm
    norm_Mlr = norm(Mlr)
    if norm_Mlr == 0:
        MMlr = numpy.zeros(Mlr.shape)
        norm_MMlr = 0
    else:
        nMlr = Mlr / norm_Mlr
        nMMlr = M * nMlr
        MMlr = nMMlr * norm_Mlr
        norm_MMlr = norm(Mlr, MMlr, inner_product=inner_product)
    # return xk and ||M*Ml*(b-A*(x0+Mr*yk))||_{M^{-1}}
    return xk, Mlr, MMlr, norm_MMlr


def orthonormality(V, ip_B=None):
    """Measure orthonormality of given basis.

    :param V: a matrix :math:`V=[v_1,\\ldots,v_n]` with ``shape==(N,n)``.
    :param ip_B: (optional) the inner product to use, see :py:meth:`inner`.

    :return: :math:`\\| I_n - \\langle V,V \\rangle \\|_2`.
    """
    return norm(numpy.eye(V.shape[1]) - inner(V, V, ip_B=ip_B))


class Projection(object):
    def __init__(self, X, Y=None, ip_B=None, orthogonalize=True, iterations=2):
        """Generic projection.

        This class can represent any projection (orthogonal and oblique)
        on a N-dimensional Hilbert space. A projection is a linear operator
        :math:`P` with :math:`P^2=P`. A projection is uniquely defined by its
        range :math:`\\mathcal{V}:=\\operatorname{range}(P)` and its kernel
        :math:`\\mathcal{W}:=\\operatorname{ker}(P)`; this projection is called
        :math:`P_{\\mathcal{V},\\mathcal{W}}`.

        Let X and Y be two full rank arrays with ``shape==(N,k)`` and let
        :math:`\\mathcal{X}\\oplus\\mathcal{Y}^\\perp=\\mathbb{C}^N` where
        :math:`\\mathcal{X}:=\\operatorname{colspan}(X)` and
        :math:`\\mathcal{Y}:=\\operatorname{colspan}(Y)`.
        Then this class constructs the projection
        :math:`P_{\\mathcal{X},\\mathcal{Y}^\\perp}`.
        The requirement
        :math:`\\mathcal{X}\\oplus\\mathcal{Y}^\\perp=\\mathbb{C}^N`
        is equivalent to ``\\langle X,Y\\rangle`` being nonsingular.

        :param X: array with ``shape==(N,k)`` and
            :math:`\\operatorname{rank}(X)=k`.
        :param Y: (optional) ``None`` or array with ``shape==(N,k)`` and
            :math:`\\operatorname{rank}(X)=k`. If Y is ``None`` then Y is
            set to X which means that the resulting projection is orthogonal.
        :param ip_B: (optional) inner product, see :py:meth:`inner`. ``None``,
            a ``numpy.array`` or a ``LinearOperator`` is preferred due to the
            applicability of the proposed algorithms in [Ste11]_, see below.
        :param orthogonalize: (optional) `True` orthogonalizes the bases
            provided in `X` and `Y` with respect to the inner product defined
            by `ip_B`. Defaults to `True` as the orthogonalization is
            suggested by the round-off error analysis in [Ste11]_.
        :param iterations: (optional) number of applications of the projection.
            It was suggested in [Ste11]_ to use 2 iterations (default) in order
            to achieve high accuracy ("twice is enough" as in the orthogonal
            case).

        This projection class makes use of the round-off error analysis of
        oblique projections in the work of Stewart [Ste11]_ and implements the
        algorithms that are considered as the most stable ones (e.g., the XQRY
        representation in [Ste11]_).
        """
        # check and store input
        self.ip_B = ip_B
        if iterations < 1:
            raise ArgumentError("iterations < 1 not allowed")
        self.orthogonalize = orthogonalize
        self.iterations = iterations

        Y = X if Y is None else Y  # default: orthogonal projection

        if len(X.shape) != 2:
            raise ArgumentError("X does not have shape==(N,k)")
        if X.shape != Y.shape:
            raise ArgumentError("X and Y have different shapes")

        # set N-by-zero vectors if input is N-by-zero
        # (results in zero operator)
        if X.shape[1] == 0:
            self.V = self.W = numpy.zeros(X.shape)
            self.VR = self.WR = self.Q = self.R = None
            return

        # orthogonalize X
        if orthogonalize:
            self.V, self.VR = qr(X, ip_B=ip_B)
        else:
            self.V = X
            self.VR = None

        if Y is X and orthogonalize:  # orthogonal projection
            self.W, self.WR = self.V, self.VR
            self.Q, self.R = None, None
        else:  # general case
            if orthogonalize:
                self.W, self.WR = qr(Y, ip_B=ip_B)
            else:
                self.W = Y
                self.WR = None
            M = inner(self.W, self.V, ip_B=ip_B)
            self.Q, self.R = scipy.linalg.qr(M)

    def _apply(self, a, return_Ya=False):
        r"""Single application of the projection.

        :param a: array with ``a.shape==(N,m)``.
        :param return_inner: (optional) should the inner product
          :math:`\langle Y,a\rangle` be returned?
        :return:

          * :math:`P_{\mathcal{X},\mathcal{Y}^\perp} a =
            X \langle Y,X\rangle^{-1} \langle Y, a\rangle`.
          * :math:`\langle Y,a\rangle` if ``return_inner==True``.
        """
        # is projection the zero operator?
        if self.V.shape[1] == 0:
            Pa = numpy.zeros(a.shape)
            if return_Ya:
                return Pa, numpy.zeros((0, a.shape[1]))
            return Pa
        c = inner(self.W, a, ip_B=self.ip_B)

        if return_Ya:
            Ya = c.copy()
            if self.WR is not None:
                Ya = self.WR.T.conj().dot(Ya)

        if self.Q is not None and self.R is not None:
            c = scipy.linalg.solve_triangular(self.R, self.Q.T.conj().dot(c))
        Pa = self.V.dot(c)
        if return_Ya:
            return Pa, Ya
        return Pa

    def _apply_adj(self, a):
        # is projection the zero operator?
        if self.V.shape[1] == 0:
            return numpy.zeros(a.shape)
        """Single application of the adjoint projection."""
        c = inner(self.V, a, ip_B=self.ip_B)
        if self.Q is not None and self.R is not None:
            c = self.Q.dot(
                scipy.linalg.solve_triangular(self.R.T.conj(), c, lower=True)
            )
        return self.W.dot(c)

    def apply(self, a, return_Ya=False):
        r"""Apply the projection to an array.

        The computation is carried out without explicitly forming the
        matrix corresponding to the projection (which would be an array with
        ``shape==(N,N)``).

        See also :py:meth:`_apply`.
        """
        # is projection the zero operator?
        if self.V.shape[1] == 0:
            Pa = numpy.zeros(a.shape)
            if return_Ya:
                return Pa, numpy.zeros((0, a.shape[1]))
            return Pa
        if return_Ya:
            x, Ya = self._apply(a, return_Ya=return_Ya)
        else:
            x = self._apply(a)
        for i in range(self.iterations - 1):
            z = a - x
            w = self._apply(z)
            x = x + w
        if return_Ya:
            return x, Ya
        return x

    def apply_adj(self, a):
        # is projection the zero operator?
        if self.V.shape[1] == 0:
            return numpy.zeros(a.shape)
        x = self._apply_adj(a)
        for i in range(self.iterations - 1):
            z = a - x
            w = self._apply_adj(z)
            x = x + w
        return x

    def apply_complement(self, a, return_Ya=False):
        """Apply the complementary projection to an array.

        :param z: array with ``shape==(N,m)``.

        :return: :math:`P_{\\mathcal{Y}^\\perp,\\mathcal{X}}z =
            z - P_{\\mathcal{X},\\mathcal{Y}^\\perp} z`.
        """
        # is projection the zero operator? --> complement is identity
        if self.V.shape[1] == 0:
            if return_Ya:
                return a.copy(), numpy.zeros((0, a.shape[1]))
            return a.copy()
        if return_Ya:
            x, Ya = self._apply(a, return_Ya=True)
        else:
            x = self._apply(a)
        z = a - x
        for i in range(self.iterations - 1):
            w = self._apply(z)
            z = z - w
        if return_Ya:
            return z, Ya
        return z

    def apply_complement_adj(self, a):
        # is projection the zero operator? --> complement is identity
        if self.V.shape[1] == 0:
            return a.copy()
        x = self._apply_adj(a)
        z = a - x
        for i in range(self.iterations - 1):
            w = self._apply_adj(z)
            z = z - w
        return z

    def _get_operator(self, fun, fun_adj):
        N = self.V.shape[0]
        t = numpy.find_common_type([self.V.dtype, self.W.dtype], [])
        return LinearOperator((N, N), t, fun, fun_adj)

    def operator(self):
        """Get a ``LinearOperator`` corresponding to apply().

        :return: a LinearOperator that calls apply().
        """
        # is projection the zero operator?
        if self.V.shape[1] == 0:
            N = self.V.shape[0]
            return ZeroLinearOperator((N, N))
        return self._get_operator(self.apply, self.apply_adj)

    def operator_complement(self):
        """Get a ``LinearOperator`` corresponding to apply_complement().

        :return: a LinearOperator that calls apply_complement().
        """
        # is projection the zero operator? --> complement is identity
        if self.V.shape[1] == 0:
            N = self.V.shape[0]
            return IdentityLinearOperator((N, N))
        return self._get_operator(self.apply_complement, self.apply_complement_adj)

    def matrix(self):
        """Builds matrix representation of projection.

        Builds the matrix representation
        :math:`P = X \\langle Y,X\\rangle^{-1} \\langle Y, I_N\\rangle`.

        **Use with care!** This routine may be helpful for testing purposes but
        should not be used in production codes for high dimensions since
        the resulting matrix is dense.
        """
        return self.apply(numpy.eye(self.V.shape[0]))


def qr(X, ip_B=None, reorthos=1):
    """QR factorization with customizable inner product.

    :param X: array with ``shape==(N,k)``
    :param ip_B: (optional) inner product, see :py:meth:`inner`.
    :param reorthos: (optional) numer of reorthogonalizations. Defaults to
      1 (i.e. 2 runs of modified Gram-Schmidt) which should be enough in most
      cases (TODO: add reference).

    :return: Q, R where :math:`X=QR` with :math:`\\langle Q,Q \\rangle=I_k` and
      R upper triangular.
    """
    if ip_B is None and X.shape[1] > 0:
        return scipy.linalg.qr(X, mode="economic")
    else:
        (N, k) = X.shape
        Q = X.copy()
        R = numpy.zeros((k, k), dtype=X.dtype)
        for i in range(k):
            for reortho in range(reorthos + 1):
                for j in range(i):
                    alpha = inner(Q[:, [j]], Q[:, [i]], ip_B=ip_B)[0, 0]
                    R[j, i] += alpha
                    Q[:, [i]] -= alpha * Q[:, [j]]
            R[i, i] = norm(Q[:, [i]], ip_B=ip_B)
            if R[i, i] >= 1e-15:
                Q[:, [i]] /= R[i, i]
        return Q, R


def angles(F, G, ip_B=None, compute_vectors=False):
    """Principal angles between two subspaces.

    This algorithm is based on algorithm 6.2 in `Knyazev, Argentati. Principal angles
    between subspaces in an A-based scalar product: algorithms and perturbation
    estimates. 2002.` This algorithm can also handle small angles (in contrast to the
    naive cosine-based svd algorithm).

    :param F: array with ``shape==(N,k)``.
    :param G: array with ``shape==(N,l)``.
    :param ip_B: (optional) angles are computed with respect to this
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

    QF, _ = qr(F, ip_B=ip_B)
    QG, _ = qr(G, ip_B=ip_B)

    # one or both matrices empty? (enough to check G here)
    if G.shape[1] == 0:
        theta = numpy.ones(F.shape[1]) * numpy.pi / 2
        U = QF
        V = QG
    else:
        Y, s, Z = scipy.linalg.svd(inner(QF, QG, ip_B=ip_B))
        Vcos = numpy.dot(QG, Z.T.conj())
        n_large = numpy.flatnonzero((s ** 2) < 0.5).shape[0]
        n_small = s.shape[0] - n_large
        theta = numpy.r_[
            numpy.arccos(s[n_small:]),  # [-i:] does not work if i==0
            numpy.ones(F.shape[1] - G.shape[1]) * numpy.pi / 2,
        ]
        if compute_vectors:
            Ucos = numpy.dot(QF, Y)
            U = Ucos[:, n_small:]
            V = Vcos[:, n_small:]

        if n_small > 0:
            RG = Vcos[:, :n_small]
            S = RG - numpy.dot(QF, inner(QF, RG, ip_B=ip_B))
            _, R = qr(S, ip_B=ip_B)
            Y, u, Z = scipy.linalg.svd(R)
            theta = numpy.r_[numpy.arcsin(u[::-1][:n_small]), theta]
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
                U = numpy.c_[Usin, U]
                V = numpy.c_[Vsin, V]

    if compute_vectors:
        if reverse:
            U, V = V, U
        return theta, U, V
    else:
        return theta


def hegedus(A, b, x0, M=None, Ml=None, ip_B=None):
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
    :py:meth:`~krylov.linsys.gmres`,
    :py:meth:`~krylov.linsys.minres` or
    :py:meth:`~krylov.linsys.cg`.

    :return: the adapted initial guess with the above property.
    """
    N = len(b)
    shape = (N, N)
    A = get_linearoperator(shape, A)
    M = get_linearoperator(shape, M)
    Ml = get_linearoperator(shape, Ml)

    MlAx0 = Ml * (A * x0)
    z = M * MlAx0
    znorm2 = inner(z, MlAx0, ip_B=ip_B)
    if znorm2 <= 1e-15:
        return numpy.zeros((N, 1))
    gamma = inner(z, Ml * b, ip_B=ip_B) / znorm2
    return gamma * x0


class Timer(list):
    """Measure execution time of multiple code blocks with ``with``.

    Example: ::

        t = Timer()
        with t:
            print('time me!')
        print('don\\\'t time me!')
        with t:
            print('time me, too!')
        print(t)

    Result: ::

        time me!
        don't time me!
        time me, too!
        [6.389617919921875e-05, 6.008148193359375e-05]

    """

    def __init__(self):
        super(Timer, self).__init__()

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, a, b, c):
        self.append(time.time() - self.tstart)


class Timings(defaultdict):
    """Manages several timers.

    If you want to measure different types of code blocks you can use ::

        tm = Timings()
        with tm['class1']:
            print('code that belongs to class1')
        with tm['class2']:
            print('code that belongs to class2')
        with tm['class1']:
            print('code that belongs to class1')
        print(tm)

    Result: ::

        code that belongs to class1
        code that belongs to class2
        code that belongs to class1
        defaultdict(<class krylov.utils.Timer at 0x23810b8>, {'class2': [2.9087066650390625e-05], 'class1': [4.696846008300781e-05, 3.2901763916015625e-05]})
    """

    def __init__(self):
        super(Timings, self).__init__(Timer)

    def get(self, key):
        """Return timings for `key`. Returns 0 if not present."""
        if key in self and len(self[key]) > 0:
            return min(self[key])
        else:
            return 0

    def get_ops(self, ops):
        """Return timings for dictionary ops holding the operation names as
        keys and the number of applications as values."""
        time = 0.0
        for op, count in ops.items():
            time += self.get(op) * count
        return time

    def __repr__(self):
        return "Timings(" + ", ".join([f"{key}: {self.get(key)}" for key in self]) + ")"


class LinearOperator(object):
    """Linear operator.

    Is partly based on the LinearOperator from scipy (BSD License).
    """

    def __init__(self, shape, dtype, dot=None, dot_adj=None):
        if len(shape) != 2 or not isintlike(shape[0]) or not isintlike(shape[1]):
            raise LinearOperatorError("shape must be (m,n) with m and n " "integer")
        self.shape = shape
        self.dtype = numpy.dtype(dtype)  # defaults to float64
        if dot is None and dot_adj is None:
            raise LinearOperatorError("dot or dot_adj have to be defined")
        self._dot = dot
        self._dot_adj = dot_adj

    def dot(self, X):
        X = numpy.asanyarray(X)
        m, n = self.shape
        if X.shape[0] != n:
            raise LinearOperatorError("dimension mismatch")
        if self._dot is None:
            raise LinearOperatorError("dot undefined")
        if X.shape[1] == 0:
            return numpy.zeros(X.shape)
        return self._dot(X)

    def dot_adj(self, X):
        X = numpy.asanyarray(X)
        m, n = self.shape
        if X.shape[0] != m:
            raise LinearOperatorError("dimension mismatch")
        if self._dot_adj is None:
            raise LinearOperatorError("dot_adj undefined")
        if X.shape[1] == 0:
            return numpy.zeros(X.shape)
        return self._dot_adj(X)

    @property
    def adj(self):
        return _AdjointLinearOperator(self)

    def __mul__(self, X):
        try:
            if isinstance(X, IdentityLinearOperator):
                return self
            elif isinstance(self, IdentityLinearOperator):
                return X
            elif isinstance(X, LinearOperator):
                return _ProductLinearOperator(self, X)
            elif numpy.isscalar(X):
                return _ScaledLinearOperator(self, X)
            else:
                return self.dot(X)
        except LinearOperatorError:
            return NotImplemented

    def __rmul__(self, X):
        try:
            return _ScaledLinearOperator(self, X)
        except LinearOperatorError:
            return NotImplemented

    def __pow__(self, X):
        try:
            return _PowerLinearOperator(self, X)
        except LinearOperatorError:
            return NotImplemented

    def __add__(self, X):
        try:
            return _SumLinearOperator(self, X)
        except LinearOperatorError:
            return NotImplemented

    def __neg__(self):
        try:
            return _ScaledLinearOperator(self, -1)
        except LinearOperatorError:
            return NotImplemented

    def __sub__(self, X):
        return self + (-X)

    def __repr__(self):
        m, n = self.shape
        return "<%dx%d %s with dtype=%s>" % (
            m,
            n,
            self.__class__.__name__,
            str(self.dtype),
        )


def _get_dtype(operators, dtypes=None):
    if dtypes is None:
        dtypes = []
    for obj in operators:
        if obj is not None and hasattr(obj, "dtype"):
            dtypes.append(obj.dtype)
    return numpy.find_common_type(dtypes, [])


class _SumLinearOperator(LinearOperator):
    def __init__(self, A, B):
        if not isinstance(A, LinearOperator) or not isinstance(B, LinearOperator):
            raise LinearOperatorError("both operands have to be a " "LinearOperator")
        if A.shape != B.shape:
            raise LinearOperatorError("shape mismatch")
        self.args = (A, B)
        super(_SumLinearOperator, self).__init__(
            A.shape, _get_dtype([A, B]), self._dot, self._dot_adj
        )

    def _dot(self, X):
        return self.args[0].dot(X) + self.args[1].dot(X)

    def _dot_adj(self, X):
        return self.args[0].dot_adj(X) + self.args[1].dot_adj(X)


class _ProductLinearOperator(LinearOperator):
    def __init__(self, A, B):
        if not isinstance(A, LinearOperator) or not isinstance(B, LinearOperator):
            raise LinearOperatorError("both operands have to be a " "LinearOperator")
        if A.shape[1] != B.shape[0]:
            raise LinearOperatorError("shape mismatch")
        self.args = (A, B)
        super(_ProductLinearOperator, self).__init__(
            (A.shape[0], B.shape[1]), _get_dtype([A, B]), self._dot, self._dot_adj
        )

    def _dot(self, X):
        return self.args[0].dot(self.args[1].dot(X))

    def _dot_adj(self, X):
        return self.args[1].dot_adj(self.args[0].dot_adj(X))


class _ScaledLinearOperator(LinearOperator):
    def __init__(self, A, alpha):
        if not isinstance(A, LinearOperator):
            raise LinearOperatorError("LinearOperator expected as A")
        if not numpy.isscalar(alpha):
            raise LinearOperatorError("scalar expected as alpha")
        self.args = (A, alpha)
        super(_ScaledLinearOperator, self).__init__(
            A.shape, _get_dtype([A], [type(alpha)]), self._dot, self._dot_adj
        )

    def _dot(self, X):
        return self.args[1] * self.args[0].dot(X)

    def _dot_adj(self, X):
        return numpy.conj(self.args[1]) * self.args[0].dot_adj(X)


class _PowerLinearOperator(LinearOperator):
    def __init__(self, A, p):
        if not isinstance(A, LinearOperator):
            raise LinearOperatorError("LinearOperator expected as A")
        if A.shape[0] != A.shape[1]:
            raise LinearOperatorError("square LinearOperator expected as A")
        if not isintlike(p):
            raise LinearOperatorError("integer expected as p")
        self.args = (A, p)
        super(_PowerLinearOperator, self).__init__(
            A.shape, A.dtype, self._dot, self._dot_adj
        )

    def _power(self, fun, X):
        res = X.copy()
        for i in range(self.args[1]):
            res = fun(res)
        return res

    def _dot(self, X):
        return self._power(self.args[0].dot, X)

    def _dot_adj(self, X):
        return self._power(self.args[0]._dot_adj, X)


class _AdjointLinearOperator(LinearOperator):
    def __init__(self, A):
        if not isinstance(A, LinearOperator):
            raise LinearOperatorError("LinearOperator expected as A")
        self.args = (A,)
        m, n = A.shape
        super(_AdjointLinearOperator, self).__init__(
            (n, m), A.dtype, A._dot_adj, A._dot
        )


class IdentityLinearOperator(LinearOperator):
    def __init__(self, shape):
        super(IdentityLinearOperator, self).__init__(
            shape, numpy.dtype(None), self._dot, self._dot_adj
        )

    def _dot(self, X):
        return X

    def _dot_adj(self, X):
        return X


class ZeroLinearOperator(LinearOperator):
    def __init__(self, shape):
        super(ZeroLinearOperator, self).__init__(
            shape, numpy.dtype(None), self._dot, self._dot_adj
        )

    def _dot(self, X):
        return numpy.zeros(X.shape)

    def _dot_adj(self, X):
        return numpy.zeros(X.shape)


class MatrixLinearOperator(LinearOperator):
    def __init__(self, A):
        super(MatrixLinearOperator, self).__init__(
            A.shape, A.dtype, self._dot, self._dot_adj
        )
        self._A = A
        self._A_adj = None

    def _dot(self, X):
        return self._A.dot(X)

    def _dot_adj(self, X):
        if self._A_adj is None:
            self._A_adj = self._A.T.conj()
        return self._A_adj.dot(X)

    def __repr__(self):
        return self._A.__repr__()


class TimedLinearOperator(LinearOperator):
    def __init__(self, linear_operator, timer=None):
        self._linear_operator = linear_operator
        super(TimedLinearOperator, self).__init__(
            shape=linear_operator.shape,
            dtype=linear_operator.dtype,
            dot=linear_operator.dot,
            dot_adj=linear_operator.dot_adj,
        )

        if timer is None:
            timer = Timer()
        self._timer = timer

    def dot(self, X):
        k = X.shape[1]
        if k == 0:
            return self._linear_operator.dot(X)
        with self._timer:
            ret = self._linear_operator.dot(X)
        self._timer[-1] /= k
        return ret

    def dot_adj(self, X):
        k = X.shape[1]
        if k == 0:
            return self._linear_operator.dot(X)
        k = X.shape[1]
        with self._timer:
            ret = self._linear_operator.dot_adj(X)
        self._timer[-1] /= k
        return ret


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


class Interval(object):
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


class Intervals(object):
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

    def min_abs(self):
        """Returns minimum absolute value."""
        if self.__len__() == 0:
            return ArgumentError("empty set has no minimum absolute value.")
        if self.contains(0):
            return 0
        return numpy.min(
            [
                numpy.abs(val)
                for val in [self.max_neg(), self.min_pos()]
                if val is not None
            ]
        )

    def max_abs(self):
        """Returns maximum absolute value."""
        if self.__len__() == 0:
            return ArgumentError("empty set has no maximum absolute value.")
        return numpy.max(numpy.abs([self.max(), self.min()]))


def bound_perturbed_gmres(pseudo, p, epsilon, deltas):
    """Compute GMRES perturbation bound based on pseudospectrum

    Computes the GMRES bound from [SifEM13]_.
    """
    if not numpy.all(numpy.array(deltas) > epsilon):
        raise ArgumentError("all deltas have to be greater than epsilon")

    bound = []
    for delta in deltas:
        # get boundary paths
        paths = pseudo.contour_paths(delta)

        # get vertices on boundary
        vertices = paths.vertices()

        # evaluate polynomial
        supremum = numpy.max(numpy.abs(p(vertices)))

        # compute bound
        bound.append(
            epsilon
            / (delta - epsilon)
            * paths.length()
            / (2 * numpy.pi * delta)
            * supremum
        )
    return bound


class NormalizedRootsPolynomial(object):
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
            raise ArgumentError(
                "scalar or one-dimensional array of points " "expected."
            )
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
        G = Givens(H[i : i + 2, [i]])
        if self_adjoint:
            H[i : i + 2, i : i + 3] = G.apply(H[i : i + 2, i : i + 3])
        else:
            H[i : i + 2, i:] = G.apply(H[i : i + 2, i:])
        y[i : i + 2] = G.apply(y[i : i + 2])
        resnorms.append(numpy.abs(y[i + 1, 0]))
    if n_ == n:
        resnorms.append(0.0)
    return numpy.array(resnorms)
