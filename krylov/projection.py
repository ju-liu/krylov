import numpy
import scipy

from .errors import ArgumentError
from .utils import IdentityLinearOperator, LinearOperator, ZeroLinearOperator, inner, qr


class Projection(object):
    def __init__(self, X, Y=None, ip_B=None, orthogonalize=True, iterations=2):
        """Generic projection.

        This class can represent any projection (orthogonal and oblique) on a
        N-dimensional Hilbert space. A projection is a linear operator :math:`P` with
        :math:`P^2=P`. A projection is uniquely defined by its range
        :math:`\\mathcal{V}:=\\operatorname{range}(P)` and its kernel
        :math:`\\mathcal{W}:=\\operatorname{ker}(P)`; this projection is called
        :math:`P_{\\mathcal{V},\\mathcal{W}}`.

        Let X and Y be two full rank arrays with ``shape==(N,k)`` and let
        :math:`\\mathcal{X}\\oplus\\mathcal{Y}^\\perp=\\mathbb{C}^N` where
        :math:`\\mathcal{X}:=\\operatorname{colspan}(X)` and
        :math:`\\mathcal{Y}:=\\operatorname{colspan}(Y)`.  Then this class constructs
        the projection :math:`P_{\\mathcal{X},\\mathcal{Y}^\\perp}`.  The requirement
        :math:`\\mathcal{X}\\oplus\\mathcal{Y}^\\perp=\\mathbb{C}^N` is equivalent to
        ``\\langle X,Y\\rangle`` being nonsingular.

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

        This projection class makes use of the round-off error analysis of oblique
        projections in the work of Stewart [Ste11]_ and implements the algorithms that
        are considered as the most stable ones (e.g., the XQRY representation in
        [Ste11]_).
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
