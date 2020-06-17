import numpy
from scipy.sparse import isspmatrix
from scipy.sparse.sputils import isintlike

from .errors import ArgumentError, LinearOperatorError


def get_linearoperator(shape, A):
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

    # check shape
    if shape != ret.shape:
        raise LinearOperatorError("shape mismatch")

    return ret


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
