from ._helpers import aslinearoperator
from .cg import cg


class AH_A:
    def __init__(self, A):
        self.shape = A.shape
        self.A = A
        self.dtype = A.dtype

    def __matmul__(self, x):
        return self.A.rmatvec(self.A @ x)


def cgnr(A, b, *args, **kwargs):
    r"""Conjugate Gradient Method on the Normal Equations

    A^H A x = A^H b
    """
    A = aslinearoperator(A)
    return cg(AH_A(A), A.rmatvec(b), *args, **kwargs)
