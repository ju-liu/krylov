from numpy.typing import ArrayLike

from ._helpers import Info, LinearOperator, aslinearoperator
from .cg import cg


class A_AH:
    def __init__(self, A: LinearOperator):
        self.shape = A.shape
        self.A = A
        self.dtype = A.dtype

    def __matmul__(self, x: ArrayLike):
        return self.A @ self.A.rmatvec(x)


def cgne(A: LinearOperator, b: ArrayLike, *args, **kwargs):
    r"""Conjugate Gradient Method on the Normal Equations

    AA^H y = b
    x = A^H y
    """
    A = aslinearoperator(A)

    sol, info = cg(A_AH(A), b, *args, **kwargs)

    xk = A.rmatvec(info.xk)

    if sol is not None:
        sol = xk

    info = Info(
        info.success,
        xk,
        info.numsteps,
        info.resnorms,
        info.num_operations,
        info.arnoldi,
    )

    return sol, info
