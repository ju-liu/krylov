from ._helpers import Info, aslinearoperator
from .cg import cg


class A_AH:
    def __init__(self, A):
        self.shape = A.shape
        self.A = A

    def __matmul__(self, x):
        return self.A @ self.A.rmatvec(x)


def cgne(A, b, *args, **kwargs):
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
        info.errnorms,
        info.num_operations,
        info.arnoldi,
    )

    return sol, info
