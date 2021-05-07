from collections import namedtuple

import numpy as np


class Identity:
    @staticmethod
    def __matmul__(x):
        return x

    @staticmethod
    def rmatvec(x):
        return x


class Product:
    def __init__(self, *operators):
        self.operators = operators

    def __matmul__(self, x):
        out = x.copy()
        for op in self.operators[::-1]:
            out = op @ out
        return out


class LinearOperatorWrapper:
    """Provides rmatvec."""

    def __init__(self, array):
        self._array = array
        self._adj_array = None
        self.shape = array.shape

    def __matmul__(self, x):
        return self._array @ x

    matvec = __matmul__

    def rmatvec(self, x):
        """Performs the operation y = A^H @ x."""
        # For dense matrices, this takes a lot of memory and the above gist analysis
        # suggests that caching isn't faster.
        # <https://gist.github.com/nschloe/eb3bd2520cdbb1378c14887d56c031a2>
        if isinstance(self._array, np.ndarray):
            return (self._array.T @ x.conj()).conj()

        # For the rest, just cache it
        if self._adj_array is None:
            self._adj_array = self._array.T.conj()
        return self._adj_array @ x


def aslinearoperator(A):
    if hasattr(A, "__matmul__") and hasattr(A, "rmatvec"):
        return A

    if hasattr(A, "__matmul__") and not hasattr(A, "rmatvec"):
        return LinearOperatorWrapper(A)

    raise ValueError(f"Unknown linear operator A = {A}")


Info = namedtuple(
    "KrylovInfo",
    ["success", "xk", "numsteps", "resnorms", "errnorms", "num_operations", "arnoldi"],
)


def get_inner(b_shape):
    # np.dot is faster than einsum for flat vectors
    # <https://gist.github.com/nschloe/33b3c93b9bc0768394ba9edee1fda2bc>
    def inner_dot(x, y):
        return np.dot(x.conj(), y)

    def inner_einsum(x, y):
        return np.einsum("i...,i...->...", x.conj(), y)

    return inner_dot if len(b_shape) == 1 else inner_einsum
