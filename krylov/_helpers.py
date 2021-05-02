from collections import namedtuple

import numpy as np


class Identity:
    def __matmul__(self, x):
        return x


class Product:
    def __init__(self, *operators):
        self.operators = operators

    def __matmul__(self, x):
        out = x.copy()
        for op in self.operators[::-1]:
            out = op @ out
        return out


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
