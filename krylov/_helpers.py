from collections import namedtuple


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
    "KrylovInfo", ["success", "xk", "resnorms", "errnorms", "num_operations", "arnoldi"]
)
