import numpy as np
import pytest

import krylov

from .helpers import assert_consistent
from .linear_problems import (
    hermitian_indefinite,
    hpd,
    spd_dense,
    spd_sparse,
    symmetric_indefinite,
)


def test_ref():
    A, b = symmetric_indefinite()
    _, info = krylov.symmlq(A, b, maxiter=10)

    ref = [
        2.23606797749979,
        0.9823441352194251,
        0.5792270481089666,
        0.2307060320183781,
        0.16833036914998076,
        2.5918417478740246e-15,
    ]
    print(ref)
    print(info.resnorms)
    assert np.all(np.abs(info.resnorms - np.array(ref)) < 1.0e-13 * (1.0 + np.abs(ref)))


@pytest.mark.parametrize(
    "A_b",
    [
        spd_dense((5,)),
        spd_sparse((5,)),
        spd_sparse((5, 1)),
        spd_sparse((5, 3)),
        # spd_rhs_0((5,)),
        # spd_rhs_0sol0(),
        hpd(),
        symmetric_indefinite(),
        hermitian_indefinite(),
    ],
)
def test_symmlq(A_b):
    A, b = A_b
    print("A:")
    print(A)
    print()
    print("b:")
    print(b)
    print()
    callback_counter = 0

    def callback(x, r):
        nonlocal callback_counter
        callback_counter += 1

    sol, info = krylov.symmlq(A, b, tol=1.0e-7, maxiter=10, callback=callback)
    print("info:")
    print(info)
    assert callback_counter == info.numsteps + 1
    assert info.success
    assert_consistent(A, b, info, sol, 1.0e-7)
