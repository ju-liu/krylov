import numpy as np
import pytest

import krylov

from .helpers import assert_consistent
from .linear_problems import (
    hermitian_indefinite,
    hpd,
    spd_dense,
    spd_rhs_0,
    spd_rhs_0sol0,
    spd_sparse,
    symmetric_indefinite,
)


def test_ref():
    A, b = symmetric_indefinite()
    _, info = krylov.symmlq(A, b)

    ref = np.array(
        [
            2.23606797749979,
            0.9823441352194251,
            0.5792270481089666,
            0.2307060320183781,
            0.16833036914998076,
            0.053350208455011996,
            0.0033119344706411423,
            0.27582030681696557,
            0.2242220189485338,
            0.07981010715785067,
            4.440892098500626e-16,
        ]
    )
    print(info.resnorms)
    assert np.all(np.abs(info.resnorms - ref) < 1.0e-13 * (1.0 + np.abs(ref)))


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

    A_dense = A if isinstance(A, np.ndarray) else A.toarray()
    sol = np.linalg.solve(A_dense, b)
    sol, info = krylov.symmlq(A, b, tol=1.0e-7, maxiter=10, exact_solution=sol)
    print("info:")
    print(info)
    assert info.success
    assert_consistent(A, b, info, sol, 1.0e-7)
