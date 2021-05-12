import numpy as np
import pytest

import krylov

from .helpers import assert_consistent
from .linear_problems import hermitian_indefinite, hpd
from .linear_problems import spd_dense as spd
from .linear_problems import spd_rhs_0, spd_rhs_0sol0, symmetric_indefinite


@pytest.mark.parametrize(
    "A_b",
    [
        spd((5,)),
        spd((5, 1)),
        spd((5, 3)),
        spd_rhs_0((5,)),
        spd_rhs_0sol0(),
        hpd(),
        # no idea why this works:
        symmetric_indefinite(),
        hermitian_indefinite(),
    ],
)
def test_cg(A_b):
    A, b = A_b

    A_dense = A if isinstance(A, np.ndarray) else A.toarray()
    sol = np.linalg.solve(A_dense, b)

    sol, info = krylov.cg(A, b, tol=1.0e-7, exact_solution=sol)
    assert info.success
    assert_consistent(A, b, info, sol, 1.0e-7)
