import numpy
import pytest

import krylov
import test_linsys


@pytest.mark.parametrize(
    "Solver",
    [
        krylov.recycling.RecyclingCg,
        krylov.recycling.RecyclingMinres,
        krylov.recycling.RecyclingGmres,
    ],
)
@pytest.mark.parametrize("which", ["lm", "sm", "lr", "sr", "li", "si", "smallest_res"])
def test_RitzFactorySimple(Solver, which):
    N = 100
    d = numpy.linspace(1, 2, N)
    d[:5] = [1e-8, 1e-4, 1e-2, 2e-2, 3e-2]
    ls = krylov.linsys.LinearSystem(
        numpy.diag(d),
        numpy.ones((N, 1)),
        normal=True,
        self_adjoint=True,
        positive_definite=True,
    )

    vector_factory = krylov.recycling.factories.RitzFactorySimple(
        n_vectors=3, which=which
    )
    recycling_solver = Solver()
    sols = []
    params = {"maxiter": 50, "tol": 1e-5, "x0": None}
    for i in range(3):
        sols.append(recycling_solver.solve(ls, vector_factory=vector_factory, **params))
        test_linsys.check_solver(sols[-1], Solver, ls, params)
        if i > 0:
            assert len(sols[-1].resnorms) <= len(sols[0].resnorms)


# def _get_ritz(A, self_adjoint):
#     linear_system = krylov.linsys.LinearSystem(A, numpy.ones((10, 1)), self_adjoint=True)
#     deflated_solver = krylov.deflation.DeflatedGmres(linear_system, store_arnoldi=True)
#     return krylov.deflation.Ritz(deflated_solver)
#
#
# @pytest.mark.parametrize(
#     "matrix",
#     [
#         test_utils.get_matrix_spd(),
#         test_utils.get_matrix_hpd(),
#         test_utils.get_symm_indef(),
#         test_utils.get_herm_indef(),
#     ],
# )
# @pytest.mark.parametrize("self_adjoint", [True])
# @pytest.mark.parametrize("max_vectors", [numpy.Inf, 2])
# def test_SmallRitzGenerator(A, self_adjoint, max_vectors):
#     ritz = _get_ritz(A, self_adjoint)
#     small = krylov.recycling.generators.SmallRitz(max_vectors=max_vectors)
#     pass
