import numpy
import pytest
from numpy.testing import assert_almost_equal

import krylov

# def dictproduct(d):
#     """enhance itertools product to process values of dicts
#
#     example:
#         d = {'a':[1,2],'b':[3,4]}
#         then list(dictproduct(d)) ==
#         [{'a':1,'b':3}, {'a':1,'b':4}, {'a':2,'b':3}, {'a':2,'b':4}]
#     """
#     for p in itertools.product(*d.values()):
#         yield dict(zip(d.keys(), p))
#
#
# def dictpick(d):
#     yield {}
#     for k in d:
#         for v in d[k]:
#             yield {k: v}


def test_LinearSystem():
    A = numpy.diag(range(1, 11))
    exact_solution = numpy.ones((10, 1))
    b = A.dot(exact_solution)
    ls = krylov.linear_system.LinearSystem(
        A, b, M=numpy.eye(10), Ml=numpy.eye(10), Mr=numpy.eye(10)
    )
    # check that r=b for z=0
    Mr, r, rnorm = ls.get_residual(numpy.zeros((10, 1)), compute_norm=True)
    assert_almost_equal(r, b)
    assert_almost_equal(r, Mr)
    assert_almost_equal(rnorm, numpy.linalg.norm(b, 2))

    # check that r=0 for exact solution
    Mr, r, rnorm = ls.get_residual(exact_solution, compute_norm=True)
    assert_almost_equal(r, numpy.zeros((10, 1)))
    assert_almost_equal(r, Mr)
    assert_almost_equal(rnorm, 0)


# def linear_systems_generator(A, **ls_kwargs):
#     ip_Bs = [None, numpy.diag(range(1, 11))]
#     xs = [
#         numpy.ones((10, 1)),
#         numpy.ones((10,)),
#         (1 + 1j) * numpy.ones((10, 1)),
#         numpy.zeros((10, 1)),
#     ]
#     for ip_B, x in itertools.product(ip_Bs, xs):
#         if (
#             ip_B is not None
#             and "self_adjoint" in ls_kwargs
#             and ls_kwargs["self_adjoint"]
#         ):
#             # make A self-adjoint again if the inner product is changed
#             A_new = numpy.linalg.inv(ip_B).dot(A)
#         else:
#             A_new = A
#
#         preconditioners = {
#             "M": [None],
#             "Ml": [None, numpy.linalg.inv(A_new)],
#             "Mr": [None, numpy.linalg.inv(A_new)],
#         }
#
#         if "positive_definite" in ls_kwargs and ls_kwargs["positive_definite"]:
#             preconditioners["M"].append(numpy.linalg.inv(A_new))
#
#         # if A is diagonal, ip_B and all o
#         if (
#             numpy.linalg.norm(numpy.diag(numpy.diag(A_new)) - A_new) == 0
#             and ip_B is None
#         ):
#             M = numpy.diag(numpy.linspace(1, 10, 10))
#             preconditioners["M"].append(M)
#
#         for exact_solution in [None, x]:
#             for preconditioner in dictpick(preconditioners):
#                 kwargs = dict(ls_kwargs)
#                 kwargs.update(preconditioner)
#
#                 yield krylov.linear_system.LinearSystem(
#                     A_new,
#                     A_new.dot(x),
#                     ip_B=ip_B,
#                     exact_solution=exact_solution,
#                     **ls_kwargs
#                 )
#
#
# def solver_params_generator(solver, ls):
#     params_add = {}
#     if solver is krylov.RestartedGmres:
#         params_add = {"maxiter": [7], "max_restarts": [20]}
#     solver_params = {
#         "x0": [None, numpy.zeros(ls.b.shape), numpy.ones(ls.b.shape)],
#         "tol": [1e-13, 1e-2],
#         "maxiter": [15],
#     }
#     if ls.exact_solution is not None:
#         solver_params["x0"].append(ls.exact_solution)
#     for params in dictproduct(
#         dict(list(solver_params.items()) + list(params_add.items()))
#     ):
#         yield params
#
#
# cases = [
#     # spd
#     {
#         "A": test_utils.get_matrix_spd(),
#         "normal": True,
#         "self_adjoint": True,
#         "positive_definite": True,
#     },
#     # hpd
#     {
#         "A": test_utils.get_matrix_hpd(),
#         "normal": True,
#         "self_adjoint": True,
#         "positive_definite": True,
#     },
#     # symmetric indefinite
#     {"A": test_utils.get_matrix_symm_indef(), "normal": True, "self_adjoint": True},
#     # hermitian indefinite
#     {"A": test_utils.get_matrix_herm_indef(), "normal": True, "self_adjoint": True},
#     # nonsymm
#     {"A": test_utils.get_matrix_nonsymm()},
#     # nonsymm
#     {"A": test_utils.get_matrix_comp_nonsymm()},
# ]
#
#
# def generate_cases():
#     for case in cases:
#         for ls in linear_systems_generator(**case):
#             solvers = [krylov.Gmres, krylov.RestartedGmres]
#             if ls.self_adjoint:
#                 solvers.append(krylov.Minres)
#             if ls.positive_definite:
#                 solvers.append(krylov.Cg)
#             for solver in solvers:
#                 for params in solver_params_generator(solver, ls):
#                     yield solver, ls, params


# @pytest.mark.parametrize("args", generate_cases())
# def test_solver(args):
#     solver, ls, params = args
#     sol = solver(ls, **params)
#
#     # pick out the interesting data
#     b = krylov.utils.shape_vec(ls.b)
#     xk = krylov.utils.shape_vec(sol.xk)
#     N = len(b)
#     shape = (N, N)
#     A = krylov.utils.get_linearoperator(shape, ls.A)
#     M = krylov.utils.get_linearoperator(shape, ls.M)
#     Ml = krylov.utils.get_linearoperator(shape, ls.Ml)
#
#     # maxiter respected?
#     if "max_restarts" not in params:
#         assert len(sol.resnorms) - 1 <= params["maxiter"]
#     else:
#         assert len(sol.resnorms) - 1 <= params["maxiter"] * (params["max_restarts"] + 1)
#
#     # tolerance reached (if not near machine eps)?
#     if params["tol"] > 1e-15:
#         assert sol.resnorms[-1] <= params["tol"]
#
#     # final residual norm correct?
#     # relresvec[-1] == ||M*Ml*(b-A*xk))||_{M^{-1}} / ||M*Ml*b||_{M^{-1}}
#     # compute residual norm
#     MMlrk, Mlrk, MMlrk_norm = ls.get_residual(xk, compute_norm=True)
#
#     # finally: the assertion
#     if ls.MMlb_norm == 0:
#         assert abs(sol.resnorms[-1]) == 0
#     else:
#         assert_almost_equal(sol.resnorms[-1], MMlrk_norm / ls.MMlb_norm, decimal=14)
#
#     # final error norm correct?
#     # (if exact_solution was provided)
#     if ls.exact_solution is not None:
#         assert_almost_equal(
#             sol.errnorms[-1],
#             krylov.utils.norm(
#                 krylov.utils.shape_vec(ls.exact_solution)
#                 - krylov.utils.shape_vec(sol.xk),
#                 ip_B=ls.ip_B,
#             ),
#         )
#
#     # if the preconditioner is the inverse, then check if convergence
#     # occured after the first iteration
#     if (
#         isinstance(A, numpy.ndarray)
#         and isinstance(params["M"], numpy.ndarray)
#         and numpy.linalg.norm(numpy.eye(*A.shape) - numpy.dot(A, ls.M)) < 1e-15
#     ):
#         assert len(sol.resnorms) <= 2
#
#     # has gmres (without restarts) found the solution after max N iterations?
#     # (cg or minres may take longer because of roundoff errors)
#     if solver == krylov.Gmres and (
#         ("max_restarts" not in params) or (params["max_restarts"] == 0)
#     ):
#         assert len(sol.resnorms) - 1 <= ls.b.shape[0]


@pytest.mark.parametrize("solver", [krylov.cg, krylov.minres, krylov.gmres])
def test_spd(solver):
    a = numpy.linspace(1.0, 2.0, 5)
    a[-1] = 1e-2
    A = numpy.diag(a)
    b = numpy.ones(5)

    sol, info = solver(A, b, tol=1.0e-7)

    assert info.resnorms[-1] <= 1.0e-7


@pytest.mark.parametrize("solver", [krylov.cg, krylov.minres, krylov.gmres])
def test_hpd(solver):
    a = numpy.array(numpy.linspace(1.0, 2.0, 5), dtype=numpy.complex)
    a[0] = 5.0
    a[-1] = 1.0e-1
    A = numpy.diag(a)
    A[-1, 0] = 1.0e-1j
    A[0, -1] = -1.0e-1j

    b = numpy.ones(5, dtype=numpy.complex)

    sol, info = solver(A, b, tol=1.0e-7)

    assert info.resnorms[-1] <= 1.0e-7


@pytest.mark.parametrize("solver", [krylov.minres, krylov.gmres])
def test_symm_indef(solver):
    n = 5
    a = numpy.linspace(1.0, 2.0, n)
    a[-1] = -1
    A = numpy.diag(a)

    b = numpy.ones(n)

    sol, info = solver(A, b, tol=1.0e-12)

    assert info.resnorms[-1] <= 1.0e-12


@pytest.mark.parametrize("solver", [krylov.cg, krylov.minres, krylov.gmres])
def test_hermitian_indef(solver):
    n = 5
    a = numpy.array(numpy.linspace(1.0, 2.0, n), dtype=numpy.complex)
    a[-1] = 1e-3
    A = numpy.diag(a)
    A[-1, 0] = 10j
    A[0, -1] = -10j

    b = numpy.ones(n, dtype=numpy.complex)

    sol, info = solver(A, b, tol=1.0e-12)

    assert info.resnorms[-1] <= 1.0e-12


@pytest.mark.parametrize("solver", [krylov.gmres])
def test_real_unsymmetric(solver):
    n = 5
    a = numpy.array(range(1, n + 1), dtype=numpy.float)
    a[-1] = -1e1
    A = numpy.diag(a)
    A[0, -1] = 1e1

    b = numpy.ones(n)

    sol, info = solver(A, b, tol=1.0e-12)

    assert info.resnorms[-1] <= 1.0e-12


@pytest.mark.parametrize("solver", [krylov.gmres])
def test_complex_unsymmetric(solver):
    n = 5
    a = numpy.array(range(1, n + 1), dtype=numpy.complex)
    a[-1] = -1e1
    A = numpy.diag(a)
    A[0, -1] = 1.0e1j

    b = numpy.ones(n, dtype=numpy.complex)

    sol, info = solver(A, b, tol=1.0e-12)

    assert info.resnorms[-1] <= 1.0e-12


# @pytest.mark.parametrize("solver", [krylov.cg, krylov.minres, krylov.gmres])
# def test_final_error_norm(solver):
#     numpy.random.seed(0)
#     A = numpy.random.rand(5, 5)
#     b = numpy.random.rand(5)
#     exact = numpy.linalg.solve(A, b)
#
#     ls = krylov.linear_system.LinearSystem(
#         A=A,
#         b=b,
#         exact_solution=exact,
#     )
#     sol, info = solver(A, b, tol=1.0e-12)
#
#     # final error norm correct?
#     # (if exact_solution was provided)
#     if ls.exact_solution is not None:
#         assert_almost_equal(
#             info.errnorms[-1],
#             krylov.utils.norm(
#                 krylov.utils.shape_vec(ls.exact_solution)
#                 - krylov.utils.shape_vec(sol.xk),
#                 ip_B=ls.ip_B,
#             ),
#         )


@pytest.mark.parametrize("solver", [krylov.cg, krylov.minres, krylov.gmres])
def test_exact_solution_as_initial_guess(solver):
    A = numpy.diag([1.0e-3] + list(range(2, 11)))
    b = numpy.ones(10)
    x0 = numpy.linalg.solve(A, b)

    sol, info = solver(A, b, x0=x0)
    assert len(info.resnorms) == 1


@pytest.mark.parametrize("solver", [krylov.cg, krylov.minres, krylov.gmres])
def test_m(solver):
    a = numpy.linspace(1.0, 2.0, 5)
    A = numpy.diag(a)
    A[0, 0] = 1e-2
    b = numpy.ones(5)
    M = numpy.diag(a)
    sol, info = solver(A, b, M=M, tol=1.0e-12)
    assert info.resnorms[-1] <= 1.0e-12


@pytest.mark.parametrize("solver", [krylov.cg, krylov.minres, krylov.gmres])
def test_ml(solver):
    a = numpy.linspace(1.0, 2.0, 5)
    A = numpy.diag(a)
    A[0, 0] = 1e-2
    b = numpy.ones(5)
    M = numpy.diag(a)
    sol, info = solver(A, b, Ml=M, tol=1.0e-12)
    assert info.resnorms[-1] <= 1.0e-12


@pytest.mark.parametrize("solver", [krylov.cg, krylov.minres, krylov.gmres])
def test_mr(solver):
    a = numpy.linspace(1.0, 2.0, 5)
    A = numpy.diag(a)
    A[0, 0] = 1e-2
    b = numpy.ones(5)
    M = numpy.diag(a)
    sol, info = solver(A, b, Mr=M, tol=1.0e-12)
    assert info.resnorms[-1] <= 1.0e-12


@pytest.mark.parametrize(
    "method, ref",
    [
        (krylov.cg, [1004.1873775173957, 1000.0003174916551, 999.9999999997555]),
        # (krylov.gmres, [1004.1873724888546, 1000.0003124630923, 999.999994971191]),
        # (krylov.minres, [1004.187372488912, 1000.0003124632159, 999.9999949713145]),
    ],
)
@pytest.mark.parametrize("shape", [(100,), (100, 1)])
def test_solvers(method, ref, shape):
    tol = 1.0e-11
    n = shape[0]
    A = numpy.diag([1.0e-3] + list(range(2, n + 1)))

    # Make sure the shapes are alright
    b = numpy.ones(shape)
    sol, _ = method(A, b)
    assert sol.shape == b.shape

    assert abs(numpy.sum(numpy.abs(sol)) - ref[0]) < tol * ref[0]
    assert abs(numpy.sqrt(numpy.dot(sol.T, sol)) - ref[1]) < tol * ref[1]
    assert abs(numpy.max(numpy.abs(sol)) - ref[2]) < tol * ref[2]


@pytest.mark.parametrize(
    "method, ref",
    [
        # (krylov.cg, [1004.1873775173957, 1000.0003174916551, 999.9999999997555]),
        (krylov.gmres, [1004.1873724888546, 1000.0003124630923, 999.999994971191]),
        (krylov.minres, [1004.187372488912, 1000.0003124632159, 999.9999949713145]),
    ],
)
@pytest.mark.parametrize("shape", [(100,)])
def test_solvers_old(method, ref, shape):
    tol = 1.0e-11
    n = shape[0]
    A = numpy.diag([1.0e-3] + list(range(2, n + 1)))

    # Make sure the shapes are alright
    b = numpy.ones(shape)
    sol, _ = method(A, b)
    assert sol.shape == b.shape

    sol, _ = method(A, b)
    assert abs(numpy.sum(numpy.abs(sol)) - ref[0]) < tol * ref[0]
    assert abs(numpy.sqrt(numpy.dot(sol.T, sol)) - ref[1]) < tol * ref[1]
    assert abs(numpy.max(numpy.abs(sol)) - ref[2]) < tol * ref[2]


@pytest.mark.parametrize("solver", [
    krylov.cg,
    krylov.minres,
    # krylov.gmres
])
def test_custom_inner_product(solver):
    tol = 1.0e-9
    n = 100
    A = numpy.diag([1.0e-3] + list(range(2, n + 1)))
    b = numpy.ones(n)

    def inner(x, y):
        # TODO remove these workarounds
        # assert x.shape == b.shape
        # assert y.shape == b.shape
        reshape = False
        if len(x.shape) == 2:
            reshape = True
            x = x.reshape(-1)
        if len(y.shape) == 2:
            y = y.reshape(-1)
        w = 10 / numpy.arange(1, n + 1)
        out = numpy.dot(x.T, w * y)
        if reshape:
            out = out.reshape((1, 1))
        return out

    sol, _ = solver(A, b, inner_product=inner)

    ref = 1004.1873775173957
    assert abs(numpy.sum(numpy.abs(sol)) - ref) < tol * ref
    ref = 1000.0003174916551
    assert abs(numpy.sqrt(numpy.dot(sol, sol)) - ref) < tol * ref
    ref = 999.9999999997555
    assert abs(numpy.max(numpy.abs(sol)) - ref) < tol * ref


# @pytest.mark.parametrize("solver", [krylov.cg, krylov.minres, krylov.gmres])
# def test_custom_inner_product(solver):
#     tol = 1.0e-9
#     n = 100
#     A = numpy.diag([1.0e-3] + list(range(2, n + 1)))
#     b = numpy.ones((n, 1))
#
#     def inner(a, b):
#         w = 10 / numpy.arange(1, n + 1)
#         out = numpy.dot(a.T, b * w[:, None])
#         print("inner", a.shape, b.shape, out.shape)[0, 0]
#         exit(1)
#         return out
#
#     sol, _ = solver(A, b, inner_product=inner)
#
#     ref = 1004.1873775173957
#     assert abs(numpy.sum(numpy.abs(sol)) - ref) < tol * ref
#     ref = 1000.0003174916551
#     assert abs(numpy.sqrt(numpy.dot(sol, sol)) - ref) < tol * ref
#     ref = 999.9999999997555
#     assert abs(numpy.max(numpy.abs(sol)) - ref) < tol * ref
