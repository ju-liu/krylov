import numpy as np
import pytest
import scipy.sparse
import scipy.sparse.linalg

import krylov


@pytest.mark.parametrize("solver", [krylov.cg, krylov.minres, krylov.gmres])
@pytest.mark.parametrize("b_shape", [(5,), (5, 1), (5, 3)])
def test_spd(solver, b_shape):
    a = np.linspace(1.0, 2.0, b_shape[0])
    a[-1] = 1e-2
    A = np.diag(a)
    b = np.ones(b_shape)

    sol, info = solver(A, b, tol=1.0e-7)

    assert sol.shape == b.shape
    res = b - A @ sol
    assert np.all(np.sqrt(np.einsum("i...,i...->...", res, res)) < 1.0e-7)
    assert np.all(info.resnorms[-1] <= 1.0e-7)


@pytest.mark.parametrize("solver", [krylov.cg, krylov.minres, krylov.gmres])
def test_spd_rhs_funny_rhs(solver):
    a = np.linspace(1.0, 2.0, 5)
    a[-1] = 1e-2
    A = np.diag(a)
    np.random.seed(0)

    b = np.column_stack(
        [
            np.zeros(5),
            np.random.rand(5),
            np.random.rand(5),
        ]
    )
    sol = np.linalg.solve(A, b[:, 1])
    b = np.column_stack(
        [
            np.zeros(5),
            sol,
            np.zeros(5),
        ]
    )

    # solve individually
    ref = []
    for k in range(b.shape[1]):
        sol, info = solver(A, b[:, k], tol=1.0e-7)
        assert np.all(info.resnorms[-1] <= 1.0e-7)
        ref.append(sol)
    ref = np.column_stack(ref)

    # solve at once
    sol, info = solver(A, b, tol=1.0e-7)
    assert np.all(np.abs(sol - ref) < 1.0e-13 * np.abs(ref) + 1.0e-15)


@pytest.mark.parametrize("solver", [krylov.cg, krylov.minres, krylov.gmres])
def test_hpd(solver):
    a = np.array(np.linspace(1.0, 2.0, 5), dtype=complex)
    a[0] = 5.0
    a[-1] = 1.0e-1
    A = np.diag(a)
    A[-1, 0] = 1.0e-1j
    A[0, -1] = -1.0e-1j

    b = np.ones(5, dtype=complex)

    sol, info = solver(A, b, tol=1.0e-7)

    assert info.resnorms[-1] <= 1.0e-7


@pytest.mark.parametrize("solver", [krylov.minres, krylov.gmres])
def test_symm_indef(solver):
    n = 5
    a = np.linspace(1.0, 2.0, n)
    a[-1] = -1
    A = np.diag(a)
    b = np.ones(n)

    sol, info = solver(A, b, tol=1.0e-12)
    assert info.resnorms[-1] <= 1.0e-12


@pytest.mark.parametrize("solver", [krylov.cg, krylov.minres, krylov.gmres])
def test_hermitian_indef(solver):
    n = 5
    a = np.array(np.linspace(1.0, 2.0, n), dtype=complex)
    a[-1] = 1e-3
    A = np.diag(a)
    A[-1, 0] = 10j
    A[0, -1] = -10j
    b = np.ones(n, dtype=complex)

    sol, info = solver(A, b, tol=1.0e-12)
    assert info.resnorms[-1] <= 1.0e-11


@pytest.mark.parametrize("solver", [krylov.minres, krylov.gmres])
@pytest.mark.parametrize("b_shape", [(5,), (5, 1), (5, 3)])
@pytest.mark.parametrize(
    "ortho",
    ["mgs", "dmgs", "lanczos"],
)
def test_orthogonalizations(solver, b_shape, ortho):
    # build Hermitian, indefinite matrix
    n = b_shape[0]
    a = np.array(np.linspace(1.0, 2.0, n), dtype=complex)
    a[-1] = 1e-3
    A = np.diag(a)
    A[-1, 0] = 10j
    A[0, -1] = -10j
    b = np.ones(b_shape, dtype=complex)

    sol, info = solver(A, b, tol=1.0e-12, ortho=ortho)
    assert info.success
    assert np.all(info.resnorms[-1] <= 1.0e-11)


# separate out the householder test because it doesn't support non-vector right-hand
# sides yes.
@pytest.mark.parametrize("solver", [krylov.minres, krylov.gmres])
@pytest.mark.parametrize("b_shape", [(5,), (5, 1)])
@pytest.mark.parametrize(
    "ortho",
    ["householder"],
)
def test_orthogonalization_householder(solver, b_shape, ortho):
    # build Hermitian, indefinite matrix
    n = b_shape[0]
    a = np.array(np.linspace(1.0, 2.0, n), dtype=complex)
    a[-1] = 1e-3
    A = np.diag(a)
    A[-1, 0] = 10j
    A[0, -1] = -10j
    b = np.ones(b_shape, dtype=complex)

    sol, info = solver(A, b, tol=1.0e-12, ortho=ortho)
    assert info.success
    assert np.all(info.resnorms[-1] <= 1.0e-11)


@pytest.mark.parametrize("solver", [krylov.gmres])
def test_real_unsymmetric(solver):
    n = 5
    a = np.arange(1, n + 1, dtype=float)
    a[-1] = -1e1
    A = np.diag(a)
    A[0, -1] = 1e1
    b = np.ones(n)

    sol, info = solver(A, b, tol=1.0e-12)
    assert info.resnorms[-1] <= 1.0e-12


@pytest.mark.parametrize("solver", [krylov.gmres])
def test_complex_unsymmetric(solver):
    n = 5
    a = np.array(range(1, n + 1), dtype=complex)
    a[-1] = -1e1
    A = np.diag(a)
    A[0, -1] = 1.0e1j

    b = np.ones(n, dtype=complex)
    sol, info = solver(A, b, tol=1.0e-12)
    assert info.resnorms[-1] <= 1.0e-12


@pytest.mark.parametrize("solver", [krylov.cg, krylov.minres, krylov.gmres])
@pytest.mark.parametrize("b_shape", [(5,), (5, 1), (5, 3)])
def test_exact_sol(solver, b_shape):
    a = np.linspace(1.0, 2.0, b_shape[0])
    a[-1] = 1e-2
    A = np.diag(a)
    b = np.ones(b_shape)

    exact_solution = (b.T / a).T
    sol, info = solver(A, b, tol=1.0e-7, exact_solution=exact_solution)
    assert np.all(info.errnorms[-1] < 1.0e-7)


@pytest.mark.parametrize("solver", [krylov.cg, krylov.minres, krylov.gmres])
@pytest.mark.parametrize("b_shape", [(5,), (5, 1), (5, 3)])
def test_explicit_residual(solver, b_shape):
    a = np.linspace(1.0, 2.0, b_shape[0])
    a[-1] = 1e-2
    A = np.diag(a)
    b = np.ones(b_shape)

    sol, info = solver(A, b, tol=1.0e-7, use_explicit_residual=True)
    assert np.all(info.resnorms[-1] < 1.0e-7)


@pytest.mark.parametrize("solver", [krylov.cg, krylov.minres, krylov.gmres])
@pytest.mark.parametrize("b_shape", [(5,), (5, 1), (5, 3)])
def test_return_arnoldi(solver, b_shape):
    a = np.linspace(1.0, 2.0, b_shape[0])
    a[-1] = 1e-2
    A = np.diag(a)
    b = np.ones(b_shape)

    sol, info = solver(A, b, tol=1.0e-7, return_arnoldi=True)
    assert np.all(info.resnorms[-1] < 1.0e-7)


# @pytest.mark.parametrize("solver", [krylov.cg, krylov.minres, krylov.gmres])
# def test_final_error_norm(solver):
#     np.random.seed(0)
#     A = np.random.rand(5, 5)
#     b = np.random.rand(5)
#     exact = np.linalg.solve(A, b)
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
#                 inner=ls.inner,
#             ),
#         )


@pytest.mark.parametrize("solver", [krylov.cg, krylov.minres, krylov.gmres])
def test_exact_solution_as_initial_guess(solver):
    A = np.diag([1.0e-3] + list(range(2, 11)))
    b = np.ones(10)
    x0 = np.linalg.solve(A, b)

    sol, info = solver(A, b, x0=x0)
    assert len(info.resnorms) == 1


@pytest.mark.parametrize("solver", [krylov.cg, krylov.minres, krylov.gmres])
def test_m(solver):
    a = np.linspace(1.0, 2.0, 5)
    A = np.diag(a)
    A[0, 0] = 1e-2
    b = np.ones(5)
    M = np.diag(a)
    sol, info = solver(A, b, M=M, tol=1.0e-12)
    assert info.resnorms[-1] <= 1.0e-12


@pytest.mark.parametrize("solver", [krylov.cg, krylov.minres, krylov.gmres])
def test_ml(solver):
    a = np.linspace(1.0, 2.0, 5)
    A = np.diag(a)
    A[0, 0] = 1e-2
    b = np.ones(5)
    M = np.diag(a)
    sol, info = solver(A, b, Ml=M, tol=1.0e-12)
    assert info.resnorms[-1] <= 1.0e-12


@pytest.mark.parametrize("solver", [krylov.minres, krylov.gmres])
def test_mr(solver):
    a = np.linspace(1.0, 2.0, 5)
    A = np.diag(a)
    A[0, 0] = 1e-2
    b = np.ones(5)
    M = np.diag(a)
    sol, info = solver(A, b, Mr=M, tol=1.0e-12)
    assert info.resnorms[-1] <= 1.0e-12


@pytest.mark.parametrize(
    "method, ref",
    [
        (krylov.cg, [1004.1873775173957, 1000.0003174916551, 999.9999999997555]),
        (krylov.gmres, [1004.1873724888546, 1000.0003124630923, 999.999994971191]),
        (krylov.minres, [1004.187372488912, 1000.0003124632159, 999.9999949713145]),
    ],
)
@pytest.mark.parametrize("shape", [(100,), (100, 1)])
def test_solvers(method, ref, shape):
    tol = 1.0e-11
    n = shape[0]
    A = np.diag([1.0e-3] + list(range(2, n + 1)))

    # Make sure the shapes are alright
    b = np.ones(shape)
    sol, _ = method(A, b)
    assert sol.shape == b.shape

    assert abs(np.sum(np.abs(sol)) - ref[0]) < tol * ref[0]
    assert abs(np.sqrt(np.dot(sol.T, sol)) - ref[1]) < tol * ref[1]
    assert abs(np.max(np.abs(sol)) - ref[2]) < tol * ref[2]


@pytest.mark.parametrize(
    "solver",
    [krylov.cg, krylov.minres, krylov.gmres],
)
def test_custom_inner_product(solver):
    tol = 1.0e-9
    n = 100
    A = np.diag([1.0e-3] + list(range(2, n + 1)))
    b = np.ones(n)

    def inner(x, y):
        assert x.shape == b.shape
        assert y.shape == b.shape
        w = 10 / np.arange(1, n + 1)
        return np.dot(x.T, w * y)

    sol, _ = solver(A, b, inner=inner)

    ref = 1004.1873775173957
    assert abs(np.sum(np.abs(sol)) - ref) < tol * ref
    ref = 1000.0003174916551
    assert abs(np.sqrt(np.dot(sol, sol)) - ref) < tol * ref
    ref = 999.9999999997555
    assert abs(np.max(np.abs(sol)) - ref) < tol * ref


@pytest.mark.parametrize(
    "solver",
    [krylov.cg, krylov.minres, krylov.gmres],
)
def test_custom_inner_product_nx1(solver):
    tol = 1.0e-9
    n = 100
    A = np.diag([1.0e-3] + list(range(2, n + 1)))
    b = np.ones((n, 1))

    def inner(x, y):
        assert x.shape == b.shape
        assert y.shape == b.shape
        w = 10 / np.arange(1, n + 1)
        return np.dot(x.T, w[:, None] * y)[0, 0]

    sol, _ = solver(A, b, inner=inner)

    ref = 1004.1873775173957
    assert abs(np.sum(np.abs(sol)) - ref) < tol * ref
    ref = 1000.0003174916551
    assert abs(np.sqrt(np.dot(sol.T, sol)) - ref) < tol * ref
    ref = 999.9999999997555
    assert abs(np.max(np.abs(sol)) - ref) < tol * ref


@pytest.mark.parametrize("solver", [krylov.cg, krylov.minres, krylov.gmres])
def test_scipy_sparse(solver):
    n = 5
    a = np.linspace(1.0, 2.0, n)
    a[-1] = 1e-2

    A = scipy.sparse.spdiags(a, [0], n, n)
    b = np.ones(n)

    sol, info = solver(A, b, tol=1.0e-12)
    assert info.resnorms[-1] <= 1.0e-12


@pytest.mark.parametrize("solver", [krylov.cg, krylov.minres, krylov.gmres])
def test_scipy_linear_operator(solver):
    n = 5
    a = np.linspace(1.0, 2.0, n)
    a[-1] = 1e-2

    A = scipy.sparse.linalg.LinearOperator((n, n), lambda x: a * x)
    b = np.ones(n)

    sol, info = solver(A, b, tol=1.0e-12)
    assert info.resnorms[-1] <= 1.0e-12


@pytest.mark.parametrize("solver", [krylov.cg, krylov.minres, krylov.gmres])
def test_custom_linear_operator(solver):
    n = 5

    class MyLinearOperator:
        def __init__(self):
            self.a = np.linspace(1.0, 2.0, n)
            self.a[-1] = 1e-2
            self.shape = (n, n)
            self.dtype = float

        def __matmul__(self, x):
            return self.a * x

    A = MyLinearOperator()
    b = np.ones(n)

    sol, info = solver(A, b, tol=1.0e-12)
    assert info.resnorms[-1] <= 1.0e-12


if __name__ == "__main__":
    test_orthogonalizations(krylov.minres, (5,), "householder")
    print()
    test_orthogonalizations(krylov.minres, (5, 1), "householder")
