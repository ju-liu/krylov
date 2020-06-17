import numpy

import krylov


def get_matrix_spd():
    a = numpy.linspace(1, 2, 10)
    a[-1] = 1e-2
    return numpy.diag(a)


def get_matrix_hpd():
    a = numpy.array(numpy.linspace(1, 2, 10), dtype=numpy.complex)
    a[0] = 5
    a[-1] = 1e-1
    A = numpy.diag(a)
    A[-1, 0] = 1e-1j
    A[0, -1] = -1e-1j
    return A


def get_matrix_symm_indef():
    a = numpy.linspace(1, 2, 10)
    a[-1] = -1
    return numpy.diag(a)


def get_matrix_herm_indef():
    a = numpy.array(numpy.linspace(1, 2, 10), dtype=numpy.complex)
    a[-1] = 1e-3
    A = numpy.diag(a)
    A[-1, 0] = 10j
    A[0, -1] = -10j
    return A


def get_matrix_nonsymm():
    a = numpy.array(range(1, 11), dtype=numpy.float)
    a[-1] = -1e1
    A = numpy.diag(a)
    A[0, -1] = 1e1
    return A


def get_matrix_comp_nonsymm():
    a = numpy.array(range(1, 11), dtype=numpy.complex)
    a[-1] = -1e1
    A = numpy.diag(a)
    A[0, -1] = 1.0e1j
    return A


def get_matrices(
    spd=True,
    hpd=True,
    symm_indef=True,
    herm_indef=True,
    nonsymm=True,
    comp_nonsymm=True,
):
    matrices = []
    if spd:
        matrices.append(get_matrix_spd())
    if hpd:
        matrices.append(get_matrix_hpd())
    if symm_indef:
        matrices.append(get_matrix_symm_indef())
    if herm_indef:
        matrices.append(get_matrix_herm_indef())
    if nonsymm:
        matrices.append(get_matrix_nonsymm())
    if comp_nonsymm:
        matrices.append(get_matrix_comp_nonsymm())
    return matrices


_matrices_herm = [
    get_matrix_spd(),
    # get_matrix_hpd(),
    # get_matrix_symm_indef(),
    # get_matrix_herm_indef(),
]
_matrices_nonherm = [get_matrix_nonsymm(), get_matrix_comp_nonsymm()]


def get_ip_Bs():
    B = numpy.diag(numpy.linspace(1, 5, 10))
    return [
        None,
        krylov.utils.MatrixLinearOperator(B),
        lambda x, y: numpy.dot(x.T.conj(), numpy.dot(B, y)),
    ]
