import warnings

import numpy
import scipy

from .errors import ArgumentError


def ritz(H, V=None, hermitian=False, type="ritz"):
    """Compute several kinds of Ritz pairs from an Arnoldi/Lanczos relation.

    This function computes Ritz, harmonic Ritz or improved harmonic Ritz values and
    vectors with respect to the Krylov subspace :math:`K_n(A,v)` from the extended
    Hessenberg matrix :math:`\\underline{H}_n` generated with n iterations the Arnoldi
    algorithm applied to A and v.

    :param H: Hessenberg matrix from Arnoldi/Lanczos algorithm.
    :param V: (optional) Arnoldi/Lanczos vectors,
      :math:`V\\in\\mathbb{C}^{N,n+1}`. If provided, the Ritz vectors are also returned.
      The Arnoldi vectors have to form an orthonormal basis with respect to an inner
      product.

      **Caution:** if you are using the Lanzcos or Gram-Schmidt Arnoldi algorithm
      without reorthogonalization, then the orthonormality of the basis is usually lost.
      For accurate results it is advisable to use the Householder Arnoldi
      (``ortho='house'``) or modified Gram-Schmidt with reorthogonalization
      (``ortho='dmgs'``).
    :param hermitian: (optional) if set to ``True`` the matrix :math:`H_n` must be
    Hermitian. A Hermitian matrix :math:`H_n` allows for faster and often more accurate
    computation of Ritz pairs.
    :param type: (optional) type of Ritz pairs, may be one of ``'ritz'``, ``'harmonic'``
    or ``'harmonic_like'``. Two choices of Ritz pairs fit in the following description:

      `Given two n-dimensional subspaces`
      :math:`X,Y\\subseteq \\mathbb{C}^N`,
      `find a basis`
      :math:`z_1,\\ldots,z_n`
      `of`
      :math:`X`
      `and` :math:`\\theta_1,\\ldots,\\theta_n\\in\\mathbb{C}`
      such that
      :math:`A z_i - \\theta_i z_i \\perp Y`
      for all :math:`i\\in\\{1,\\ldots,n\\}`.

      In this setting the choices are

      * ``'ritz'``: regular Ritz pairs, i.e. :math:`X=Y=K_n(A,v)`.
      * ``'harmonic'``: harmonic Ritz pairs, i.e.
          :math:`X=K_n(A,v)` and :math:`Y=AK_n(A,v)`.
      * ``'harmonic_improved'``: the returned vectors ``U`` (and ``V``, if requested)
      are the same as with ``type='harmonic'``. The ``theta`` array contains the
      improved Ritz values :math:`\\theta_i = u_i^* H_n u_i`, cf. section 2 in *Morgan,
      Zeng.  Harmonic Projection Methods for Large Non-symmetric Eigenvalue Problems.
      1998.* It can be shown that the residual norm of improved Ritz pairs is always
      less than or equal to the residual norm of the harmonic Ritz pairs. However, the
      improved Ritz pairs do not fit into the framework above since the orthogonality
      condition is lost.

    :return:

      * If V is not ``None`` then ``theta, U, resnorm, Z`` is returned.
      * If V is ``None`` then ``theta, U, resnorm`` is returned.

      Where

      * ``theta`` are the Ritz values :math:`[\\theta_1,\\ldots,\\theta_n]`.
      * ``U`` are the coefficients of the Ritz vectors in the Arnoldi basis, i.e.
        :math:`z_i=Vu_i` where :math:`u_i` is the i-th column of U.
      * ``resnorm`` is a residual norm vector.
      * ``Z`` are the actual Ritz vectors, i.e. ``Z=dot(V,U)``.
    """
    # TODO: enhance ritz to accept an augmented space

    n = H.shape[1]
    if V is not None and V.shape[0] != H.shape[0]:
        raise ArgumentError("shape mismatch with V and H")
    if not H.shape[0] in [n, n + 1]:
        raise ArgumentError("H not of shape (n+1,n) or (n,n)")
    symmres = numpy.linalg.norm(H[:n, :] - H[:n, :].T.conj())
    if hermitian and symmres >= 5e-14:
        warnings.warn(f"Hessenberg matrix is not symmetric: |H-H^*|={symmres}")

    # choose eig for Hermitian or non-Hermitian matrices
    eig = scipy.linalg.eigh if hermitian else scipy.linalg.eig

    if type == "ritz":
        theta, U = eig(H[:n, :])
        beta = 0 if H.shape[0] == n else H[-1, -1]
        resnorm = numpy.abs(beta * U[-1, :])
    elif type == "harmonic":
        theta, U = eig(H[:n, :].T.conj(), numpy.dot(H.T.conj(), H))
        theta = 1 / theta
        resnorm = []
        for i in range(n):
            U[:, i] /= numpy.linalg.norm(U[:, i], 2)
            resi = numpy.dot(H, U[:, i])
            if resi.dtype != numpy.complex and theta.dtype == numpy.complex:
                resi = numpy.array(resi, dtype=numpy.complex)
            resi[:n] -= theta[i] * U[:, i]
            resnorm.append(numpy.linalg.norm(resi, 2))
        resnorm = numpy.array(resnorm)
    elif type == "harmonic_improved":
        theta, U = eig(H[:n, :].T.conj(), numpy.dot(H.T.conj(), H))
        rho = []
        for i in range(n):
            U[:, i] /= numpy.linalg.norm(U[:, i], 2)
            rho.append(numpy.dot(U[:, i].T.conj(), numpy.dot(H[:n, :], U[:, i])))
        theta = numpy.array(rho)
        resnorm = []
        for i in range(n):
            resi = numpy.dot(H, U[:, i])
            resi[:n] -= theta[i] * U[:, i]
            resnorm.append(numpy.linalg.norm(resi, 2))
        resnorm = numpy.array(resnorm)
        pass
    else:
        raise ArgumentError(f"unknown Ritz type {type}")

    if V is not None:
        V = V.reshape(V.shape[:2]).T  # TODO remove
        return theta, U, resnorm, numpy.dot(V[:, :n], U)

    return theta, U, resnorm
