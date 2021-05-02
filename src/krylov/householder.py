import numpy as np


class Householder:
    def __init__(self, x):
        """Compute Householder transformation for given vector.

        Initialize Householder transformation :math:`H` such that :math:`Hx = \\alpha
        \\|x\\|_2 e_1` with :math:`|\\alpha|=1`

        The algorithm is a combination of Algorithm 5.1.1 on page 236 and the treatment
        of the complex case in Section 5.1.13 on page 243 in Golub, Van Loan. Matrix
        computations. Fourth Edition. 2013.
        """
        # TODO vectorize
        assert len(x.shape) == 1 or (len(x.shape) == 2 and x.shape[1] == 1), (
            "Householder only works for quasi-1D vectors for now. "
            f"Input vector has shape {x.shape}."
        )

        # Householder only works with the Euclidean inner product
        if len(x.shape) == 1:
            # np.dot is faster than einsum for flat vectors
            def inner(a, b):
                return np.dot(a.conj(), b)

        else:

            def inner(a, b):
                return np.einsum("i...,i...->...", a.conj(), b)

        self.inner = inner

        v = x.copy()

        gamma = v[0].copy()
        v[0] = 1

        sigma2 = inner(v[1:], v[1:])
        xnorm = np.sqrt(np.abs(gamma) ** 2 + sigma2)

        # is x a multiple of first unit vector?
        if sigma2 == 0:
            beta = 0
            xnorm = np.abs(gamma)
            alpha = 1 if gamma == 0 else gamma / xnorm
        else:
            beta = 2
            if gamma == 0:
                v[0] = -np.sqrt(sigma2)
                alpha = 1
            else:
                v[0] = gamma + gamma / np.abs(gamma) * xnorm
                alpha = -gamma / np.abs(gamma)

        self.xnorm = xnorm
        self.v = v / np.sqrt(np.abs(v[0]) ** 2 + sigma2)
        self.alpha = alpha
        self.beta = beta

    def apply(self, x):
        """Apply Householder transformation to vector x.

        Applies the Householder transformation efficiently to the given vector.
        """
        if self.beta == 0:
            return x
        assert (
            x.shape == self.v.shape
        ), "Shape mismatch! (v.shape = {} != {} = x.shape)".format(
            self.v.shape, x.shape
        )
        return x - self.beta * self.v * self.inner(self.v, x)

    def matrix(self):
        """Build matrix representation of Householder transformation.

        Builds the matrix representation :math:`H = I - \\beta vv^*`.

        **Use with care!** This routine may be helpful for testing purposes but should
        not be used in production codes for high dimensions since the resulting matrix
        is dense.
        """
        n = self.v.shape[0]

        # create identity matrix
        eye = np.zeros([n, n] + list(self.v.shape[1:]))
        idx = np.arange(n)
        eye[idx, idx] = 1.0

        vvH = np.einsum("i...,j...->ij...", self.v, self.v.conj())
        return eye - self.beta * vvH
