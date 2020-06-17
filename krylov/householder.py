import numpy

from .errors import ArgumentError


class Householder:
    def __init__(self, x):
        """Compute Householder transformation for given vector.

        Initialize Householder transformation :math:`H` such that :math:`Hx = \\alpha
        \\|x\\|_2 e_1` with :math:`|\\alpha|=1`

        The algorithm is a combination of Algorithm 5.1.1 on page 236 and the treatment
        of the complex case in Section 5.1.13 on page 243 in Golub, Van Loan. Matrix
        computations. Fourth Edition. 2013.
        """
        # make sure that x is a vector ;)
        if len(x.shape) != 2 or x.shape[1] != 1:
            raise ArgumentError("x is not a vector of dim (N,1)")

        v = x.copy()

        gamma = v[0].item()
        v[0] = 1
        if x.shape[0] == 1:
            sigma = 0
            xnorm = numpy.abs(gamma)
            beta = 0
            alpha = 1 if gamma == 0 else gamma / xnorm
        else:
            sigma = numpy.linalg.norm(v[1:], 2)
            xnorm = numpy.sqrt(numpy.abs(gamma) ** 2 + sigma ** 2)

            # is x the multiple of first unit vector?
            if sigma == 0:
                beta = 0
                xnorm = numpy.abs(gamma)
                alpha = 1 if gamma == 0 else gamma / xnorm
            else:
                beta = 2
                if gamma == 0:
                    v[0] = -sigma
                    alpha = 1
                else:
                    v[0] = gamma + gamma / numpy.abs(gamma) * xnorm
                    alpha = -gamma / numpy.abs(gamma)

        self.xnorm = xnorm
        self.v = v / numpy.sqrt(numpy.abs(v[0]) ** 2 + sigma ** 2)
        self.alpha = alpha
        self.beta = beta

    def apply(self, x):
        """Apply Householder transformation to vector x.

        Applies the Householder transformation efficiently to the given vector.
        """
        # make sure that x is a (N,*) matrix
        if len(x.shape) != 2:
            raise ArgumentError("x is not a matrix of shape (N,*)")
        if self.beta == 0:
            return x
        return x - self.beta * self.v * numpy.dot(self.v.T.conj(), x)

    def matrix(self):
        """Build matrix representation of Householder transformation.

        Builds the matrix representation :math:`H = I - \\beta vv^*`.

        **Use with care!** This routine may be helpful for testing purposes but should
        not be used in production codes for high dimensions since the resulting matrix
        is dense.
        """
        n = self.v.shape[0]
        return numpy.eye(n, n) - self.beta * numpy.dot(self.v, self.v.T.conj())
