class ArgumentError(Exception):
    """Raised when an argument is invalid.

    Analogue to ``ValueError`` which is not used here in order to be able
    to distinguish between built-in errors and ``krylov`` errors.
    """


class AssumptionError(Exception):
    """Raised when an assumption is not satisfied.

    Differs from :py:class:`ArgumentError` in that all passed arguments are
    valid but computations reveal that assumptions are not satisfied and
    the result cannot be computed.
    """

    pass


class ConvergenceError(Exception):
    """Raised when a method did not converge.

    The ``ConvergenceError`` holds a message describing the error and
    the attribute ``solver`` through which the last approximation and other
    relevant information can be retrieved.
    """

    def __init__(self, msg, solver):
        super(ConvergenceError, self).__init__(msg)
        self.solver = solver


class LinearOperatorError(Exception):
    """Raised when a :py:class:`LinearOperator` cannot be applied."""


class InnerProductError(Exception):
    """Raised when the inner product is indefinite."""


class RuntimeError(Exception):
    """Raised for errors that do not fit in any other exception."""
