class ArgumentError(Exception):
    """Raised when an argument is invalid.

    Analogue to ``ValueError`` which is not used here in order to be able
    to distinguish between built-in errors and ``krylov`` errors.
    """

    def __init__(self, message):
        super().__init__(message)


class AssumptionError(Exception):
    """Raised when an assumption is not satisfied.

    Differs from :py:class:`ArgumentError` in that all passed arguments are
    valid but computations reveal that assumptions are not satisfied and
    the result cannot be computed.
    """


class ConvergenceError(Exception):
    """Raised when a method did not converge.

    The ``ConvergenceError`` holds a message describing the error and
    the attribute ``solver`` through which the last approximation and other
    relevant information can be retrieved.
    """

    def __init__(self, msg):
        super().__init__(msg)


class LinearOperatorError(Exception):
    """Raised when a :py:class:`LinearOperator` cannot be applied."""


class InnerProductError(Exception):
    """Raised when the inner product is indefinite."""


class RuntimeError(Exception):
    """Raised for errors that do not fit in any other exception."""
