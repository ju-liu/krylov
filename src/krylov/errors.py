class ArgumentError(Exception):
    """Raised when an argument is invalid.

    Analogue to ``ValueError`` which is not used here in order to be able
    to distinguish between built-in errors and ``krylov`` errors.
    """

    def __init__(self, message):
        super().__init__(message)
