"""Errors the user can act on. The UI shows their message as is, without a traceback."""


class ChurnAppError(Exception):
    """Base class for problems caused by the input rather than by a bug."""


class DataFileError(ChurnAppError):
    """The uploaded file could not be read."""


class DataValidationError(ChurnAppError):
    """The file was read, but it cannot be used to train or score."""
