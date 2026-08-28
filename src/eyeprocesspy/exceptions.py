class EyeProcessError(Exception):
    """Base error for eyeprocesspy."""

class EyeProcessValidationError(EyeProcessError):
    """Raised when a canonical dataset violates validation contracts."""

class EyeProcessSchemaError(EyeProcessError):
    """Raised for canonical schema errors."""

class EyeProcessTimebaseError(EyeProcessError):
    """Raised for timebase normalization or alignment errors."""

class EyeProcessCoordinateError(EyeProcessError):
    """Raised for coordinate-space errors."""

class EyeProcessBackendError(EyeProcessError):
    """Raised when an optional backend is required or fails."""

class EyeProcessModelError(EyeProcessError):
    """Raised for modelling failures."""

class EyeProcessGovernanceError(EyeProcessError):
    """Raised when an evidence/governance gate is violated."""
