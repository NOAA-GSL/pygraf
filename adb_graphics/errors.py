"""Errors specific to the ADB Graphics package."""


class FieldNotUniqueError(Exception):
    """Exception raised when multiple Grib fields are found with input parameters."""


class GribReadError(Exception):
    """Exception raised when there is an error reading the grib file."""

    def __init__(self, name: str, message: str = "was not found"):
        self.name = name
        self.message = message

        super().__init__(message)

    def __str__(self):
        return f'"{self.name}" {self.message}'


class NoGraphicsDefinitionForVariableError(Exception):
    """Exception raised when there is no configuration for the variable."""
