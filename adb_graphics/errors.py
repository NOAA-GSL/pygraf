''' Errors specific to the ADB Graphics package. '''


class Error(Exception):
    '''Base class for handling errors'''


class FieldNotUnique(Error):
    '''Exception raised when multiple Grib fields are found with input parameters'''

class GribReadError(Error):
    '''Exception raised when there is an error reading the grib file.'''

    def __init__(self, name, message="was not found"):
        self.name = name
        self.message = message

        super().__init__(message)

    def __str__(self):
        return f'"{self.name}" {self.message}'

class NoGraphicsDefinitionForVariable(Error):
    '''Exception raised when there is no configuration for the variable.'''

class LevelNotFound(Error):
    '''Exception raised when there is no configuration for the variable.'''

class OutsideDomain(Error):
    '''Exception raised when there is no configuration for the variable.'''
