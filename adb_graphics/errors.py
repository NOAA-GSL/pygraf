''' Errors specific to the ADB Graphics package. '''


class Error(Exception):
    '''Base class for handling errors'''


class FieldNotUnique(Error):
    '''Exception raised when multiple Grib fields are found with input parameters'''

class GribReadError(Error):
    '''Exception raised when there is an error reading the grib file.'''

class NoGraphicsDefinitionForVariable(Error):
    '''Exception raised when there is no configuration for the variable.'''
