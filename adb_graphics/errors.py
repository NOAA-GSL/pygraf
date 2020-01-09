''' Errors specific to the ADB Graphics package. '''


class Error(Exception):
    '''Base class for handling errors'''


class FieldNotUnique(Error):
    '''Exception raised when multiple Grib fields are found with input parameters'''
