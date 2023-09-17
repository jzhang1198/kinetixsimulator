""" 
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains classes for handling errors.
"""

class Error(Exception):
    """Base class for other exceptions"""
    pass

class NoRxnError(Error):
    """Raised when no reaction dictionaries are provided"""
    pass

class MalformedRxnError(Error):
    """Raised when a reaction key does not have the required keys"""

    def __init__(self, message):
        message

class SliderNameNotFoundWarning(Warning):
    def __init__(self, slider_name: str):
        self.msg = f'{slider_name} not a valid specie of rate constant. No slider will be displayed.'
    
    def __str__(self):
        return repr(self.msg)
