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