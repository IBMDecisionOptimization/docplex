"""CPLEX Community Edition Native Libraries

This package provides the native CPLEX solver libraries (_pycplex extension)
for the Community Edition with limited capabilities.
"""

# Import the SWIG-generated wrapper module
# This imports _pycplex_platform which loads the correct native library
from . import _pycplex

# Expose it at package level so entry point can find it
__all__ = ['_pycplex']

# Version info
from ._version import __version__

# Made with Bob
