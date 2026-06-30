# ------------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2008, 2026. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ------------------------------------------------------------------------------
"""Imports the shared library on supported platforms."""

import platform
import sys
from pathlib import Path
from sys import version_info

from cplex_native_ce._platform_utils import get_library_path


# ---------------------------------------------------------------------------
# Library path setup
# ---------------------------------------------------------------------------

def _setup_library_path():
    """Add the platform-specific bin/<platform> directory to sys.path for imports."""
    # get_library_path() returns e.g. 'bin/x86-64_linux', relative to the package root
    lib_dir = get_library_path()

    # Absolute path: <package>/bin/<platform>/
    package_dir = Path(__file__).parent.resolve()
    lib_path = package_dir / lib_dir

    if lib_path.exists():
        lib_path_str = str(lib_path)
        if lib_path_str not in sys.path:
            sys.path.insert(0, lib_path_str)


# Only load native libraries when imported as part of the package (not during setup.py).
# When setup.py does a bare sys.path import, __package__ is None.

# Setup library path before importing
_setup_library_path()

ERROR_STRING = "CPLEX 22.2.0.0 is not compatible with this version of Python."
if platform.system() in ('Darwin', 'Linux', 'AIX', 'Windows', 'Microsoft'):
    if version_info < (3, 10, 0):
        raise Exception(ERROR_STRING)
    elif version_info < (3, 11, 0):
        import py310_cplex2220
        from py310_cplex2220 import *
    elif version_info < (3, 12, 0):
        import py311_cplex2220
        from py311_cplex2220 import *
    elif version_info < (3, 13, 0):
        import py312_cplex2220
        from py312_cplex2220 import *
    elif version_info < (3, 14, 0):
        import py313_cplex2220
        from py313_cplex2220 import *
    elif version_info < (3, 15, 0):
        import py314_cplex2220
        from py314_cplex2220 import *
    else:
        raise Exception(ERROR_STRING)
else:
    raise Exception("The CPLEX Python API is not supported on this platform.")
