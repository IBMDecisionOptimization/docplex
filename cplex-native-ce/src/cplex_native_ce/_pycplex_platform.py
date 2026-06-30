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

def _setup_library_path() -> None:
    """Prepend the package's platform-specific native library directory to sys.path."""
    # get_library_path() returns e.g. 'bin/x86-64_linux', relative to the package root
    relative_lib_dir = get_library_path()
    library_path = Path(__file__).resolve().parent / relative_lib_dir

    if not library_path.is_dir():
        return

    library_path_str = str(library_path)
    if library_path_str in sys.path:
        return

    sys.path.insert(0, library_path_str)


# Ensure the native library directory is on sys.path before importing versioned bindings.
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
