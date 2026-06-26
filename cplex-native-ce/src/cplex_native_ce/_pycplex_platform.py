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

import os
import sys
import platform
from pathlib import Path
from sys import version_info


def _setup_library_path():
    """Add the platform-specific library directory to sys.path for imports."""
    # Determine platform library directory
    system = platform.system()
    machine = platform.machine()
    
    if system == 'Linux':
        if machine in ('x86_64', 'AMD64'):
            lib_dir = 'x86-64_linux'
        elif machine == 'aarch64':
            lib_dir = 'aarch64_linux'
        elif machine == 'ppc64le':
            lib_dir = 'ppc64le_linux'
        else:
            lib_dir = None
    elif system == 'Darwin':
        if machine == 'arm64':
            lib_dir = 'arm64_osx'
        else:
            lib_dir = 'x86-64_osx'
    elif system in ('Windows', 'Microsoft'):
        if machine in ('AMD64', 'x86_64'):
            lib_dir = 'x86-64_windows'
        else:
            lib_dir = 'x86_windows'
    elif system == 'AIX':
        lib_dir = 'ppc64_aix'
    else:
        lib_dir = None
    
    if not lib_dir:
        return  # Will fail later with appropriate error
    
    # Get package directory (absolute path)
    package_dir = Path(__file__).parent.resolve()
    
    # Check if libraries are in package directory (installed wheel)
    if any(package_dir.glob('*.so')) or any(package_dir.glob('*.pyd')) or any(package_dir.glob('*.dylib')):
        # Libraries already in package directory, no need to add path
        return
    
    # Libraries are in libs directory (development/editable install)
    # package_dir is: .../cplex-native-ce/src/cplex_native_ce
    # We need to go up to: .../cplex-native-ce/libs/<platform>
    project_root = package_dir.parent.parent  # Go up from src/cplex_native_ce to cplex-native-ce
    lib_path = project_root / 'libs' / lib_dir
    
    if lib_path.exists():
        # Add to sys.path so Python can import the .so/.pyd files
        lib_path_str = str(lib_path)
        if lib_path_str not in sys.path:
            sys.path.insert(0, lib_path_str)


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
