"""
Setup script for cplex-native-ce with platform-specific library filtering.

This setup.py imports the platform detection logic from _platform_utils
to ensure consistency between build-time and runtime platform detection.
"""

import sys
from pathlib import Path
from setuptools import setup

# Import the shared platform detection function directly
# We can't import through cplex_ce package because __init__.py
# tries to load the native libraries which don't exist yet during setup
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'cplex_ce'))
from _platform_utils import get_library_path


def get_platform_libs():
    """
    Get list of library files for the current platform.

    Returns glob patterns relative to the package directory
    (src/cplex_ce/) so setuptools includes them as package data.
    """
    library_path = get_library_path()

    # Absolute path to the platform library directory
    libs_path = Path(__file__).parent / 'src' / 'cplex_ce' / library_path

    if not libs_path.exists():
        print(f"Warning: Library directory not found: {libs_path}")
        return []

    lib_files = [
        lib_file
        for lib_file in libs_path.glob('*')
        if lib_file.suffix in ('.so', '.pyd', '.dylib')
    ]

    if lib_files:
        print(f"Including libraries from {libs_path}:")
        for lib_file in lib_files:
            print(f"  - {lib_file.name}")

    # Return paths relative to src/cplex_ce/ for package_data
    pkg_root = Path(__file__).parent / 'src' / 'cplex_ce'
    return [str(f.relative_to(pkg_root)) for f in lib_files]


setup(
    package_data={
        "cplex_ce": get_platform_libs()
    },
)

# Made with Bob
