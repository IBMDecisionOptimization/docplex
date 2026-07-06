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

import importlib
import platform
import sys
from pathlib import Path
from sys import version_info

from cplex_native_ce._platform_utils import get_library_path
from cplex_native_ce._version import __version__


# ---------------------------------------------------------------------------
# Library path setup
# ---------------------------------------------------------------------------

def _setup_library_path() -> None:
    """Prepend the package's platform-specific native library directory to sys.path."""
    try:
        relative_lib_dir = get_library_path()
    except ValueError:
        return

    pkg_dir = Path(__file__).resolve().parent
    # Skip if native extensions are already co-located with the package
    # (i.e. installed from a platform wheel — no bin/ subdir needed).
    if any(pkg_dir.glob("*.so")) or any(pkg_dir.glob("*.pyd")) or any(pkg_dir.glob("*.dylib")):
        return

    library_path = pkg_dir / relative_lib_dir
    if not library_path.is_dir():
        return

    library_path_str = str(library_path)
    if library_path_str in sys.path:
        return

    sys.path.insert(0, library_path_str)


# Ensure the native library directory is on sys.path before importing versioned bindings.
_setup_library_path()

# Derive the versioned binding module name from the Python version and the
# CPLEX version, e.g. Python 3.14 + CPLEX 22.2.0 → "py314_cplex2220".
_cplex_tag = __version__.replace(".", "")[:4]      # "22.2.0" → "2220"
_py_tag    = f"py3{version_info[1]}"               # e.g. "py314"
_mod_name  = f"{_py_tag}_cplex{_cplex_tag}"        # e.g. "py314_cplex2220"

_SUPPORTED_PLATFORMS = ('Darwin', 'Linux', 'AIX', 'Windows', 'Microsoft')
_MIN_PYTHON = (3, 10, 0)
_MAX_PYTHON = (3, 15, 0)
_ERROR_STRING = f"CPLEX {__version__} is not compatible with this version of Python."

if platform.system() not in _SUPPORTED_PLATFORMS:
    raise Exception("The CPLEX Python API is not supported on this platform.")
if not (_MIN_PYTHON <= version_info < _MAX_PYTHON):
    raise Exception(_ERROR_STRING)

_native = importlib.import_module(_mod_name)
globals().update({k: v for k, v in vars(_native).items() if not k.startswith("_")})

# Verify the loaded native library reports the same version as _version.py.
_native_version = (
    f"{_native.CPX_VERSION_VERSION}"
    f".{_native.CPX_VERSION_RELEASE}"
    f".{_native.CPX_VERSION_MODIFICATION}"
)
if _native_version != __version__:
    raise RuntimeError(
        f"CPLEX native library version mismatch: "
        f"expected {__version__}, got {_native_version} (from {_mod_name})"
    )
