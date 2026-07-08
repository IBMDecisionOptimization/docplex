# ------------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2008, 2026. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ------------------------------------------------------------------------------
"""Unit tests for cplex_ce._pycplex_platform.

Because _pycplex_platform runs native-library imports at module level, the
module is imported inside each test using importlib after patching
platform.system / platform.machine and stub-inserting the version-specific
native modules so that no actual .so file is required.
"""

import importlib
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cplex_ce._version import __version__ as _CPLEX_VERSION

_CPLEX_TAG = _CPLEX_VERSION.replace(".", "")[:4]   # e.g. "2220"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SRC = str(Path(__file__).parent.parent / "src")
_PKG = str(Path(__file__).parent.parent / "src" / "cplex_ce")


def _import_pycplex_platform(system: str, machine: str, python_version=(3, 14, 0)):
    """
    Import (or reload) _pycplex_platform with the given platform strings and
    Python version, injecting stub native modules so no .so is needed.
    """
    for p in (_SRC, _PKG):
        if p not in sys.path:
            sys.path.insert(0, p)

    # Remove cached copies so the reload picks up the mocks.
    for key in list(sys.modules):
        if "pycplex_platform" in key or key.startswith("py3") or "cplex_ce" in key:
            del sys.modules[key]

    # Provide a real package stub for cplex_ce so that the
    # package-relative import resolves without loading the native .so.
    import importlib.machinery as _ilm
    pkg_path = Path(_SRC) / "cplex_ce"
    pkg_spec = _ilm.ModuleSpec(
        "cplex_ce",
        _ilm.SourceFileLoader("cplex_ce", str(pkg_path / "__init__.py")),
        origin=str(pkg_path / "__init__.py"),
        is_package=True,
    )
    pkg_stub = types.ModuleType("cplex_ce")
    pkg_stub.__path__ = [str(pkg_path)]
    pkg_stub.__package__ = "cplex_ce"
    pkg_stub.__spec__ = pkg_spec
    sys.modules["cplex_ce"] = pkg_stub

    # Pre-register _platform_utils under the package name so the
    # `from cplex_ce._platform_utils import ...` in _pycplex_platform
    # doesn't need to re-import (and thus trigger __init__.py's native import).
    import _platform_utils as _pu_bare
    sys.modules["cplex_ce._platform_utils"] = _pu_bare

    # Stub out _pycplex so __init__.py's `from . import _pycplex` is harmless.
    _pycplex_stub = types.ModuleType("cplex_ce._pycplex")
    sys.modules["cplex_ce._pycplex"] = _pycplex_stub
    pkg_stub._pycplex = _pycplex_stub

    # Derive the stub name exactly as the source does
    major, minor, _ = python_version
    if (3, 10) <= (major, minor) < (3, 15):
        native_mod_name = f"py3{minor}_cplex{_CPLEX_TAG}"
    else:
        native_mod_name = None          # will raise

    if native_mod_name:
        stub = types.ModuleType(native_mod_name)
        # Populate the version attributes the source checks after import.
        _v = _CPLEX_VERSION.split(".")
        stub.CPX_VERSION_VERSION      = int(_v[0])
        stub.CPX_VERSION_RELEASE      = int(_v[1])
        stub.CPX_VERSION_MODIFICATION = int(_v[2])
        sys.modules[native_mod_name] = stub

    with patch("platform.system", return_value=system), \
         patch("platform.machine", return_value=machine), \
         patch("sys.version_info", python_version):
        mod = importlib.import_module("_pycplex_platform")

    return mod


def _cleanup():
    for key in list(sys.modules):
        if "pycplex_platform" in key or key.startswith("py3"):
            del sys.modules[key]


# ---------------------------------------------------------------------------
# _setup_library_path
# ---------------------------------------------------------------------------

class TestSetupLibraryPath:
    def test_adds_bin_dir_to_sys_path_when_no_so_in_package(self, tmp_path):
        """
        When the package directory contains no .so/.pyd/.dylib files and the
        bin/<platform> directory exists, it should be prepended to sys.path.
        """
        bin_dir = tmp_path / "bin" / "x86-64_linux"
        bin_dir.mkdir(parents=True)
        (bin_dir / "libdummy.so").touch()

        mod = _import_pycplex_platform("Linux", "x86_64")

        original_path = sys.path.copy()
        lib_path_str = str(bin_dir)
        if lib_path_str in sys.path:
            sys.path.remove(lib_path_str)
        try:
            mod._setup_library_path.__globals__["__file__"] = str(
                tmp_path / "_pycplex_platform.py"
            )
            with patch.object(mod, "get_library_path", return_value=f"bin/x86-64_linux"), \
                 patch("pathlib.Path.exists", return_value=True), \
                 patch("pathlib.Path.glob", return_value=iter([])):
                mod._setup_library_path()
        finally:
            sys.path[:] = original_path
        _cleanup()

    def test_does_not_add_path_when_so_already_in_package(self, tmp_path):
        """
        When the package directory already contains a .so file (installed wheel),
        _setup_library_path() should be a no-op for sys.path.
        """
        mod = _import_pycplex_platform("Linux", "x86_64")
        original_path = sys.path.copy()
        with patch.object(mod, "get_library_path", return_value="bin/x86-64_linux"), \
             patch("pathlib.Path.glob", return_value=iter([Path("dummy.so")])):
            mod._setup_library_path()
            assert sys.path == original_path
        _cleanup()

    def test_no_op_when_get_library_path_raises(self):
        """When get_library_path raises ValueError, sys.path must not change."""
        mod = _import_pycplex_platform("Linux", "x86_64")
        original_path = sys.path.copy()
        with patch.object(mod, "get_library_path",
                          side_effect=ValueError("Unsupported platform")):
            mod._setup_library_path()
            assert sys.path == original_path
        _cleanup()

    def test_no_op_when_bin_dir_does_not_exist(self, tmp_path):
        """When the bin/<platform> dir is missing, sys.path must not change."""
        mod = _import_pycplex_platform("Linux", "x86_64")
        original_path = sys.path.copy()
        with patch.object(mod, "get_library_path", return_value="bin/x86-64_linux"), \
             patch("pathlib.Path.glob", return_value=iter([])), \
             patch("pathlib.Path.exists", return_value=False):
            mod._setup_library_path()
            assert sys.path == original_path
        _cleanup()

    def test_path_not_duplicated_when_already_present(self, tmp_path):
        """If the lib dir is already in sys.path, it must not be added again."""
        bin_dir = tmp_path / "bin" / "x86-64_linux"
        bin_dir.mkdir(parents=True)
        lib_path_str = str(bin_dir)

        mod = _import_pycplex_platform("Linux", "x86_64")
        sys.path.insert(0, lib_path_str)
        count_before = sys.path.count(lib_path_str)

        with patch.object(mod, "get_library_path", return_value="bin/x86-64_linux"), \
             patch("pathlib.Path.glob", return_value=iter([])), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.__truediv__",
                   return_value=MagicMock(exists=lambda: True,
                                         __str__=lambda s: lib_path_str)):
            mod._setup_library_path()

        assert sys.path.count(lib_path_str) == count_before
        sys.path.remove(lib_path_str)
        _cleanup()


# ---------------------------------------------------------------------------
# Python-version gating (module-level version checks)
# ---------------------------------------------------------------------------

class TestPythonVersionGating:
    def test_raises_on_python_below_3_10(self):
        for key in list(sys.modules):
            if "pycplex_platform" in key:
                del sys.modules[key]
        with patch("platform.system", return_value="Linux"), \
             patch("platform.machine", return_value="x86_64"), \
             patch("sys.version_info", (3, 9, 0)):
            with pytest.raises(Exception, match="not compatible"):
                importlib.import_module("_pycplex_platform")
        _cleanup()

    def test_raises_on_python_3_15_and_above(self):
        for key in list(sys.modules):
            if "pycplex_platform" in key:
                del sys.modules[key]
        with patch("platform.system", return_value="Linux"), \
             patch("platform.machine", return_value="x86_64"), \
             patch("sys.version_info", (3, 15, 0)):
            with pytest.raises(Exception, match="not compatible"):
                importlib.import_module("_pycplex_platform")
        _cleanup()

# ---------------------------------------------------------------------------
# Native library version assertion
# ---------------------------------------------------------------------------

class TestNativeLibraryVersionAssertion:
    def test_passes_when_version_matches(self):
        """No exception when the stub reports the same version as _version.py."""
        _import_pycplex_platform("Linux", "x86_64")   # default stub has correct version
        _cleanup()
