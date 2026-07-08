# ------------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2008, 2026. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ------------------------------------------------------------------------------
"""Unit tests for cplex_ce._platform_utils."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "cplex_ce"))
from _platform_utils import Arch, OS, Port, Folder, get_library_path


# ---------------------------------------------------------------------------
# PortId
# ---------------------------------------------------------------------------

class TestPortId:
    def test_match_by_canonical_name(self):
        pid = Folder("linux", "Linux")
        assert pid.match("linux")

    def test_match_by_string_alias(self):
        pid = Folder("linux", "Linux")
        assert pid.match("Linux")

    def test_match_by_tuple_alias(self):
        pid = Folder("x86_64", ("AMD64", "amd64"))
        assert pid.match("AMD64")
        assert pid.match("amd64")
        assert pid.match("x86_64")

    def test_no_match_unrelated_string(self):
        pid = Folder("linux", "Linux")
        assert not pid.match("Darwin")
        assert not pid.match("")
        assert not pid.match("LINUX")  # case-sensitive

    def test_no_alias_defaults_to_empty_tuple(self):
        pid = Folder("ppc64le")
        assert pid.platforms == ()
        assert pid.match("ppc64le")
        assert not pid.match("ppc64")

    def test_frozen_raises_on_mutation(self):
        pid = Folder("linux", "Linux")
        with pytest.raises(Exception):
            pid.name = "other"  # type: ignore[misc]

    def test_equality_by_value(self):
        assert Folder("linux", "Linux") == Folder("linux", "Linux")
        assert Folder("linux", "Linux") != Folder("osx", "Darwin")


# ---------------------------------------------------------------------------
# OS
# ---------------------------------------------------------------------------

class TestOS:
    @pytest.mark.parametrize("system,member", [
        ("Linux",   OS.LINUX),
        ("linux",   OS.LINUX),   # canonical name also matches
        ("Darwin",  OS.OSX),
        ("osx",     OS.OSX),
        ("Windows", OS.WINDOWS),
        ("windows", OS.WINDOWS),
        ("AIX",     OS.AIX),
        ("aix",     OS.AIX),
    ])
    def test_match_platform_true(self, system, member):
        with patch("platform.system", return_value=system):
            assert member.match_platform()

    @pytest.mark.parametrize("system,member", [
        ("Darwin",  OS.LINUX),
        ("Linux",   OS.OSX),
        ("Linux",   OS.WINDOWS),
        ("Windows", OS.LINUX),
        ("AIX",     OS.LINUX),
    ])
    def test_match_platform_false(self, system, member):
        with patch("platform.system", return_value=system):
            assert not member.match_platform()

    def test_all_members_present(self):
        assert {m.name for m in OS} == {"LINUX", "OSX", "WINDOWS", "AIX"}

    def test_values_are_portid_instances(self):
        for member in OS:
            assert isinstance(member.value, Folder), (
                f"OS.{member.name}.value should be a PortId, got {type(member.value)}"
            )


# ---------------------------------------------------------------------------
# Arch
# ---------------------------------------------------------------------------

class TestArch:
    @pytest.mark.parametrize("machine,member", [
        ("x86_64",  Arch.X64),
        ("AMD64",   Arch.X64),
        ("arm64",   Arch.ARM64),
        ("aarch64", Arch.ARM64),
        ("ppc64le", Arch.PPC64),
    ])
    def test_match_platform_true(self, machine, member):
        with patch("platform.machine", return_value=machine):
            assert member.match_platform()

    @pytest.mark.parametrize("machine,member", [
        ("aarch64", Arch.X64),
        ("x86_64",  Arch.ARM64),
        ("x86_64",  Arch.PPC64),
        ("ppc64le", Arch.X64),
    ])
    def test_match_platform_false(self, machine, member):
        with patch("platform.machine", return_value=machine):
            assert not member.match_platform()

    def test_all_members_present(self):
        assert {m.name for m in Arch} == {"ARM64", "PPC64", "X64"}

    def test_values_are_portid_instances(self):
        for member in Arch:
            assert isinstance(member.value, Folder), (
                f"Arch.{member.name}.value should be a PortId, got {type(member.value)}"
            )


# ---------------------------------------------------------------------------
# Port
# ---------------------------------------------------------------------------

class TestPort:
    def test_default_folder_is_arch_underscore_os_name(self):
        assert Port.X64_LINUX.folder   == "x86-64_linux"
        assert Port.ARM64_LINUX.folder == "arm64_linux"
        assert Port.PPC64_LINUX.folder == "ppc64le_linux"
        assert Port.ARM64_OSX.folder   == "arm64_osx"
        assert Port.X64_WINDOWS.folder == "x86-64_windows"

    def test_arch_and_os_attributes(self):
        assert Port.X64_LINUX.arch  == Arch.X64
        assert Port.X64_LINUX.os    == OS.LINUX
        assert Port.ARM64_OSX.arch  == Arch.ARM64
        assert Port.ARM64_OSX.os    == OS.OSX
        assert Port.PPC64_LINUX.arch == Arch.PPC64

    @pytest.mark.parametrize("system,machine,expected", [
        ("Linux",   "x86_64",  Port.X64_LINUX),
        ("Linux",   "AMD64",   Port.X64_LINUX),
        ("Linux",   "aarch64", Port.ARM64_LINUX),
        ("Linux",   "ppc64le", Port.PPC64_LINUX),
        ("Darwin",  "arm64",   Port.ARM64_OSX),
        ("Windows", "x86_64",  Port.X64_WINDOWS),
        ("Windows", "AMD64",   Port.X64_WINDOWS),
    ])
    def test_match_platform_exact_one_match(self, system, machine, expected):
        with patch("platform.system", return_value=system), \
             patch("platform.machine", return_value=machine):
            matched = [p for p in Port if p.match_platform()]
            assert matched == [expected]

    def test_unsupported_platform_no_match(self):
        with patch("platform.system", return_value="Haiku"), \
             patch("platform.machine", return_value="m68k"):
            assert [p for p in Port if p.match_platform()] == []

    def test_all_members_present(self):
        assert {m.name for m in Port} == {
            "X64_LINUX", "ARM64_LINUX", "PPC64_LINUX", "ARM64_OSX", "X64_WINDOWS"
        }


# ---------------------------------------------------------------------------
# get_library_path
# ---------------------------------------------------------------------------

class TestGetLibraryPath:
    @pytest.mark.parametrize("system,machine,expected", [
        ("Linux",   "x86_64",  "bin/x86-64_linux"),
        ("Linux",   "AMD64",   "bin/x86-64_linux"),
        ("Linux",   "aarch64", "bin/arm64_linux"),
        ("Linux",   "ppc64le", "bin/ppc64le_linux"),
        ("Darwin",  "arm64",   "bin/arm64_osx"),
        ("Windows", "x86_64",  "bin/x86-64_windows"),
        ("Windows", "AMD64",   "bin/x86-64_windows"),
    ])
    def test_returns_bin_slash_folder(self, system, machine, expected):
        with patch("platform.system", return_value=system), \
             patch("platform.machine", return_value=machine):
            assert get_library_path() == expected

    def test_unsupported_platform_raises_value_error(self):
        with patch("platform.system", return_value="Haiku"), \
             patch("platform.machine", return_value="m68k"):
            with pytest.raises(ValueError, match="Unsupported platform"):
                get_library_path()

    def test_error_message_contains_system_and_machine(self):
        with patch("platform.system", return_value="BeOS"), \
             patch("platform.machine", return_value="ppc"):
            with pytest.raises(ValueError, match="BeOS-ppc"):
                get_library_path()

    def test_return_type_is_str(self):
        with patch("platform.system", return_value="Linux"), \
             patch("platform.machine", return_value="x86_64"):
            result = get_library_path()
            assert isinstance(result, str)
            assert result.startswith("bin/")
