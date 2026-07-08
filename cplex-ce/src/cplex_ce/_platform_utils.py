# ------------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2008, 2026. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ------------------------------------------------------------------------------
"""Platform detection utilities shared between setup.py and runtime."""

import platform
from dataclasses import dataclass
from enum import Enum
from typing import Union, Tuple


@dataclass(frozen=True)
class Folder:
    """Canonical identifier for an OS or architecture token.

    Holds a *canonical* folder-name fragment (e.g. ``'linux'``, ``'x86-64'``)
    together with the aliases that the ``platform`` module may return for the same
    system (e.g. ``'Linux'``, ``('AMD64', 'x86_64')``).

    Attributes:
        name: The canonical fragment used when building directory names.
        platforms: A single alias string or a tuple of alias strings accepted
            by ``match()``.  Defaults to an empty tuple (no aliases).
    """

    name: str
    platforms: str | Tuple[str, ...] = ()

    def match(self, platform_str: str) -> bool:
        """Return ``True`` if *platform_str* equals the canonical name or any alias."""
        if isinstance(self.platforms, tuple):
            return platform_str == self.name or platform_str in self.platforms
        return platform_str in (self.name, self.platforms)


class OS(Enum):
    """Supported operating systems.

    Each member's value is a ``Folder`` whose ``name`` is the lowercase
    fragment used in CPLEX directory names (e.g. ``'linux'``) and whose
    ``platforms`` alias matches the string returned by ``platform.system()``.
    """

    LINUX   = Folder('linux',   'Linux')
    OSX     = Folder('osx',     'Darwin')
    WINDOWS = Folder('windows', 'Windows')
    AIX     = Folder('aix',     'AIX')

    def match_platform(self) -> bool:
        """Return ``True`` if the current OS matches this member."""
        return self.value.match(platform.system())


class Arch(Enum):
    """Supported CPU architectures.

    Each member's value is a ``Folder`` whose ``name`` is the lowercase
    fragment used in CPLEX directory names (e.g. ``'x86-64'``) and whose
    ``platforms`` aliases match the strings returned by ``platform.machine()``.
    """

    ARM64 = Folder('arm64',  'aarch64')
    PPC64 = Folder('ppc64le')
    X64   = Folder('x86-64', ('AMD64', 'x86_64'))

    def match_platform(self) -> bool:
        """Return ``True`` if the current CPU architecture matches this member."""
        return self.value.match(platform.machine())


class Port(Enum):
    """Known (arch, OS) combinations that CPLEX ships native libraries for.

    The ``folder`` property returns the CPLEX-style directory name for the
    port, formed as ``<arch.value.name>_<os.value.name>``
    (e.g. ``'x86-64_linux'``, ``'arm64_osx'``).  An explicit override can be
    supplied as the optional third constructor argument for exotic ports not
    following this convention.
    """

    X64_LINUX   = (Arch.X64,   OS.LINUX)
    ARM64_LINUX = (Arch.ARM64, OS.LINUX)
    PPC64_LINUX = (Arch.PPC64, OS.LINUX)
    ARM64_OSX   = (Arch.ARM64, OS.OSX)
    X64_WINDOWS = (Arch.X64,   OS.WINDOWS)

    def __init__(self, arch: Arch, os: OS, folder: str | None = None):
        self.arch = arch
        self.os = os
        self._folder = folder

    @property
    def folder(self) -> str:
        """CPLEX native-library sub-directory name for this port.

        Returns the explicit override if one was provided, otherwise derives
        the name as ``'<arch>_<os>'`` (e.g. ``'x86-64_linux'``).
        """
        if self._folder is not None:
            return self._folder
        return f'{self.arch.value.name}_{self.os.value.name}'

    def match_platform(self) -> bool:
        """Return ``True`` if both the arch and OS match the current platform."""
        return self.arch.match_platform() and self.os.match_platform()


def get_library_path() -> str:
    """Return the ``bin/<folder>`` path for the current platform.

    Iterates over all ``Port`` members and returns the path for the first
    one that matches the running platform.

    Returns:
        A relative path string such as ``'bin/x86-64_linux'``.

    Raises:
        ValueError: If no ``Port`` member matches the current platform.
    """
    port: Union[Port, None] = next((p for p in Port if p.match_platform()), None)
    if port is None:
        raise ValueError(f'Unsupported platform: {platform.system()}-{platform.machine()}')
    return f'bin/{port.folder}'
