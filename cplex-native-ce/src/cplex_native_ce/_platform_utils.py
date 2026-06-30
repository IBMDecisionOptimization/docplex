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
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Union, Tuple

@dataclass(frozen=True)
class PortId:
    name: str
    platforms: str | Tuple[str, ...] = ()
    def match(self, platform_str: str) -> bool:
        if isinstance(self.platforms, tuple):
            return platform_str == self.name or platform_str in self.platforms
        return platform_str in (self.name, self.platforms)


class OS(Enum):
    LINUX   = PortId('linux',   'Linux')
    OSX     = PortId('osx',     'Darwin')
    WINDOWS = PortId('windows', 'Windows')
    AIX     = PortId('aix',     'AIX')
    def match_platform(self) -> bool:
        return self.value.match(platform.system())


class Arch(Enum):
    ARM64 = PortId('arm64',   'aarch64')
    PPC64 = PortId('ppc64le')
    X64   = PortId('x86-64',  ('AMD64', 'x86_64'))

    def match_platform(self) -> bool:
        return self.value.match(platform.machine())


class Port(Enum):

    X64_LINUX = (Arch.X64, OS.LINUX)
    ARM64_LINUX = (Arch.ARM64, OS.LINUX)
    PPC64_LINUX = (Arch.PPC64, OS.LINUX)
    ARM64_OSX = (Arch.ARM64, OS.OSX)
    X64_WINDOWS = (Arch.X64, OS.WINDOWS)

    def __init__(self, arch: Arch, os: OS, path: str|None = None):
        self.arch = arch
        self.os = os
        self.folder: str = path if path is not None else f'{self.arch.value.name}_{self.os.value.name}'

    def match_platform(self) -> bool:
        return self.arch.match_platform() and self.os.match_platform()

    def get_folder(self) -> str:
        return self.folder


def get_library_path() -> str:
    port : Union[Port|None] = next((p for p in Port if p.match_platform()), None)
    if port is None:
        raise ValueError(f'Unsupported platform: {platform.system()}-{platform.machine()}')
    print("### Library path:", f'bin/{port.folder}', file=sys.stderr)
    return f'bin/{port.folder}'
