"""
Setup script for cplex-native-ce with platform-specific library filtering.
"""

import platform
from pathlib import Path
from setuptools import setup


# Platform to library directory mapping
PLATFORM_LIB_DIRS = {
    'manylinux_2_17_x86_64': 'x86-64_linux',
    'manylinux_2_17_aarch64': 'aarch64_linux',
    'manylinux_2_17_ppc64le': 'ppc64le_linux',
    'linux_x86_64': 'x86-64_linux',
    'linux_aarch64': 'aarch64_linux',
    'linux_ppc64le': 'ppc64le_linux',
    'macosx_10_9_x86_64': 'x86-64_osx',
    'macosx_11_0_arm64': 'arm64_osx',
    'win_amd64': 'x86-64_windows',
    'win32': 'x86_windows',
    'aix_7_2_ppc64': 'ppc64_aix',
}


def get_platform_lib_dir():
    """Determine which library directory to use based on the current platform."""
    system = platform.system()
    machine = platform.machine()
    
    if system == 'Linux':
        if machine in ('x86_64', 'AMD64'):
            return 'x86-64_linux'
        elif machine == 'aarch64':
            return 'aarch64_linux'
        elif machine == 'ppc64le':
            return 'ppc64le_linux'
    elif system == 'Darwin':
        if machine == 'arm64':
            return 'arm64_osx'
        else:
            return 'x86-64_osx'
    elif system == 'Windows':
        if machine in ('AMD64', 'x86_64'):
            return 'x86-64_windows'
        else:
            return 'x86_windows'
    elif system == 'AIX':
        return 'ppc64_aix'
    
    return None


def get_platform_libs():
    """
    Get list of library files for the current platform.
    
    Returns paths relative to the src/ directory so setuptools can find them
    in libs/<platform>/ and copy them into the cplex_native_ce package.
    """
    lib_dir = get_platform_lib_dir()
    
    if not lib_dir:
        print(f"Warning: Could not determine library directory for platform")
        return []
    
    # Path relative to project root
    libs_path = Path('libs') / lib_dir
    
    if not libs_path.exists():
        print(f"Warning: Library directory not found: {libs_path}")
        return []
    
    # Return paths relative to src/ directory (where cplex_native_ce package is)
    # This tells setuptools: "copy these files from ../../libs/<platform>/ into the package"
    lib_files = []
    for lib_file in libs_path.glob('*'):
        if lib_file.suffix in ('.so', '.pyd', '.dylib'):
            # Path from src/cplex_native_ce/ to libs/<platform>/<file>
            relative_path = f"../../../libs/{lib_dir}/{lib_file.name}"
            lib_files.append(relative_path)
    
    if lib_files:
        print(f"Including libraries from {lib_dir}:")
        for lib_file in lib_files:
            print(f"  - {Path(lib_file).name}")
    
    return lib_files


setup(
    package_data={
        "cplex_native_ce": get_platform_libs()
    },
)

# Made with Bob
