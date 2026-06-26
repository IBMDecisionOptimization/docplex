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

from sys import version_info

ERROR_STRING = "CPLEX 22.2.0.0 is not compatible with this version of Python."

if platform.system() in ('Darwin', 'Linux', 'AIX', 'Windows', 'Microsoft'):
    if version_info < (3, 10, 0):
        raise Exception(ERROR_STRING)
    elif version_info < (3, 11, 0):
        from .py310_cplex2220 import *
    elif version_info < (3, 12, 0):
        from .py311_cplex2220 import *
    elif version_info < (3, 13, 0):
        from .py312_cplex2220 import *
    elif version_info < (3, 14, 0):
        from .py313_cplex2220 import *
    elif version_info < (3, 15, 0):
        from .py314_cplex2220 import *
    else:
        raise Exception(ERROR_STRING)
else:
    raise Exception("The CPLEX Python API is not supported on this platform.")
