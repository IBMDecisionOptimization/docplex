# ------------------------------------------------------------------------
# File: __init__.py
# ------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2008, 2026. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ------------------------------------------------------------------------
"""Error codes and Exceptions raised by the CPLEX Python API.

For documentation of CPLEX error codes, see the group
optim.cplex.errorcodes in the reference manual of the CPLEX Callable
Library, and the topic Interpreting Error Codes in the Overview of the
APIs.
"""
from . import error_codes  # noqa: F401
from .errors import (CplexError, CplexSolverError,  # noqa: F401
                     WrongNumberOfArgumentsError,
                     ErrorChannelMessage)

__all__ = ["CplexError", "CplexSolverError", "WrongNumberOfArgumentsError", "ErrorChannelMessage"]
